"""AurumEdge AI - Historical Research Backtesting Engine.

Provides:
  1. Signal Study Mode: Forward mean-reversion analysis without execution assumptions.
  2. Executable Strategy Mode: Walk-forward chronologically ordered simulation with
     strict next-bar (t+1) execution, lot-size alignment, MCX transaction costs,
     stop losses, and delivery tender buffers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import (
    compute_max_drawdown,
    compute_mcx_transaction_costs,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    summarize_trade_performance,
)


@dataclass
class BacktestResult:
    """Complete result container for strategy backtest."""

    pair_id: str
    mode: str  # 'STRATEGY' or 'SIGNAL_STUDY'
    initial_capital: float = 1_000_000.0
    final_capital: float = 1_000_000.0
    total_net_pnl: float = 0.0
    total_gross_pnl: float = 0.0
    total_costs: float = 0.0
    total_trades: int = 0
    win_rate_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    benchmark_return_pct: float = 0.0
    strategy_return_pct: float = 0.0
    trade_ledger: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    signal_study_results: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "mode": self.mode,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_net_pnl": self.total_net_pnl,
            "total_gross_pnl": self.total_gross_pnl,
            "total_costs": self.total_costs,
            "total_trades": self.total_trades,
            "win_rate_pct": self.win_rate_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "benchmark_return_pct": self.benchmark_return_pct,
            "strategy_return_pct": self.strategy_return_pct,
            "warnings": self.warnings,
        }


def run_signal_study(
    signals_df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> dict[str, Any]:
    """Perform forward mean-reversion analysis on candidate anomalies without execution modeling.

    For each horizon H, measures whether the spread moved back toward the rolling mean.
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]
    if signals_df.empty or "spread" not in signals_df.columns:
        return {"error": "Insufficient data for signal study."}

    df = signals_df.copy()
    anomalies = df[df["is_candidate_anomaly"]].copy()

    if anomalies.empty:
        return {"message": "No candidate anomalies found in the dataset to study."}

    results_by_horizon = {}
    for h in horizons:
        df[f"spread_fwd_{h}"] = df["spread"].shift(-h)
        # Expected direction: if z > 0, spread should decrease (fwd - cur < 0)
        # if z < 0, spread should increase (fwd - cur > 0)
        # Reversion score = -sign(z) * (spread_fwd - spread)
        sub = df.loc[anomalies.index].copy()
        sub = sub.dropna(subset=[f"spread_fwd_{h}"])
        if sub.empty:
            continue

        reversion = -np.sign(sub["z_score"]) * (sub[f"spread_fwd_{h}"] - sub["spread"])
        pct_reverted = (reversion > 0).mean() * 100.0
        avg_reversion_bps = float(reversion.mean() * 10000.0)

        results_by_horizon[f"{h}_day"] = {
            "eval_count": len(sub),
            "reversion_rate_pct": round(float(pct_reverted), 2),
            "avg_reversion_bps": round(avg_reversion_bps, 2),
        }

    return {
        "total_anomalies_evaluated": len(anomalies),
        "horizon_reversion_results": results_by_horizon,
    }


def run_pairs_backtest(
    signals_df: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_loss_z: float = 3.5,
    max_holding_bars: int = 15,
    target_exposure_grams: float = 100.0,
    cost_config: dict[str, float] | None = None,
) -> BacktestResult:
    """Execute walk-forward chronological pairs backtest with t+1 execution.

    Parameters:
        signals_df: DataFrame output from src.signals.generate_relative_value_signals.
        initial_capital: Starting portfolio cash (INR).
        entry_z: Z-score threshold for opening relative value position.
        exit_z: Z-score threshold for closing position on convergence.
        stop_loss_z: Z-score threshold for emergency stop-loss.
        max_holding_bars: Maximum trading bars to maintain position.
        target_exposure_grams: Baseline pure gold exposure in grams (e.g. 100g).
        cost_config: Transaction cost parameters.

    Returns:
        BacktestResult
    """
    pair_id = signals_df["pair_id"].iloc[0] if not signals_df.empty else "UNKNOWN"
    res = BacktestResult(pair_id=pair_id, mode="STRATEGY", initial_capital=initial_capital)

    if signals_df.empty or len(signals_df) < 5:
        res.warnings.append("Insufficient data bars (< 5) to conduct backtest.")
        return res

    df = signals_df.sort_values("trading_date").reset_index(drop=True).copy()

    # Determine lot multipliers to balance exposure grams
    tu_a = float(df["trading_unit_A"].iloc[0])
    tu_b = float(df["trading_unit_B"].iloc[0])

    lots_a = max(1, round(target_exposure_grams / tu_a))
    lots_b = max(1, round(target_exposure_grams / tu_b))

    trades = []
    in_position = False
    pos_direction = 0  # +1: Long A / Short B (A cheap, B rich); -1: Short A / Long B (A rich, B cheap)
    entry_idx = 0
    entry_row = None

    # Track daily portfolio value
    [df["trading_date"].iloc[0]]

    # Step through bars chronologically
    # Signal generated at bar i close -> Order executed at bar i+1
    for i in range(len(df) - 1):
        cur_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        # 1. Manage existing position
        if in_position:
            holding_bars = i - entry_idx
            should_exit = False
            exit_reason = ""

            current_z = cur_row["z_score"]

            # Exit Condition A: Convergence
            if (pos_direction == -1 and current_z <= exit_z) or (
                pos_direction == 1 and current_z >= -exit_z
            ):
                should_exit = True
                exit_reason = f"Convergence reached (Z: {current_z:.2f})"

            # Exit Condition B: Stop loss
            elif abs(current_z) >= stop_loss_z:
                should_exit = True
                exit_reason = f"Stop-loss triggered (|Z| >= {stop_loss_z:.1f})"

            # Exit Condition C: Near Expiry / Tender period
            elif cur_row["days_to_expiry"] <= cur_row["expiry_buffer_days"]:
                should_exit = True
                exit_reason = "Tender buffer reached (forced expiry exit)"

            # Exit Condition D: Max holding period
            elif holding_bars >= max_holding_bars:
                should_exit = True
                exit_reason = f"Max holding period ({max_holding_bars} bars) reached"

            # Exit Condition E: Final bar of available data
            elif i == len(df) - 2:
                should_exit = True
                exit_reason = "End of historical data sample"

            if should_exit:
                # Fulfill exit at next bar (i+1)
                exit_price_a = next_row["raw_price_A"]
                exit_price_b = next_row["raw_price_B"]

                # PnL Calculation:
                # Leg A: pos_direction * lots_a * (exit_A - entry_A) * (trading_unit_A / quotation_unit_A)
                mult_a = lots_a * (tu_a / float(entry_row["quotation_unit_A"]))
                mult_b = lots_b * (tu_b / float(entry_row["quotation_unit_B"]))

                # When pos_direction == 1: Long A, Short B
                pnl_a = pos_direction * (exit_price_a - entry_row["entry_price_A"]) * mult_a
                pnl_b = -pos_direction * (exit_price_b - entry_row["entry_price_B"]) * mult_b
                gross_trade_pnl = pnl_a + pnl_b

                # Turnovers for cost calculation
                entry_turnover_a = entry_row["entry_price_A"] * mult_a
                entry_turnover_b = entry_row["entry_price_B"] * mult_b
                exit_turnover_a = exit_price_a * mult_a
                exit_turnover_b = exit_price_b * mult_b

                if pos_direction == 1:
                    # Entry: Buy A, Sell B. Exit: Sell A, Buy B.
                    buy_turnover = entry_turnover_a + exit_turnover_b
                    sell_turnover = entry_turnover_b + exit_turnover_a
                else:
                    # Entry: Sell A, Buy B. Exit: Buy A, Sell B.
                    buy_turnover = entry_turnover_b + exit_turnover_a
                    sell_turnover = entry_turnover_a + exit_turnover_b

                costs_dict = compute_mcx_transaction_costs(
                    turnover_buy=buy_turnover,
                    turnover_sell=sell_turnover,
                    num_legs=4,
                    **(cost_config or {}),
                )
                net_trade_pnl = gross_trade_pnl - costs_dict["total_cost"]

                trades.append(
                    {
                        "trade_id": len(trades) + 1,
                        "entry_date": entry_row["exec_date"],
                        "exit_date": next_row["trading_date"],
                        "direction": "LONG A / SHORT B" if pos_direction == 1 else "SHORT A / LONG B",
                        "holding_days": holding_bars + 1,
                        "entry_z": entry_row["entry_z"],
                        "exit_z": current_z,
                        "lots_A": lots_a,
                        "lots_B": lots_b,
                        "pnl_leg_A": pnl_a,
                        "pnl_leg_B": pnl_b,
                        "gross_pnl": gross_trade_pnl,
                        "total_cost": costs_dict["total_cost"],
                        "net_pnl": net_trade_pnl,
                        "exit_reason": exit_reason,
                    }
                )

                in_position = False
                pos_direction = 0

        # 2. Check for new entry signal (only if currently flat)
        elif not in_position:
            # Check eligibility
            if cur_row["signal_status"] == "ELIGIBLE_SIGNAL":
                sig_z = cur_row["z_score"]
                # Enter at next bar (i+1)
                direction = -1 if sig_z >= entry_z else (+1 if sig_z <= -entry_z else 0)
                if direction != 0:
                    in_position = True
                    pos_direction = direction
                    entry_idx = i + 1
                    entry_row = {
                        "exec_date": next_row["trading_date"],
                        "entry_price_A": next_row["raw_price_A"],
                        "entry_price_B": next_row["raw_price_B"],
                        "quotation_unit_A": next_row["quotation_unit_A"],
                        "quotation_unit_B": next_row["quotation_unit_B"],
                        "entry_z": sig_z,
                    }

    # Aggregate trade results
    trades_df = pd.DataFrame(trades)
    perf_summary = summarize_trade_performance(trades_df, initial_capital=initial_capital)

    res.total_trades = perf_summary["total_trades"]
    res.win_rate_pct = perf_summary["win_rate_pct"]
    res.total_gross_pnl = perf_summary["gross_pnl"]
    res.total_net_pnl = perf_summary["net_pnl"]
    res.total_costs = perf_summary["total_costs"]
    res.final_capital = initial_capital + res.total_net_pnl
    res.strategy_return_pct = (res.total_net_pnl / initial_capital) * 100.0
    res.trade_ledger = trades_df

    # Construct cumulative equity series
    equity_curve = pd.DataFrame(
        {"trading_date": df["trading_date"], "portfolio_value": initial_capital}
    )
    if not trades_df.empty:
        # Step equity at each trade exit
        cum_pnl = 0.0
        pnl_map = {row["exit_date"]: row["net_pnl"] for _, row in trades_df.iterrows()}
        values = []
        for d in equity_curve["trading_date"]:
            if d in pnl_map:
                cum_pnl += pnl_map[d]
            values.append(initial_capital + cum_pnl)
        equity_curve["portfolio_value"] = values

    # Drawdown & Risk ratios
    equity_series = equity_curve["portfolio_value"]
    _, max_dd_pct, dd_series = compute_max_drawdown(equity_series)
    res.max_drawdown_pct = max_dd_pct * 100.0
    equity_curve["drawdown_pct"] = dd_series * 100.0
    res.equity_curve = equity_curve

    daily_returns = equity_series.pct_change().fillna(0.0)
    res.sharpe_ratio = compute_sharpe_ratio(daily_returns)
    res.sortino_ratio = compute_sortino_ratio(daily_returns)

    # Benchmark: Buy and Hold Reference Contract A
    p_start = df["raw_price_A"].iloc[0]
    p_end = df["raw_price_A"].iloc[-1]
    res.benchmark_return_pct = ((p_end - p_start) / p_start) * 100.0 if p_start > 0 else 0.0

    # Also compute signal study diagnostics
    res.signal_study_results = run_signal_study(df)

    if res.total_trades < 5:
        res.warnings.append(
            f"Small sample size warning: Only {res.total_trades} trades executed. "
            "Results lack statistical significance."
        )

    if res.total_net_pnl < 0 and res.total_gross_pnl > 0:
        res.warnings.append(
            "Transaction Cost Drag Warning: Gross strategy was positive, "
            "but exchange levies, CTT, and slippage resulted in negative net PnL."
        )

    return res
