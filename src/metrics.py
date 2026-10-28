"""AurumEdge AI - Financial Risk, Performance & Transaction Cost Metrics.

Calculates institutional performance metrics, drawdowns, Sharpe/Sortino ratios,
and full Indian MCX statutory transaction costs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_mcx_transaction_costs(
    turnover_buy: float,
    turnover_sell: float,
    slippage_per_leg: float = 1.0,
    num_legs: int = 2,
    brokerage_pct: float = 0.0002,
    exchange_pct: float = 0.000021,
    sebi_pct: float = 0.000001,
    ctt_sell_pct: float = 0.0001,
    stamp_duty_buy_pct: float = 0.00002,
    gst_pct: float = 0.18,
) -> dict[str, float]:
    """Calculate detailed transaction costs and statutory levies for MCX trades.

    Levies based on official MCX tariff circulars:
      - Brokerage: default 2 bps (0.02%)
      - Exchange turnover fee: 0.0021%
      - SEBI turnover charge: ₹10 per crore (0.0001%)
      - CTT (Commodity Transaction Tax): 0.01% on non-agri sell turnover
      - Stamp duty: 0.002% on buy turnover
      - GST: 18% on (brokerage + exchange fee)
      - Slippage: estimated slippage cost in ₹
    """
    total_turnover = turnover_buy + turnover_sell

    brokerage = total_turnover * brokerage_pct
    exchange_fee = total_turnover * exchange_pct
    sebi_fee = total_turnover * sebi_pct
    ctt = turnover_sell * ctt_sell_pct
    stamp_duty = turnover_buy * stamp_duty_buy_pct
    gst = (brokerage + exchange_fee) * gst_pct
    slippage = slippage_per_leg * num_legs

    total_cost = brokerage + exchange_fee + sebi_fee + ctt + stamp_duty + gst + slippage

    return {
        "turnover_buy": turnover_buy,
        "turnover_sell": turnover_sell,
        "total_turnover": total_turnover,
        "brokerage": brokerage,
        "exchange_fee": exchange_fee,
        "sebi_fee": sebi_fee,
        "ctt": ctt,
        "stamp_duty": stamp_duty,
        "gst": gst,
        "slippage": slippage,
        "total_cost": total_cost,
    }


def compute_max_drawdown(equity_series: pd.Series) -> tuple[float, float, pd.Series]:
    """Calculate maximum absolute drawdown, maximum percentage drawdown, and drawdown series.

    Returns:
        (max_dd_absolute, max_dd_pct, drawdown_series)
    """
    if equity_series.empty or len(equity_series) < 2:
        return 0.0, 0.0, pd.Series(dtype=float)

    running_max = equity_series.cummax()
    drawdown = equity_series - running_max
    drawdown_pct = (drawdown / running_max.replace(0, np.nan)).fillna(0.0)

    max_dd_abs = float(drawdown.min())
    max_dd_pct = float(drawdown_pct.min())

    return max_dd_abs, max_dd_pct, drawdown_pct


def compute_sharpe_ratio(
    daily_returns: pd.Series,
    annual_risk_free_rate: float = 0.065,
    trading_days_per_year: int = 252,
) -> float:
    """Compute annualized Sharpe ratio using daily returns."""
    clean = daily_returns.dropna()
    if len(clean) < 5 or clean.std() == 0:
        return 0.0

    daily_rf = (1.0 + annual_risk_free_rate) ** (1.0 / trading_days_per_year) - 1.0
    excess_returns = clean - daily_rf
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std(ddof=1)

    if std_excess == 0 or np.isnan(std_excess):
        return 0.0

    annualized_sharpe = float((mean_excess / std_excess) * np.sqrt(trading_days_per_year))
    return annualized_sharpe


def compute_sortino_ratio(
    daily_returns: pd.Series,
    annual_risk_free_rate: float = 0.065,
    trading_days_per_year: int = 252,
) -> float:
    """Compute annualized Sortino ratio using downside standard deviation."""
    clean = daily_returns.dropna()
    if len(clean) < 5:
        return 0.0

    daily_rf = (1.0 + annual_risk_free_rate) ** (1.0 / trading_days_per_year) - 1.0
    excess_returns = clean - daily_rf
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) < 2 or downside_returns.std() == 0:
        return 0.0

    downside_std = downside_returns.std(ddof=1)
    return float((excess_returns.mean() / downside_std) * np.sqrt(trading_days_per_year))


def summarize_trade_performance(trades_df: pd.DataFrame, initial_capital: float = 1_000_000.0) -> dict[str, Any]:
    """Calculate summary statistics over completed trades."""
    if trades_df.empty:
        return {
            "total_trades": 0,
            "profitable_trades": 0,
            "loss_making_trades": 0,
            "win_rate_pct": 0.0,
            "gross_pnl": 0.0,
            "net_pnl": 0.0,
            "total_costs": 0.0,
            "cost_drag_pct": 0.0,
            "avg_trade_pnl": 0.0,
            "avg_holding_days": 0.0,
            "return_on_capital_pct": 0.0,
        }

    total_trades = len(trades_df)
    profitable = int((trades_df["net_pnl"] > 0).sum())
    loss_making = int((trades_df["net_pnl"] < 0).sum())
    win_rate = (profitable / total_trades) * 100.0 if total_trades > 0 else 0.0

    gross_pnl = float(trades_df["gross_pnl"].sum())
    net_pnl = float(trades_df["net_pnl"].sum())
    total_costs = float(trades_df["total_cost"].sum())
    cost_drag = (total_costs / abs(gross_pnl) * 100.0) if gross_pnl != 0 else 0.0
    avg_pnl = float(trades_df["net_pnl"].mean())
    avg_holding = (
        float(trades_df["holding_days"].mean()) if "holding_days" in trades_df.columns else 0.0
    )

    return {
        "total_trades": total_trades,
        "profitable_trades": profitable,
        "loss_making_trades": loss_making,
        "win_rate_pct": win_rate,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "total_costs": total_costs,
        "cost_drag_pct": cost_drag,
        "avg_trade_pnl": avg_pnl,
        "avg_holding_days": avg_holding,
        "return_on_capital_pct": (net_pnl / initial_capital) * 100.0,
    }
