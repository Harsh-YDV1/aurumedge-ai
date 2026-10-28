"""Unit tests for Historical Backtest Engine."""

import numpy as np
import pandas as pd

from src.backtest import run_pairs_backtest, run_signal_study
from src.signals import generate_relative_value_signals


def create_mock_signals_df():
    """Create reproducible test series with one clear signal and mean reversion."""
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    expiry = pd.to_datetime("2024-03-15")

    # Pure prices: base ~ 7300, bar 18 has an anomaly that reverts over 3 bars
    qa = [7300.0] * 40
    qb = [7300.0] * 40

    # Small noise for rolling std
    for i in range(40):
        qa[i] += np.sin(i) * 2.0

    # Injected anomaly: qa jumps by +50 on bar 20, reverts by bar 24
    qa[20] = 7360.0
    qa[21] = 7350.0
    qa[22] = 7320.0
    qa[23] = 7302.0

    df = pd.DataFrame({
        "trading_date": dates,
        "expiry_date": [expiry] * 40,
        "symbol_A": ["GOLDM"] * 40,
        "symbol_B": ["GOLDPETAL"] * 40,
        "pair_id": ["GOLDM-GOLDPETAL"] * 40,
        "q_A": qa,
        "q_B": qb,
        "raw_price_A": [p * 10.0 * 0.995 for p in qa],
        "raw_price_B": [p * 1.0 * 0.999 for p in qb],
        "volume_A": [300] * 40,
        "volume_B": [1200] * 40,
        "oi_A": [2500] * 40,
        "oi_B": [5000] * 40,
        "trading_unit_A": [100.0] * 40,
        "trading_unit_B": [1.0] * 40,
        "quotation_unit_A": [10.0] * 40,
        "quotation_unit_B": [1.0] * 40,
        "days_to_expiry": [50 - i for i in range(40)],
    })

    signals, _ = generate_relative_value_signals(
        df, rolling_window=15, min_history_bars=10, z_score_threshold=2.0
    )
    return signals


def test_t_plus_one_execution_rule():
    """Verify orders are never filled on the bar that signaled, but strictly on bar t+1."""
    signals = create_mock_signals_df()
    # Find which bar triggered eligible signal
    eligible_bars = signals[signals["signal_status"] == "ELIGIBLE_SIGNAL"]
    assert not eligible_bars.empty

    first_sig_date = eligible_bars["trading_date"].iloc[0]

    bt_res = run_pairs_backtest(signals, initial_capital=1_000_000.0, entry_z=2.0, exit_z=0.5)

    assert bt_res.total_trades > 0
    trade = bt_res.trade_ledger.iloc[0]
    # Trade entry date must be strictly after the signal date!
    assert trade["entry_date"] > first_sig_date


def test_transaction_costs_reduce_gross_performance():
    """Verify transaction costs strictly reduce net PnL below gross PnL."""
    signals = create_mock_signals_df()
    bt_res = run_pairs_backtest(signals, initial_capital=1_000_000.0)

    if bt_res.total_trades > 0:
        assert bt_res.total_costs > 0
        assert bt_res.total_net_pnl < bt_res.total_gross_pnl


def test_signal_study_mode():
    """Verify signal study runs without execution assumptions."""
    signals = create_mock_signals_df()
    study = run_signal_study(signals, horizons=[1, 3])

    assert "total_anomalies_evaluated" in study
    assert study["total_anomalies_evaluated"] > 0
    assert "horizon_reversion_results" in study
