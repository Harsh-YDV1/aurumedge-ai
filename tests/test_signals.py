"""Unit tests for Relative-Value Signal Engine."""

import numpy as np
import pandas as pd
import pytest

from src.signals import (
    compute_point_in_time_statistics,
    generate_relative_value_signals,
)


def test_zero_look_ahead_bias():
    """Verify rolling mean and std strictly exclude observation bar t."""
    spread = pd.Series([0.01 * i for i in range(1, 21)])
    roll_mean, _roll_std = compute_point_in_time_statistics(spread, window=5, min_periods=3)

    # For index 3 (4th item, value=0.04):
    # Shifted values are: NaN, 0.01, 0.02, 0.03
    # The rolling mean at index 3 must be the average of (0.01, 0.02, 0.03) = 0.02!
    # It must NOT include the value at index 3 (0.04)!
    assert pytest.approx(roll_mean.iloc[3], abs=1e-6) == 0.02
    assert roll_mean.iloc[3] != np.mean(spread.iloc[0:4])


def test_anomaly_detection_under_injected_dislocation():
    """Verify that a controlled dislocation triggers candidate anomaly."""
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    expiry = pd.to_datetime("2024-03-01")

    # Baseline flat spread: q_A == q_B == 7300.0
    q_a = [7300.0] * 30
    q_b = [7300.0] * 30

    # Inject a 1% jump on bar 25 for q_A
    q_a[24] = 7373.0  # +1% jump

    df = pd.DataFrame({
        "trading_date": dates,
        "expiry_date": [expiry] * 30,
        "symbol_A": ["GOLDM"] * 30,
        "symbol_B": ["GOLDPETAL"] * 30,
        "pair_id": ["GOLDM-GOLDPETAL"] * 30,
        "q_A": q_a,
        "q_B": q_b,
        "raw_price_A": [72500.0] * 30,
        "raw_price_B": [7300.0] * 30,
        "volume_A": [200] * 30,
        "volume_B": [800] * 30,
        "oi_A": [1000] * 30,
        "oi_B": [3000] * 30,
        "trading_unit_A": [100.0] * 30,
        "trading_unit_B": [1.0] * 30,
        "quotation_unit_A": [10.0] * 30,
        "quotation_unit_B": [1.0] * 30,
        "days_to_expiry": [40 - i for i in range(30)],
    })

    # Add small historical noise so std is non-zero
    np.random.seed(42)
    df["q_B"] = df["q_B"] + np.random.normal(0, 0.5, 30)

    signals, _summary = generate_relative_value_signals(
        df, rolling_window=15, min_history_bars=10, z_score_threshold=2.0
    )

    # Check that at bar 25, an anomaly was detected
    assert bool(signals.loc[24, "is_candidate_anomaly"]) is True
    assert abs(signals.loc[24, "z_score"]) >= 2.0
    assert signals.loc[24, "signal_status"] == "ELIGIBLE_SIGNAL"
    assert "ELIGIBLE ANOMALY" in signals.loc[24, "explanation"]


def test_liquidity_filter_disqualifies_anomaly():
    """Verify that an anomaly occurring during illiquid session is filtered."""
    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    df = pd.DataFrame({
        "trading_date": dates,
        "expiry_date": [pd.to_datetime("2024-03-01")] * 25,
        "symbol_A": ["GOLDM"] * 25,
        "symbol_B": ["GOLDGUINEA"] * 25,
        "pair_id": ["GOLDM-GOLDGUINEA"] * 25,
        "q_A": [7300.0 + (50.0 if i == 20 else np.sin(i)*0.2) for i in range(25)],
        "q_B": [7300.0] * 25,
        "raw_price_A": [72500.0] * 25,
        "raw_price_B": [58400.0] * 25,
        "volume_A": [200] * 25,
        "volume_B": [2 if i == 20 else 50 for i in range(25)],  # Volume = 2 on bar 20!
        "oi_A": [1000] * 25,
        "oi_B": [500] * 25,
        "trading_unit_A": [100.0] * 25,
        "trading_unit_B": [8.0] * 25,
        "quotation_unit_A": [10.0] * 25,
        "quotation_unit_B": [8.0] * 25,
        "days_to_expiry": [30] * 25,
    })

    signals, _summary = generate_relative_value_signals(
        df, rolling_window=15, min_history_bars=10, z_score_threshold=2.0, min_volume=10
    )

    assert bool(signals.loc[20, "is_candidate_anomaly"]) is True
    assert signals.loc[20, "signal_status"] == "FILTERED_LIQUIDITY"
    assert "FILTERED by liquidity" in signals.loc[20, "explanation"]


def test_tender_buffer_filter():
    """Verify that an anomaly within the tender delivery window is filtered."""
    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    df = pd.DataFrame({
        "trading_date": dates,
        "expiry_date": [pd.to_datetime("2024-01-26")] * 25,
        "symbol_A": ["GOLDM"] * 25,
        "symbol_B": ["GOLDPETAL"] * 25,
        "pair_id": ["GOLDM-GOLDPETAL"] * 25,
        "q_A": [7300.0 + (60.0 if i == 23 else np.sin(i)*0.2) for i in range(25)],
        "q_B": [7300.0] * 25,
        "raw_price_A": [72500.0] * 25,
        "raw_price_B": [7300.0] * 25,
        "volume_A": [200] * 25,
        "volume_B": [500] * 25,
        "oi_A": [1000] * 25,
        "oi_B": [3000] * 25,
        "trading_unit_A": [100.0] * 25,
        "trading_unit_B": [1.0] * 25,
        "quotation_unit_A": [10.0] * 25,
        "quotation_unit_B": [1.0] * 25,
        "days_to_expiry": [25 - i for i in range(25)],  # bar 23 has DTE = 2 <= 3!
    })

    signals, _summary = generate_relative_value_signals(
        df, rolling_window=15, min_history_bars=10, z_score_threshold=2.0, expiry_buffer_days=3
    )

    assert bool(signals.loc[23, "is_candidate_anomaly"]) is True
    assert signals.loc[23, "signal_status"] == "FILTERED_NEAR_EXPIRY"
    assert "tender period buffer" in signals.loc[23, "explanation"]
