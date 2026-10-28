"""AurumEdge AI - Relative-Value Signal Engine.

Computes scale-invariant log spreads, strictly point-in-time rolling statistics
(zero look-ahead bias), Gaussian Z-scores, and robust MAD outlier scores.
Enforces multi-stage eligibility filtering (liquidity, lifecycle, and history)
and generates deterministic human-readable explanations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SignalResult:
    """Detailed summary of signal generation for a pair."""

    pair_id: str
    total_bars: int = 0
    candidate_anomalies: int = 0
    eligible_signals: int = 0
    filtered_liquidity: int = 0
    filtered_expiry: int = 0
    filtered_history: int = 0
    latest_z_score: float | None = None
    latest_signal_status: str = "NO_DATA"
    latest_explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "total_bars": self.total_bars,
            "candidate_anomalies": self.candidate_anomalies,
            "eligible_signals": self.eligible_signals,
            "filtered_liquidity": self.filtered_liquidity,
            "filtered_expiry": self.filtered_expiry,
            "filtered_history": self.filtered_history,
            "latest_z_score": self.latest_z_score,
            "latest_signal_status": self.latest_signal_status,
            "latest_explanation": self.latest_explanation,
        }


def compute_relative_spread(q_a: pd.Series, q_b: pd.Series) -> pd.Series:
    """Compute scale-invariant log price spread: s_t = ln(q_A / q_B)."""
    if (q_a <= 0).any() or (q_b <= 0).any():
        raise ValueError("Normalized fine gold prices must be strictly positive.")
    return np.log(q_a / q_b)


def compute_point_in_time_statistics(
    spread: pd.Series,
    window: int = 20,
    min_periods: int = 15,
) -> tuple[pd.Series, pd.Series]:
    """Compute strictly point-in-time rolling mean and standard deviation.

    CRITICAL ANTI-LOOK-AHEAD:
    The statistics for observation at index t must use only observations from
    t - window up to t - 1. We achieve this by shifting the series by 1 period
    prior to rolling computation.
    """
    shifted_spread = spread.shift(1)
    rolling_mean = shifted_spread.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = shifted_spread.rolling(window=window, min_periods=min_periods).std(ddof=1)
    return rolling_mean, rolling_std


def compute_point_in_time_mad(
    spread: pd.Series,
    window: int = 20,
    min_periods: int = 15,
) -> tuple[pd.Series, pd.Series]:
    """Compute strictly point-in-time rolling median and MAD."""
    shifted = spread.shift(1)

    def calc_mad(x: np.ndarray) -> float:
        valid = x[~np.isnan(x)]
        if len(valid) < min_periods:
            return np.nan
        med = np.median(valid)
        mad = np.median(np.abs(valid - med))
        return float(mad)

    rolling_median = shifted.rolling(window=window, min_periods=min_periods).median()
    rolling_mad = shifted.rolling(window=window, min_periods=min_periods).apply(
        calc_mad, raw=True
    )
    return rolling_median, rolling_mad


def generate_signal_explanation(row: pd.Series, z_thresh: float) -> str:
    """Generate deterministic, human-readable reason for the signal classification."""
    status = row["signal_status"]
    z = row["z_score"]
    spread_pct = (np.exp(row["spread"]) - 1.0) * 100.0
    dte = row["days_to_expiry"]

    if status == "INSUFFICIENT_HISTORY":
        return (
            f"Insufficient historical data ({row['bar_number']} bars available, "
            f"minimum required: {row['min_history_bars']})."
        )
    if status == "NORMAL":
        return (
            f"Normal relative value. Spread is {spread_pct:+.2f}% (Z-Score: {z:+.2f}), "
            f"within [-{z_thresh:.1f}, +{z_thresh:.1f}] threshold."
        )

    rich_contract = row["symbol_A"] if z > 0 else row["symbol_B"]
    cheap_contract = row["symbol_B"] if z > 0 else row["symbol_A"]

    if status == "ELIGIBLE_SIGNAL":
        return (
            f"ELIGIBLE ANOMALY: Spread is {spread_pct:+.2f}% ({abs(z):.2f}σ). "
            f"{rich_contract} is rich relative to {cheap_contract}. "
            f"Passed liquidity (Vol: A={row['volume_A']:.0f}/B={row['volume_B']:.0f}, "
            f"OI: A={row['oi_A']:.0f}/B={row['oi_B']:.0f}) and tender buffer ({dte} DTE)."
        )
    if status == "FILTERED_LIQUIDITY":
        return (
            f"STATISTICAL ANOMALY ({abs(z):.2f}σ) FILTERED by liquidity: "
            f"Volume (A={row['volume_A']:.0f}, B={row['volume_B']:.0f}) or "
            f"OI (A={row['oi_A']:.0f}, B={row['oi_B']:.0f}) below thresholds."
        )
    if status == "FILTERED_NEAR_EXPIRY":
        return (
            f"STATISTICAL ANOMALY ({abs(z):.2f}σ) FILTERED by tender period buffer: "
            f"Contract is {dte} days to expiry (tender buffer is {row['expiry_buffer_days']} days)."
        )
    return f"Status: {status} with Z-Score: {z:+.2f}."


def generate_relative_value_signals(
    paired_df: pd.DataFrame,
    rolling_window: int = 20,
    z_score_threshold: float = 2.0,
    min_history_bars: int = 15,
    min_volume: int = 10,
    min_open_interest: int = 20,
    expiry_buffer_days: int = 3,
) -> tuple[pd.DataFrame, SignalResult]:
    """Generate point-in-time signals and multi-stage eligibility evaluation.

    Returns:
        (signals_df, SignalResult)
    """
    pair_id = paired_df["pair_id"].iloc[0] if not paired_df.empty else "UNKNOWN"
    summary = SignalResult(pair_id=pair_id, total_bars=len(paired_df))

    if paired_df.empty:
        return paired_df.copy(), summary

    df = paired_df.sort_values("trading_date").reset_index(drop=True).copy()
    df["bar_number"] = np.arange(1, len(df) + 1)
    df["min_history_bars"] = min_history_bars
    df["expiry_buffer_days"] = expiry_buffer_days

    # 1. Log Spread
    df["spread"] = compute_relative_spread(df["q_A"], df["q_B"])

    # 2. Point-in-time Gaussian rolling statistics (Shifted 1 period)
    df["rolling_mean"], df["rolling_std"] = compute_point_in_time_statistics(
        df["spread"], window=rolling_window, min_periods=min_history_bars
    )

    # 3. Z-score calculation
    diff = df["spread"] - df["rolling_mean"]
    std_safe = df["rolling_std"].replace(0, np.nan)
    df["z_score"] = (diff / std_safe).fillna(0.0)

    # 4. Robust MAD metrics
    df["rolling_median"], df["rolling_mad"] = compute_point_in_time_mad(
        df["spread"], window=rolling_window, min_periods=min_history_bars
    )
    mad_denom = df["rolling_mad"] * 1.4826
    mad_denom = mad_denom.replace(0, np.nan)
    df["mad_score"] = ((df["spread"] - df["rolling_median"]) / mad_denom).fillna(0.0)

    # 5. Upper / Lower reference bands
    df["band_upper"] = df["rolling_mean"] + (z_score_threshold * df["rolling_std"])
    df["band_lower"] = df["rolling_mean"] - (z_score_threshold * df["rolling_std"])

    # 6. Filters
    df["filter_liquidity_passed"] = (
        (df["volume_A"] >= min_volume)
        & (df["volume_B"] >= min_volume)
        & (df["oi_A"] >= min_open_interest)
        & (df["oi_B"] >= min_open_interest)
    )
    df["filter_expiry_passed"] = df["days_to_expiry"] > expiry_buffer_days
    df["filter_history_passed"] = df["bar_number"] >= min_history_bars

    # 7. Classification logic
    is_anomaly = df["z_score"].abs() >= z_score_threshold
    df["is_candidate_anomaly"] = is_anomaly

    status_list = []
    for idx, row in df.iterrows():
        if not row["filter_history_passed"]:
            status_list.append("INSUFFICIENT_HISTORY")
        elif not row["is_candidate_anomaly"]:
            status_list.append("NORMAL")
        elif not row["filter_expiry_passed"]:
            status_list.append("FILTERED_NEAR_EXPIRY")
        elif not row["filter_liquidity_passed"]:
            status_list.append("FILTERED_LIQUIDITY")
        else:
            status_list.append("ELIGIBLE_SIGNAL")

    df["signal_status"] = status_list

    # 8. Human-readable explanations
    df["explanation"] = [
        generate_signal_explanation(row, z_score_threshold) for _, row in df.iterrows()
    ]

    # Summarize
    summary.candidate_anomalies = int(df["is_candidate_anomaly"].sum())
    summary.eligible_signals = int((df["signal_status"] == "ELIGIBLE_SIGNAL").sum())
    summary.filtered_liquidity = int((df["signal_status"] == "FILTERED_LIQUIDITY").sum())
    summary.filtered_expiry = int((df["signal_status"] == "FILTERED_NEAR_EXPIRY").sum())
    summary.filtered_history = int((df["signal_status"] == "INSUFFICIENT_HISTORY").sum())

    if not df.empty:
        summary.latest_z_score = float(df["z_score"].iloc[-1])
        summary.latest_signal_status = str(df["signal_status"].iloc[-1])
        summary.latest_explanation = str(df["explanation"].iloc[-1])

    return df, summary
