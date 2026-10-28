"""AurumEdge AI - Contract Matching Engine.

Pairs comparable MCX gold futures contracts by enforcing:
  1. Exact same contract expiry date
  2. Common trading date intersection
  3. Minimum historical calendar overlap
  4. Active lifecycle validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.lifecycle import compute_lifecycle_metrics


@dataclass
class MatchDiagnostic:
    """Diagnostic details for a contract pair matching evaluation."""

    symbol_a: str
    symbol_b: str
    expiry_date: str | None = None
    common_trading_days: int = 0
    days_to_expiry_current: int | None = None
    is_eligible: bool = False
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol_a": self.symbol_a,
            "symbol_b": self.symbol_b,
            "expiry_date": self.expiry_date,
            "common_trading_days": self.common_trading_days,
            "days_to_expiry_current": self.days_to_expiry_current,
            "is_eligible": self.is_eligible,
            "rejection_reasons": self.rejection_reasons,
        }


def find_available_expiries(df: pd.DataFrame, symbol_a: str, symbol_b: str) -> list[str]:
    """Find all common expiry dates shared by two contract symbols."""
    if df.empty:
        return []
    df_a = df[df["symbol"] == symbol_a.upper()]
    df_b = df[df["symbol"] == symbol_b.upper()]

    exp_a_series = pd.to_datetime(df_a["expiry_date"], errors="coerce").dropna()
    exp_b_series = pd.to_datetime(df_b["expiry_date"], errors="coerce").dropna()
    exp_a = set(exp_a_series.dt.strftime("%Y-%m-%d").unique())
    exp_b = set(exp_b_series.dt.strftime("%Y-%m-%d").unique())
    return sorted(exp_a.intersection(exp_b))


def match_contract_pair(
    normalized_df: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    target_expiry: str | None = None,
    min_overlap_bars: int = 15,
    expiry_buffer_days: int = 3,
) -> tuple[pd.DataFrame, MatchDiagnostic]:
    """Align and merge two contract series on identical trading date and expiry.

    Parameters:
        normalized_df: DataFrame from src.normalization.normalize_dataframe.
        symbol_a: First contract symbol (e.g. 'GOLDM').
        symbol_b: Second contract symbol (e.g. 'GOLDPETAL').
        target_expiry: Optional specific expiry date (YYYY-MM-DD). If None, picks the nearest active common expiry.
        min_overlap_bars: Minimum shared trading bars required for valid research baseline.
        expiry_buffer_days: Days before expiration reserved for tender period.

    Returns:
        (paired_df, MatchDiagnostic)
    """
    sym_a = symbol_a.strip().upper()
    sym_b = symbol_b.strip().upper()
    diag = MatchDiagnostic(symbol_a=sym_a, symbol_b=sym_b)

    if sym_a == sym_b:
        diag.rejection_reasons.append("Cannot match contract with itself.")
        return pd.DataFrame(), diag

    if normalized_df.empty:
        diag.rejection_reasons.append("Empty dataset provided.")
        return pd.DataFrame(), diag

    df = normalized_df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])

    # Isolate symbols
    df_a = df[df["symbol"] == sym_a].copy()
    df_b = df[df["symbol"] == sym_b].copy()

    if df_a.empty:
        diag.rejection_reasons.append(f"No records found for Symbol A ({sym_a}).")
        return pd.DataFrame(), diag
    if df_b.empty:
        diag.rejection_reasons.append(f"No records found for Symbol B ({sym_b}).")
        return pd.DataFrame(), diag

    # Determine expiry
    common_expiries = find_available_expiries(df, sym_a, sym_b)
    if not common_expiries:
        diag.rejection_reasons.append(
            f"No common expiry dates shared between {sym_a} and {sym_b}."
        )
        return pd.DataFrame(), diag

    if target_expiry:
        chosen_exp = target_expiry
        if chosen_exp not in common_expiries:
            diag.rejection_reasons.append(
                f"Requested expiry {chosen_exp} is not in shared expiries: {common_expiries}"
            )
            return pd.DataFrame(), diag
    else:
        # Default to the most liquid or latest active expiry
        chosen_exp = common_expiries[0]

    diag.expiry_date = chosen_exp
    exp_dt = pd.to_datetime(chosen_exp)

    # Filter both to chosen expiry
    df_a_exp = df_a[df_a["expiry_date"] == exp_dt].copy()
    df_b_exp = df_b[df_b["expiry_date"] == exp_dt].copy()

    # Ensure price and optional columns exist
    for sub_df in [df_a_exp, df_b_exp]:
        if "raw_price" not in sub_df.columns:
            if "close_price" in sub_df.columns:
                sub_df["raw_price"] = sub_df["close_price"]
            else:
                sub_df["raw_price"] = sub_df["pure_gold_equivalent_price"]
        if "provenance" not in sub_df.columns:
            sub_df["provenance"] = "UNKNOWN"
        if "turnover_lakhs" not in sub_df.columns:
            sub_df["turnover_lakhs"] = 0.0

    # Suffixes for merge
    cols_a = {
        "raw_price": "raw_price_A",
        "pure_gold_equivalent_price": "q_A",
        "volume": "volume_A",
        "open_interest": "oi_A",
        "turnover_lakhs": "turnover_lakhs_A",
        "trading_unit_grams": "trading_unit_A",
        "quotation_unit_grams": "quotation_unit_A",
        "purity_per_mille": "purity_A",
    }
    cols_b = {
        "raw_price": "raw_price_B",
        "pure_gold_equivalent_price": "q_B",
        "volume": "volume_B",
        "open_interest": "oi_B",
        "turnover_lakhs": "turnover_lakhs_B",
        "trading_unit_grams": "trading_unit_B",
        "quotation_unit_grams": "quotation_unit_B",
        "purity_per_mille": "purity_B",
    }

    sub_a = df_a_exp[["trading_date", "expiry_date", "provenance"] + list(cols_a.keys())].rename(
        columns=cols_a
    )
    sub_b = df_b_exp[["trading_date", "expiry_date"] + list(cols_b.keys())].rename(
        columns=cols_b
    )

    # Inner merge on trading_date and expiry_date (common trading days only)
    paired = pd.merge(sub_a, sub_b, on=["trading_date", "expiry_date"], how="inner")
    paired = paired.sort_values("trading_date").reset_index(drop=True)

    diag.common_trading_days = len(paired)

    if paired.empty:
        diag.rejection_reasons.append(
            f"No overlapping trading sessions found for expiry {chosen_exp}."
        )
        return pd.DataFrame(), diag

    # Add lifecycle metrics
    paired = compute_lifecycle_metrics(paired, expiry_buffer_days=expiry_buffer_days)

    latest_dte = paired["days_to_expiry"].iloc[-1]
    diag.days_to_expiry_current = int(latest_dte)

    # Check minimum history overlap
    if len(paired) < min_overlap_bars:
        diag.rejection_reasons.append(
            f"Insufficient historical overlap ({len(paired)} bars < required {min_overlap_bars})."
        )
    else:
        diag.is_eligible = True

    paired["symbol_A"] = sym_a
    paired["symbol_B"] = sym_b
    paired["pair_id"] = f"{sym_a}-{sym_b}"

    return paired, diag
