"""AurumEdge AI - Data Validation Module.

Enforces structural integrity, market data sanity checks, price bounds,
high/low consistency, duplicate elimination, and lifecycle validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ValidationReport:
    """Detailed summary of data validation execution."""

    total_records: int = 0
    clean_records: int = 0
    rejected_records: int = 0
    duplicate_records: int = 0
    missing_field_records: int = 0
    price_anomaly_warnings: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "clean_records": self.clean_records,
            "rejected_records": self.rejected_records,
            "duplicate_records": self.duplicate_records,
            "missing_field_records": self.missing_field_records,
            "price_anomaly_warnings": self.price_anomaly_warnings,
            "rejection_reasons": self.rejection_reasons,
            "warning_count": len(self.warnings),
        }


REQUIRED_COLUMNS = [
    "trading_date",
    "symbol",
    "expiry_date",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume",
    "open_interest",
]


def validate_records(
    df: pd.DataFrame,
    allowed_symbols: list[str] | None = None,
    jump_threshold_pct: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, ValidationReport]:
    """Validate DataFrame against canonical exchange rules.

    Parameters:
        df: Input DataFrame with normalized column names.
        allowed_symbols: Optional whitelist of valid symbols.
        jump_threshold_pct: Relative day-over-day price change triggering a review warning.

    Returns:
        (clean_df, rejected_df, report)
    """
    report = ValidationReport(total_records=len(df))
    if df.empty:
        return df.copy(), pd.DataFrame(), report

    working = df.copy()

    # Step 1: Check required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in working.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")

    working["rejection_reason"] = ""

    # Clean strings
    working["symbol"] = working["symbol"].astype(str).str.strip().str.upper()

    # Check allowed symbols if provided
    if allowed_symbols:
        symbol_mask = working["symbol"].isin([s.upper() for s in allowed_symbols])
        invalid_symbols = working[~symbol_mask]
        for idx in invalid_symbols.index:
            working.loc[idx, "rejection_reason"] = (
                f"Invalid symbol: {working.loc[idx, 'symbol']}"
            )

    # Date sanity
    working["trading_date"] = pd.to_datetime(working["trading_date"], errors="coerce")
    working["expiry_date"] = pd.to_datetime(working["expiry_date"], errors="coerce")

    missing_date_mask = working["trading_date"].isna() | working["expiry_date"].isna()
    for idx in working[missing_date_mask].index:
        working.loc[idx, "rejection_reason"] = "Missing or unparseable trading/expiry date"

    # Numeric conversion
    price_cols = ["open_price", "high_price", "low_price", "close_price"]
    for col in price_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    working["volume"] = pd.to_numeric(working["volume"], errors="coerce").fillna(0)
    working["open_interest"] = pd.to_numeric(working["open_interest"], errors="coerce").fillna(0)

    # Non-positive price check
    for col in price_cols:
        non_pos = working[working["rejection_reason"] == ""][col] <= 0
        for idx in non_pos[non_pos].index:
            working.loc[idx, "rejection_reason"] = f"Non-positive price in {col}"

    # High / Low price consistency check
    hl_mask = (working["rejection_reason"] == "") & (working["high_price"] < working["low_price"])
    for idx in working[hl_mask].index:
        working.loc[idx, "rejection_reason"] = "High price strictly lower than Low price"

    close_bound_mask = (working["rejection_reason"] == "") & (
        (working["close_price"] > working["high_price"] * 1.0001)
        | (working["close_price"] < working["low_price"] * 0.9999)
    )
    for idx in working[close_bound_mask].index:
        working.loc[idx, "rejection_reason"] = "Close price outside [Low, High] bounds"

    # Expiry before trading date
    expiry_past_mask = (working["rejection_reason"] == "") & (
        working["expiry_date"] < working["trading_date"]
    )
    for idx in working[expiry_past_mask].index:
        working.loc[idx, "rejection_reason"] = "Expiry date precedes trading date"

    # Duplicate detection on (trading_date, symbol, expiry_date)
    dup_mask = (working["rejection_reason"] == "") & working.duplicated(
        subset=["trading_date", "symbol", "expiry_date"], keep="first"
    )
    for idx in working[dup_mask].index:
        working.loc[idx, "rejection_reason"] = (
            "Duplicate observation for contract-date-expiry tuple"
        )
        report.duplicate_records += 1

    # Split into clean and rejected
    rejected_mask = working["rejection_reason"] != ""
    rejected_df = working[rejected_mask].copy()
    clean_df = working[~rejected_mask].copy().drop(columns=["rejection_reason"])

    # Tally rejection reasons
    for reason in rejected_df["rejection_reason"]:
        report.rejection_reasons[reason] = report.rejection_reasons.get(reason, 0) + 1

    report.rejected_records = len(rejected_df)
    report.clean_records = len(clean_df)

    # Detect suspicious price jumps for review (Warnings, not rejects)
    clean_df = clean_df.sort_values(["symbol", "expiry_date", "trading_date"])
    for (sym, exp), grp in clean_df.groupby(["symbol", "expiry_date"]):
        if len(grp) > 1:
            pct_chg = grp["close_price"].pct_change().abs()
            jump_idx = pct_chg[pct_chg > jump_threshold_pct].index
            for j_idx in jump_idx:
                chg_val = pct_chg.loc[j_idx]
                report.warnings.append(
                    {
                        "type": "PRICE_JUMP_WARNING",
                        "symbol": sym,
                        "expiry_date": str(exp)[:10],
                        "trading_date": str(clean_df.loc[j_idx, "trading_date"])[:10],
                        "close_price": float(clean_df.loc[j_idx, "close_price"]),
                        "abs_change_pct": round(float(chg_val) * 100, 2),
                        "message": (
                            f"Suspicious price jump of {chg_val*100:.1f}% on {sym} "
                            f"(Expiry {str(exp)[:10]}). Retained for analytical review."
                        ),
                    }
                )

    report.price_anomaly_warnings = len(report.warnings)

    return clean_df.reset_index(drop=True), rejected_df.reset_index(drop=True), report
