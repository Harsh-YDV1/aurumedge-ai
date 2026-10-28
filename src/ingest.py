"""AurumEdge AI - Data Ingestion Module.

Handles loading, standardizing, and parsing MCX Bhavcopy CSV files and
historical derivatives records across multiple layouts and date formats.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.storage import (
    RAW_DATA_DIR,
    ensure_directories,
    save_processed_data,
)
from src.validation import validate_records

logger = logging.getLogger(__name__)

# Canonical column mapping dictionary supporting varied exchange exports
COLUMN_ALIASES: dict[str, str] = {
    # Symbol
    "symbol": "symbol",
    "instrument name": "symbol",
    "contract": "symbol",
    "commodity": "symbol",
    # Dates
    "timestamp": "trading_date",
    "date": "trading_date",
    "trading date": "trading_date",
    "trade_date": "trading_date",
    "trade date": "trading_date",
    "expiry_dt": "expiry_date",
    "expiry date": "expiry_date",
    "expiry": "expiry_date",
    "expiry_date": "expiry_date",
    # Prices
    "open": "open_price",
    "open price": "open_price",
    "high": "high_price",
    "high price": "high_price",
    "low": "low_price",
    "low price": "low_price",
    "close": "close_price",
    "close price": "close_price",
    "ltp": "close_price",
    "last price": "close_price",
    "settle_pr": "settle_price",
    "settle price": "settle_price",
    "settlement price": "settle_price",
    # Quantities & Exposure
    "contracts": "volume",
    "volume": "volume",
    "traded qty": "volume",
    "vol": "volume",
    "open_int": "open_interest",
    "open interest": "open_interest",
    "oi": "open_interest",
    "val_inlakh": "turnover_lakhs",
    "value (lakhs)": "turnover_lakhs",
    "turnover": "turnover_lakhs",
}

DEFAULT_TARGET_SYMBOLS = ["GOLD", "GOLDM", "GOLDTEN", "GOLDGUINEA", "GOLDPETAL"]


@dataclass
class IngestionSummary:
    """Summary of data ingestion operation."""

    source_files: list[str] = field(default_factory=list)
    records_read: int = 0
    records_accepted: int = 0
    records_rejected: int = 0
    date_min: str | None = None
    date_max: str | None = None
    contracts_found: list[str] = field(default_factory=list)
    validation_report: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_files": self.source_files,
            "records_read": self.records_read,
            "records_accepted": self.records_accepted,
            "records_rejected": self.records_rejected,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "contracts_found": self.contracts_found,
            "validation_report": self.validation_report,
            "errors": self.errors,
        }


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names using case-insensitive mapping."""
    rename_dict = {}
    for col in df.columns:
        cleaned = str(col).strip().lower()
        if cleaned in COLUMN_ALIASES:
            rename_dict[col] = COLUMN_ALIASES[cleaned]
    return df.rename(columns=rename_dict)


def parse_bhavcopy_file(
    file_path: Path,
    target_symbols: list[str] | None = None,
    provenance: str = "MCX_RAW",
) -> pd.DataFrame:
    """Load a single Bhavcopy file and map to standardized fields."""
    if not file_path.exists():
        raise FileNotFoundError(f"Source file not found: {file_path}")

    # Read CSV safely
    try:
        raw = pd.read_csv(file_path, skipinitialspace=True)
    except Exception as e:
        raise ValueError(f"Failed to read CSV at {file_path}: {e}") from e

    std = standardize_columns(raw)

    # If Instrument column exists in raw MCX Bhavcopy, ensure FUTCOM futures
    inst_col = next((c for c in raw.columns if c.strip().lower() == "instrument"), None)
    if inst_col:
        is_futcom = raw[inst_col].astype(str).str.strip().str.upper() == "FUTCOM"
        std = std.loc[std.index.intersection(raw[is_futcom].index)].copy()

    # Filter target symbols if symbol column exists
    if "symbol" in std.columns:
        std["symbol"] = std["symbol"].astype(str).str.strip().str.upper()
        targets = [s.upper() for s in (target_symbols or DEFAULT_TARGET_SYMBOLS)]
        std = std[std["symbol"].isin(targets)].copy()

    # Fill optional settle_price if missing
    if "settle_price" not in std.columns and "close_price" in std.columns:
        std["settle_price"] = std["close_price"]

    # Fill turnover_lakhs if missing
    if "turnover_lakhs" not in std.columns:
        std["turnover_lakhs"] = 0.0

    # Ensure open, high, low, close, volume, open_interest exist
    for col in ["open_price", "high_price", "low_price", "close_price"]:
        if col not in std.columns:
            std[col] = None
    for col in ["volume", "open_interest"]:
        if col not in std.columns:
            std[col] = 0

    std["provenance"] = provenance
    std["source_file"] = file_path.name
    return std


def ingest_bhavcopy_files(
    file_paths: list[Path] | None = None,
    directory_path: Path | None = None,
    target_symbols: list[str] | None = None,
    save_output: bool = True,
    provenance: str = "MCX_RAW",
) -> tuple[pd.DataFrame, pd.DataFrame, IngestionSummary]:
    """Ingest multiple Bhavcopy files, apply validation, and produce a detailed report.

    Parameters:
        file_paths: Explicit list of file paths.
        directory_path: Directory containing CSVs (defaults to RAW_DATA_DIR if empty).
        target_symbols: List of commodity symbols to isolate.
        save_output: Whether to persist clean outputs as Parquet and CSV.
        provenance: Provenance tag (MCX_RAW or SYNTHETIC).

    Returns:
        (clean_df, rejected_df, IngestionSummary)
    """
    ensure_directories()
    summary = IngestionSummary()
    paths_to_load: list[Path] = []

    if file_paths:
        paths_to_load = [p for p in file_paths if p.is_file()]
    elif directory_path and directory_path.exists():
        paths_to_load = sorted(directory_path.glob("*.csv"))
    else:
        paths_to_load = sorted(RAW_DATA_DIR.glob("*.csv"))

    if not paths_to_load:
        summary.errors.append("No source files found to ingest.")
        return pd.DataFrame(), pd.DataFrame(), summary

    dfs = []
    for p in paths_to_load:
        summary.source_files.append(p.name)
        try:
            parsed = parse_bhavcopy_file(p, target_symbols=target_symbols, provenance=provenance)
            dfs.append(parsed)
        except Exception as e:  # noqa: BLE001
            summary.errors.append(f"Error parsing {p.name}: {e}")

    if not dfs:
        return pd.DataFrame(), pd.DataFrame(), summary

    combined = pd.concat(dfs, ignore_index=True)
    summary.records_read = len(combined)

    # Apply validation rules
    clean_df, rejected_df, val_report = validate_records(
        combined, allowed_symbols=target_symbols or DEFAULT_TARGET_SYMBOLS
    )

    summary.records_accepted = len(clean_df)
    summary.records_rejected = len(rejected_df)
    summary.validation_report = val_report.to_dict()

    if not clean_df.empty:
        summary.date_min = str(clean_df["trading_date"].min())[:10]
        summary.date_max = str(clean_df["trading_date"].max())[:10]
        summary.contracts_found = sorted(clean_df["symbol"].unique().tolist())

        if save_output:
            save_processed_data(clean_df, "mcx_cleaned_data", format_type="parquet")
            save_processed_data(clean_df, "mcx_cleaned_data", format_type="csv")
            if not rejected_df.empty:
                save_processed_data(rejected_df, "mcx_rejected_data", format_type="csv")

    return clean_df, rejected_df, summary
