"""AurumEdge AI - Contract Master & Price Normalization Module.

Handles contract specifications (trading unit vs quotation unit vs purity)
and converts raw quoted futures prices into purity-adjusted fine-gold-equivalent
prices per gram:

    q_i = P_i / (g_i * (u_i / 1000))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.storage import CONTRACTS_FILE


class ContractMasterError(ValueError):
    """Raised when contract specification lookup or validation fails."""


def load_contract_master(contracts_path: Path | None = None) -> pd.DataFrame:
    """Load and validate the contract master specifications CSV.

    Expected columns:
      symbol, name, trading_unit_grams, quotation_unit_grams,
      purity_per_mille, tick_size, delivery_type, specification_source, notes
    """
    path = contracts_path or CONTRACTS_FILE
    if not path.exists():
        raise FileNotFoundError(f"Contract master file not found at: {path}")

    df = pd.read_csv(path)
    required_cols = [
        "symbol",
        "trading_unit_grams",
        "quotation_unit_grams",
        "purity_per_mille",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ContractMasterError(f"Missing required contract master column: {col}")

    # Standardize types and clean whitespace
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["trading_unit_grams"] = df["trading_unit_grams"].astype(float)
    df["quotation_unit_grams"] = df["quotation_unit_grams"].astype(float)
    df["purity_per_mille"] = df["purity_per_mille"].astype(float)

    # Sanity checks
    if (df["quotation_unit_grams"] <= 0).any():
        raise ContractMasterError("Quotation unit grams must be strictly positive.")
    if (df["trading_unit_grams"] <= 0).any():
        raise ContractMasterError("Trading unit grams must be strictly positive.")
    if ((df["purity_per_mille"] <= 0) | (df["purity_per_mille"] > 1000)).any():
        raise ContractMasterError("Purity per mille must be in (0, 1000].")

    return df


def calculate_fine_gold_price(
    quoted_price: float,
    quotation_unit_grams: float,
    purity_per_mille: float,
) -> float:
    """Calculate the purity-adjusted price per gram of 100% fine gold equivalent.

    Formula:
        q_i = P_i / (g_i * (u_i / 1000))

    Parameters:
        quoted_price: Observed market price P_i.
        quotation_unit_grams: Gram weight for the quote g_i.
        purity_per_mille: Purity u_i (e.g. 995.0 or 999.0).
    """
    if quoted_price <= 0:
        raise ValueError(f"Quoted price must be positive, got {quoted_price}")
    if quotation_unit_grams <= 0:
        raise ValueError(f"Quotation unit grams must be positive, got {quotation_unit_grams}")
    if not (0 < purity_per_mille <= 1000):
        raise ValueError(f"Purity per mille must be in (0, 1000], got {purity_per_mille}")

    purity_fraction = purity_per_mille / 1000.0
    effective_fine_grams = quotation_unit_grams * purity_fraction
    return quoted_price / effective_fine_grams


def normalize_single_observation(
    symbol: str,
    raw_price: float,
    contract_master: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Normalize a single price observation and return step-by-step audit details."""
    master = contract_master if contract_master is not None else load_contract_master()
    row = master[master["symbol"] == symbol.strip().upper()]
    if row.empty:
        raise ContractMasterError(f"Symbol '{symbol}' not found in contract master.")

    spec = row.iloc[0]
    g_i = float(spec["quotation_unit_grams"])
    u_i = float(spec["purity_per_mille"])
    trading_unit = float(spec["trading_unit_grams"])

    purity_fraction = u_i / 1000.0
    raw_per_gram = raw_price / g_i
    fine_gold_equivalent = calculate_fine_gold_price(raw_price, g_i, u_i)

    return {
        "symbol": symbol.strip().upper(),
        "raw_price": raw_price,
        "quotation_unit_grams": g_i,
        "trading_unit_grams": trading_unit,
        "purity_per_mille": u_i,
        "purity_fraction": purity_fraction,
        "raw_per_gram": raw_per_gram,
        "pure_gold_equivalent_price": fine_gold_equivalent,
        "formula": "q_i = P_i / (g_i * (u_i / 1000))",
        "explanation": (
            f"Quoted ₹{raw_price:,.2f} per {g_i}g with {u_i} fineness ({purity_fraction*100:.1f}%) "
            f"-> ₹{raw_per_gram:,.3f}/g raw -> ₹{fine_gold_equivalent:,.3f}/g fine gold."
        ),
    }


def normalize_dataframe(
    df: pd.DataFrame,
    price_col: str = "close_price",
    contract_master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Vectorized normalization of an entire dataset of contract records.

    Merges contract specifications, validates inputs, and appends:
      - quotation_unit_grams
      - trading_unit_grams
      - purity_per_mille
      - purity_fraction
      - raw_per_gram
      - pure_gold_equivalent_price
    """
    master = contract_master if contract_master is not None else load_contract_master()
    if df.empty:
        return df.copy()

    result = df.copy()
    result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()
    if "trading_date" in result.columns:
        result["trading_date"] = pd.to_datetime(result["trading_date"], errors="coerce")
    if "expiry_date" in result.columns:
        result["expiry_date"] = pd.to_datetime(result["expiry_date"], errors="coerce")

    # Merge on symbol
    specs_to_merge = master[
        ["symbol", "trading_unit_grams", "quotation_unit_grams", "purity_per_mille"]
    ].copy()
    merged = result.merge(specs_to_merge, on="symbol", how="left")

    # Check for missing specifications
    missing_specs = merged[merged["quotation_unit_grams"].isna()]["symbol"].unique()
    if len(missing_specs) > 0:
        raise ContractMasterError(
            f"Unrecognized contract symbols with no master spec: {missing_specs.tolist()}"
        )

    # Compute intermediate and final normalized columns
    merged["purity_fraction"] = merged["purity_per_mille"] / 1000.0
    merged["raw_per_gram"] = merged[price_col] / merged["quotation_unit_grams"]
    merged["pure_gold_equivalent_price"] = merged[price_col] / (
        merged["quotation_unit_grams"] * merged["purity_fraction"]
    )

    return merged
