"""Unit tests for Contract Matching and Pairing."""

import pandas as pd

from src.matching import match_contract_pair
from src.normalization import normalize_dataframe


def create_mock_normalized_df():
    """Create mock dataset with matching and non-matching expiries."""
    records = []
    # 20 trading sessions
    dates = pd.date_range("2024-09-01", periods=20, freq="B")
    for d in dates:
        # GOLDM expiring 2024-11-05
        records.append({
            "trading_date": d,
            "symbol": "GOLDM",
            "expiry_date": pd.to_datetime("2024-11-05"),
            "open_price": 72000.0,
            "high_price": 72500.0,
            "low_price": 71800.0,
            "close_price": 72300.0,
            "volume": 500,
            "open_interest": 2000,
            "turnover_lakhs": 360.0,
            "provenance": "SYNTHETIC",
        })
        # GOLDPETAL expiring 2024-11-05 (Matching expiry!)
        records.append({
            "trading_date": d,
            "symbol": "GOLDPETAL",
            "expiry_date": pd.to_datetime("2024-11-05"),
            "open_price": 7250.0,
            "high_price": 7290.0,
            "low_price": 7220.0,
            "close_price": 7280.0,
            "volume": 800,
            "open_interest": 4000,
            "turnover_lakhs": 58.0,
            "provenance": "SYNTHETIC",
        })
        # GOLDTEN expiring 2024-12-05 (Different expiry!)
        records.append({
            "trading_date": d,
            "symbol": "GOLDTEN",
            "expiry_date": pd.to_datetime("2024-12-05"),
            "open_price": 72600.0,
            "high_price": 73000.0,
            "low_price": 72400.0,
            "close_price": 72900.0,
            "volume": 120,
            "open_interest": 600,
            "turnover_lakhs": 87.0,
            "provenance": "SYNTHETIC",
        })

    raw_df = pd.DataFrame(records)
    return normalize_dataframe(raw_df)


def test_matching_same_expiry_succeeds():
    """Verify pair sharing exact same expiry aligns properly."""
    norm_df = create_mock_normalized_df()
    paired, diag = match_contract_pair(norm_df, "GOLDM", "GOLDPETAL", min_overlap_bars=15)

    assert diag.is_eligible is True
    assert len(paired) == 20
    assert diag.expiry_date == "2024-11-05"
    assert "q_A" in paired.columns
    assert "q_B" in paired.columns
    assert "days_to_expiry" in paired.columns


def test_mismatched_expiry_rejection():
    """Verify contracts with no common expiry dates are rejected."""
    norm_df = create_mock_normalized_df()
    paired, diag = match_contract_pair(norm_df, "GOLDM", "GOLDTEN")

    assert diag.is_eligible is False
    assert len(paired) == 0
    assert any("No common expiry" in r for r in diag.rejection_reasons)


def test_self_match_rejection():
    """Verify matching a contract with itself is rejected."""
    norm_df = create_mock_normalized_df()
    _paired, diag = match_contract_pair(norm_df, "GOLDM", "GOLDM")
    assert diag.is_eligible is False
    assert any("Cannot match contract with itself" in r for r in diag.rejection_reasons)


def test_insufficient_history_overlap():
    """Verify pairs with fewer than required bars are flagged as ineligible."""
    norm_df = create_mock_normalized_df()
    # Require 25 bars when only 20 exist
    _paired, diag = match_contract_pair(norm_df, "GOLDM", "GOLDPETAL", min_overlap_bars=25)

    assert diag.is_eligible is False
    assert any("Insufficient historical overlap" in r for r in diag.rejection_reasons)
