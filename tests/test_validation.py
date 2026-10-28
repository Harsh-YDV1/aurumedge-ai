"""Unit tests for Data Validation Module."""

import pandas as pd

from src.validation import validate_records


def test_valid_records_pass_cleanly():
    """Verify clean dataset passes with 0 rejections."""
    df = pd.DataFrame({
        "trading_date": ["2024-10-01", "2024-10-02"],
        "symbol": ["GOLDM", "GOLDM"],
        "expiry_date": ["2024-11-05", "2024-11-05"],
        "open_price": [72000.0, 72200.0],
        "high_price": [72500.0, 72600.0],
        "low_price": [71900.0, 72100.0],
        "close_price": [72300.0, 72400.0],
        "volume": [100, 150],
        "open_interest": [500, 520],
    })

    clean, rejected, report = validate_records(df)
    assert len(clean) == 2
    assert len(rejected) == 0
    assert report.clean_records == 2
    assert report.rejected_records == 0


def test_nonpositive_price_rejection():
    """Verify non-positive prices are rejected."""
    df = pd.DataFrame({
        "trading_date": ["2024-10-01"],
        "symbol": ["GOLDM"],
        "expiry_date": ["2024-11-05"],
        "open_price": [72000.0],
        "high_price": [72500.0],
        "low_price": [71900.0],
        "close_price": [-50.0],  # Negative price!
        "volume": [100],
        "open_interest": [500],
    })

    clean, rejected, _report = validate_records(df)
    assert len(clean) == 0
    assert len(rejected) == 1
    assert "Non-positive price" in rejected["rejection_reason"].iloc[0]


def test_high_low_violation_rejection():
    """Verify High < Low is rejected."""
    df = pd.DataFrame({
        "trading_date": ["2024-10-01"],
        "symbol": ["GOLDM"],
        "expiry_date": ["2024-11-05"],
        "open_price": [72000.0],
        "high_price": [71000.0],  # High < Low!
        "low_price": [72500.0],
        "close_price": [72000.0],
        "volume": [100],
        "open_interest": [500],
    })

    clean, rejected, _report = validate_records(df)
    assert len(clean) == 0
    assert len(rejected) == 1
    assert "strictly lower than Low" in rejected["rejection_reason"].iloc[0]


def test_duplicate_record_rejection():
    """Verify duplicate observation for the same contract-date-expiry is filtered."""
    df = pd.DataFrame({
        "trading_date": ["2024-10-01", "2024-10-01"],
        "symbol": ["GOLDM", "GOLDM"],
        "expiry_date": ["2024-11-05", "2024-11-05"],
        "open_price": [72000.0, 72000.0],
        "high_price": [72500.0, 72500.0],
        "low_price": [71900.0, 71900.0],
        "close_price": [72300.0, 72300.0],
        "volume": [100, 100],
        "open_interest": [500, 500],
    })

    clean, rejected, report = validate_records(df)
    assert len(clean) == 1
    assert len(rejected) == 1
    assert report.duplicate_records == 1


def test_expiry_precedes_trading_date():
    """Verify expired contracts at trade date are rejected."""
    df = pd.DataFrame({
        "trading_date": ["2024-11-10"],
        "symbol": ["GOLDM"],
        "expiry_date": ["2024-11-05"],  # Expiry was 5 days prior!
        "open_price": [72000.0],
        "high_price": [72500.0],
        "low_price": [71900.0],
        "close_price": [72300.0],
        "volume": [100],
        "open_interest": [500],
    })

    clean, rejected, _report = validate_records(df)
    assert len(clean) == 0
    assert len(rejected) == 1
    assert "precedes trading date" in rejected["rejection_reason"].iloc[0]
