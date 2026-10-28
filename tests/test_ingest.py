"""Unit tests for Data Ingestion Module."""

from pathlib import Path

import pandas as pd

from src.ingest import ingest_bhavcopy_files, parse_bhavcopy_file, standardize_columns


def test_standardize_columns_variations():
    """Verify divergent headers map to canonical names."""
    df_var1 = pd.DataFrame(columns=["TIMESTAMP", "SYMBOL", "EXPIRY_DT", "OPEN", "HIGH", "LOW", "CLOSE", "CONTRACTS", "OPEN_INT"])
    res1 = standardize_columns(df_var1)
    assert "trading_date" in res1.columns
    assert "expiry_date" in res1.columns
    assert "volume" in res1.columns
    assert "open_interest" in res1.columns

    df_var2 = pd.DataFrame(columns=["Trading Date", "Contract", "Expiry Date", "Open Price", "Close Price", "Traded Qty", "OI"])
    res2 = standardize_columns(df_var2)
    assert "trading_date" in res2.columns
    assert "symbol" in res2.columns
    assert "expiry_date" in res2.columns
    assert "volume" in res2.columns
    assert "open_interest" in res2.columns


def test_parse_bhavcopy_file_with_synthetic_sample(tmp_path: Path):
    """Verify parsing a simulated MCX CSV file."""
    csv_file = tmp_path / "test_mcx.csv"
    data = (
        "INSTRUMENT,SYMBOL,EXPIRY_DT,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,OPEN_INT,VAL_INLAKH,TIMESTAMP\n"
        "FUTCOM,GOLDM,05-DEC-2024,72000,72500,71900,72300,72300,150,800,108.45,01-OCT-2024\n"
        "FUTCOM,GOLDPETAL,05-DEC-2024,7250,7295,7230,7280,7280,850,3200,61.88,01-OCT-2024\n"
        "FUTCOM,SILVER,05-DEC-2024,85000,85500,84500,85000,85000,100,500,85.0,01-OCT-2024\n"
    )
    csv_file.write_text(data)

    parsed = parse_bhavcopy_file(csv_file, target_symbols=["GOLDM", "GOLDPETAL"])
    assert len(parsed) == 2
    assert "SILVER" not in parsed["symbol"].values
    assert set(parsed["symbol"].unique()) == {"GOLDM", "GOLDPETAL"}
    assert parsed["provenance"].iloc[0] == "MCX_RAW"


def test_ingest_bhavcopy_files_empty_dir(tmp_path: Path):
    """Verify empty directory returns empty dataframe and records error."""
    clean, _rejected, summary = ingest_bhavcopy_files(directory_path=tmp_path)
    assert clean.empty
    assert "No source files found" in summary.errors[0]
