"""Unit and regression tests for Contract Master & Price Normalization."""

import pandas as pd
import pytest

from src.normalization import (
    ContractMasterError,
    calculate_fine_gold_price,
    normalize_dataframe,
    normalize_single_observation,
)


def test_calculate_fine_gold_price_manual_examples():
    """Verify normalization against exact hand-calculated figures.

    Example 1: GOLDM
      Quoted price P_i = ₹72,500
      Quotation unit g_i = 10 grams
      Purity u_i = 995 per mille (99.5%)
      Effective pure grams = 10 * 0.995 = 9.95g
      q_i = 72500 / 9.95 = 7286.43216...
    """
    q_goldm = calculate_fine_gold_price(
        quoted_price=72500.0,
        quotation_unit_grams=10.0,
        purity_per_mille=995.0,
    )
    expected_goldm = 72500.0 / (10.0 * 0.995)
    assert pytest.approx(q_goldm, rel=1e-5) == expected_goldm
    assert pytest.approx(q_goldm, abs=0.01) == 7286.43

    """
    Example 2: GOLDPETAL
      Quoted price P_i = ₹7,295
      Quotation unit g_i = 1 gram
      Purity u_i = 999 per mille (99.9%)
      Effective pure grams = 1 * 0.999 = 0.999g
      q_i = 7295 / 0.999 = 7302.3023...
    """
    q_petal = calculate_fine_gold_price(
        quoted_price=7295.0,
        quotation_unit_grams=1.0,
        purity_per_mille=999.0,
    )
    expected_petal = 7295.0 / 0.999
    assert pytest.approx(q_petal, rel=1e-5) == expected_petal
    assert pytest.approx(q_petal, abs=0.01) == 7302.30

    """
    Example 3: GOLDGUINEA
      Quoted price P_i = ₹58,400
      Quotation unit g_i = 8 grams
      Purity u_i = 999 per mille
      Effective pure grams = 8 * 0.999 = 7.992g
      q_i = 58400 / 7.992 = 7307.307...
    """
    q_guinea = calculate_fine_gold_price(
        quoted_price=58400.0,
        quotation_unit_grams=8.0,
        purity_per_mille=999.0,
    )
    assert pytest.approx(q_guinea, rel=1e-5) == (58400.0 / (8.0 * 0.999))


def test_quotation_unit_vs_trading_unit_separation():
    """Ensure trading unit (contract lot size) is NEVER conflated with quotation unit."""
    obs = normalize_single_observation("GOLDM", 72500.0)
    assert obs["quotation_unit_grams"] == 10.0
    assert obs["trading_unit_grams"] == 100.0
    assert obs["quotation_unit_grams"] != obs["trading_unit_grams"]

    obs_petal = normalize_single_observation("GOLDPETAL", 7300.0)
    assert obs_petal["quotation_unit_grams"] == 1.0
    assert obs_petal["trading_unit_grams"] == 1.0


def test_invalid_price_and_purity_inputs():
    """Ensure non-positive prices or invalid purities raise exceptions."""
    with pytest.raises(ValueError, match="positive"):
        calculate_fine_gold_price(-100.0, 10.0, 995.0)

    with pytest.raises(ValueError, match="positive"):
        calculate_fine_gold_price(72000.0, -10.0, 995.0)

    with pytest.raises(ValueError, match="Purity"):
        calculate_fine_gold_price(72000.0, 10.0, 1200.0)


def test_unknown_contract_symbol():
    """Ensure unknown contract symbols raise ContractMasterError."""
    with pytest.raises(ContractMasterError, match="not found in contract master"):
        normalize_single_observation("UNKNOWN_COIN", 50000.0)


def test_normalize_dataframe():
    """Test vectorized dataframe normalization."""
    raw_df = pd.DataFrame({
        "symbol": ["GOLDM", "GOLDPETAL"],
        "close_price": [72500.0, 7295.0],
        "trading_date": ["2024-10-01", "2024-10-01"],
    })
    norm_df = normalize_dataframe(raw_df)

    assert "pure_gold_equivalent_price" in norm_df.columns
    assert "raw_per_gram" in norm_df.columns
    assert "trading_unit_grams" in norm_df.columns

    p_goldm = norm_df[norm_df["symbol"] == "GOLDM"]["pure_gold_equivalent_price"].iloc[0]
    p_petal = norm_df[norm_df["symbol"] == "GOLDPETAL"]["pure_gold_equivalent_price"].iloc[0]

    assert pytest.approx(p_goldm, abs=0.01) == 7286.43
    assert pytest.approx(p_petal, abs=0.01) == 7302.30
