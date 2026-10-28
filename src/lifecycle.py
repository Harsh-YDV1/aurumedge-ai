"""AurumEdge AI - Contract Lifecycle & Liquidity Diagnostics.

Tracks contract expiration proximity, delivery tender period buffers,
trading calendar health, and liquidity attributes.
"""

from __future__ import annotations

import pandas as pd


def compute_lifecycle_metrics(
    df: pd.DataFrame,
    expiry_buffer_days: int = 3,
) -> pd.DataFrame:
    """Calculate days to expiry, lifecycle state, and tender-period buffer flags.

    Parameters:
        df: Cleaned dataframe with trading_date and expiry_date.
        expiry_buffer_days: Days before contract expiration treated as tender period.

    Returns:
        DataFrame augmented with:
          - days_to_expiry
          - is_tender_period
          - lifecycle_state ('ACTIVE', 'TENDER_NEAR_EXPIRY', 'EXPIRED')
    """
    if df.empty:
        return df.copy()

    res = df.copy()
    res["trading_date"] = pd.to_datetime(res["trading_date"])
    res["expiry_date"] = pd.to_datetime(res["expiry_date"])

    # Calendar days to expiry
    res["days_to_expiry"] = (res["expiry_date"] - res["trading_date"]).dt.days

    # Tender period flag
    res["is_tender_period"] = res["days_to_expiry"] <= expiry_buffer_days

    # Lifecycle state
    conditions = [
        res["days_to_expiry"] < 0,
        res["days_to_expiry"] <= expiry_buffer_days,
        res["days_to_expiry"] > expiry_buffer_days,
    ]
    choices = ["EXPIRED", "TENDER_NEAR_EXPIRY", "ACTIVE"]

    res["lifecycle_state"] = pd.Series(
        choices[1], index=res.index
    )  # default fallback
    for cond, choice in zip(conditions, choices):
        res.loc[cond, "lifecycle_state"] = choice

    return res


def evaluate_liquidity_filters(
    df: pd.DataFrame,
    min_volume: int = 10,
    min_open_interest: int = 20,
) -> pd.Series:
    """Evaluate whether each row passes minimum liquidity thresholds.

    Returns a boolean Series (True = Passed, False = Failed).
    """
    vol_pass = df["volume"] >= min_volume
    oi_pass = df["open_interest"] >= min_open_interest
    return vol_pass & oi_pass
