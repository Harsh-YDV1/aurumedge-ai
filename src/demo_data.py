"""AurumEdge AI - Deterministic Synthetic Demo Data Generator.

Generates realistic, cointegrated multi-contract MCX gold futures records
with controlled anomalies for interactive demonstration, testing, and evaluation.
All generated records are explicitly tagged with provenance='SYNTHETIC'.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.storage import SAMPLE_DATA_DIR, ensure_directories

DEFAULT_SYMBOLS = ["GOLD", "GOLDM", "GOLDTEN", "GOLDGUINEA", "GOLDPETAL"]


def generate_synthetic_gold_dataset(
    start_date: str = "2024-07-01",
    num_days: int = 70,
    seed: int = 42,
    include_injected_anomaly: bool = True,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Generate cointegrated synthetic MCX gold futures dataset.

    Parameters:
        start_date: Starting calendar date.
        num_days: Number of trading sessions to simulate.
        seed: Fixed random seed for deterministic reproducibility.
        include_injected_anomaly: Whether to inject controlled relative-value dislocations.
        output_dir: Directory to persist the generated files.

    Returns:
        pd.DataFrame containing standardized canonical columns.
    """
    np.random.seed(seed)
    ensure_directories()

    # Generate business dates
    trading_dates = pd.date_range(start=start_date, periods=num_days, freq="B")

    # Contract Expiry Date: Set to 5 business days after the simulation ends
    expiry_date = trading_dates[-1] + pd.Timedelta(days=5)

    # 1. Simulate base fine-gold price (₹ per pure gram) using Geometric Brownian Motion
    s0 = 7300.0  # ₹7,300 per gram pure gold
    mu = 0.08 / 252  # 8% annualized drift
    sigma = 0.14 / np.sqrt(252)  # 14% annualized volatility

    daily_returns = np.random.normal(mu, sigma, num_days)
    pure_price_path = s0 * np.exp(np.cumsum(daily_returns))

    # Contract specifications:
    # Symbol: (quotation_unit_grams, purity_per_mille, vol_mean, vol_std, oi_base)
    specs = {
        "GOLD": (10.0, 995.0, 1500, 300, 12000),
        "GOLDM": (10.0, 995.0, 2200, 450, 8500),
        "GOLDTEN": (10.0, 999.0, 180, 50, 1200),
        "GOLDGUINEA": (8.0, 999.0, 95, 30, 650),
        "GOLDPETAL": (1.0, 999.0, 950, 200, 4200),
    }

    all_rows = []

    # Inject specific anomalies for demonstration:
    # 1. Day 40-42: Dislocation in GOLDPETAL (retail surge causes petal to trade rich)
    # 2. Day 55: Illiquid spike in GOLDGUINEA (triggers liquidity filter)
    # 3. Day 68: Dislocation right before expiry (triggers tender buffer filter)

    for sym, (g_i, u_i, vol_base, vol_sd, oi_base) in specs.items():
        purity_fraction = u_i / 1000.0

        for t_idx, t_date in enumerate(trading_dates):
            base_pure_g = pure_price_path[t_idx]

            # Contract-specific basis noise (microstructure spread ~ 0.05% - 0.15%)
            basis_noise = np.random.normal(0, 0.0008)

            # Injected controlled anomaly logic
            anomaly_factor = 0.0
            if include_injected_anomaly:
                # Case 1: Real eligible anomaly (Day 38-41) on GOLDPETAL
                if sym == "GOLDPETAL" and 37 <= t_idx <= 40:
                    anomaly_factor = 0.0075  # +0.75% relative dislocation (~3 sigma)

                # Case 2: Illiquidity test (Day 52) on GOLDGUINEA
                if sym == "GOLDGUINEA" and t_idx == 52:
                    anomaly_factor = -0.008

            pure_contract_price = base_pure_g * (1.0 + basis_noise + anomaly_factor)

            # Invert formula to obtain the raw quoted exchange price:
            # P_i = q_i * g_i * (u_i / 1000)
            nominal_quote = pure_contract_price * g_i * purity_fraction

            # Add intraday range
            intraday_vol = np.random.uniform(0.004, 0.012)
            high_price = nominal_quote * (1.0 + intraday_vol)
            low_price = nominal_quote * (1.0 - intraday_vol)
            open_price = nominal_quote + np.random.uniform(-0.003, 0.003) * nominal_quote

            # Ensure bounds
            high_price = max(high_price, open_price, nominal_quote)
            low_price = min(low_price, open_price, nominal_quote)

            # Volume & OI
            vol = max(5, int(np.random.normal(vol_base, vol_sd)))
            oi = max(10, int(oi_base + np.random.normal(0, oi_base * 0.05)))

            # If Day 52 illiquid test for GOLDGUINEA, drop volume to 2 contracts
            if sym == "GOLDGUINEA" and t_idx == 52:
                vol = 3
                oi = 8

            turnover = (nominal_quote * vol * (specs[sym][0] / g_i)) / 100000.0

            all_rows.append(
                {
                    "trading_date": t_date.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "expiry_date": expiry_date.strftime("%Y-%m-%d"),
                    "open_price": round(open_price, 2),
                    "high_price": round(high_price, 2),
                    "low_price": round(low_price, 2),
                    "close_price": round(nominal_quote, 2),
                    "settle_price": round(nominal_quote, 2),
                    "volume": vol,
                    "open_interest": oi,
                    "turnover_lakhs": round(turnover, 2),
                    "provenance": "SYNTHETIC",
                }
            )

    df_synth = pd.DataFrame(all_rows)

    # Persist sample files
    target_dir = output_dir or SAMPLE_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_out = target_dir / "mcx_gold_sample_synthetic.csv"
    parquet_out = target_dir / "mcx_gold_sample_synthetic.parquet"

    df_synth.to_csv(csv_out, index=False)
    df_synth.to_parquet(parquet_out, index=False)

    return df_synth


def load_synthetic_demo_data() -> pd.DataFrame:
    """Load or generate deterministic synthetic demo dataset."""
    sample_parquet = SAMPLE_DATA_DIR / "mcx_gold_sample_synthetic.parquet"
    if sample_parquet.exists():
        return pd.read_parquet(sample_parquet)

    sample_csv = SAMPLE_DATA_DIR / "mcx_gold_sample_synthetic.csv"
    if sample_csv.exists():
        return pd.read_csv(sample_csv)

    return generate_synthetic_gold_dataset()
