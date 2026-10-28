"""AurumEdge AI - Storage & Configuration Utilities.

Handles reading/writing data frames (Parquet/CSV), settings parsing,
and storage directory management.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
SETTINGS_FILE = CONFIG_DIR / "settings.toml"
CONTRACTS_FILE = CONFIG_DIR / "contracts.csv"


def ensure_directories() -> None:
    """Ensure all required workspace data and config directories exist."""
    for directory in [CONFIG_DIR, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_settings(settings_path: Path | None = None) -> dict[str, Any]:
    """Load settings from TOML configuration file.

    Falls back to safe defaults if settings file is not found.
    """
    path = settings_path or SETTINGS_FILE
    if not path.exists():
        return {
            "app": {"name": "AurumEdge AI", "version": "1.0.0"},
            "signals": {
                "rolling_window": 20,
                "z_score_threshold": 2.0,
                "exit_z_score": 0.5,
                "stop_loss_z": 3.5,
                "min_history_bars": 15,
                "mad_threshold": 2.5,
            },
            "filters": {
                "min_volume": 10,
                "min_open_interest": 20,
                "expiry_buffer_days": 3,
                "require_exact_expiry_match": True,
            },
            "costs": {
                "brokerage_pct": 0.0002,
                "exchange_turnover_pct": 0.000021,
                "sebi_turnover_pct": 0.000001,
                "ctt_sell_pct": 0.0001,
                "stamp_duty_buy_pct": 0.00002,
                "gst_pct": 0.18,
                "slippage_ticks": 1.0,
            },
            "backtest": {
                "initial_capital": 1000000.0,
                "max_holding_days": 15,
                "margin_pct": 0.12,
                "risk_free_rate_pct": 0.065,
            },
        }

    with open(path, "rb") as f:
        return tomllib.load(f)


def save_processed_data(df: pd.DataFrame, filename_stem: str, format_type: str = "parquet") -> Path:
    """Save processed dataframe to the processed directory."""
    ensure_directories()
    if format_type.lower() == "parquet":
        out_path = PROCESSED_DATA_DIR / f"{filename_stem}.parquet"
        df.to_parquet(out_path, index=False)
    else:
        out_path = PROCESSED_DATA_DIR / f"{filename_stem}.csv"
        df.to_csv(out_path, index=False)
    return out_path


def load_processed_data(filename_stem: str) -> pd.DataFrame | None:
    """Load processed dataset, checking Parquet first then CSV."""
    parquet_path = PROCESSED_DATA_DIR / f"{filename_stem}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_path = PROCESSED_DATA_DIR / f"{filename_stem}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None
