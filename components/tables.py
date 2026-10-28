"""AurumEdge AI - Formatted Data Tables & Reporting Displays.

Provides clean, financial-grade tabular views for contract specifications,
anomaly audit logs, trade ledgers, and data-quality reports.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_contract_master_table(df: pd.DataFrame) -> None:
    """Render formatted contract master specifications table."""
    display_df = df.copy()
    col_order = [
        "symbol",
        "name",
        "trading_unit_grams",
        "quotation_unit_grams",
        "purity_per_mille",
        "tick_size",
        "delivery_type",
        "notes",
    ]
    cols = [c for c in col_order if c in display_df.columns]
    display_df = display_df[cols].rename(
        columns={
            "symbol": "Symbol",
            "name": "Contract Name",
            "trading_unit_grams": "Trading Unit (Lot g)",
            "quotation_unit_grams": "Quotation Unit (Quote g)",
            "purity_per_mille": "Purity (‰)",
            "tick_size": "Tick (₹)",
            "delivery_type": "Settlement Type",
            "notes": "Contract Description",
        }
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_signals_audit_table(signals_df: pd.DataFrame, max_rows: int = 50) -> None:
    """Render anomaly candidate signals ledger with filter statuses and explanations."""
    if signals_df.empty:
        st.info("No signal records available.")
        return

    # Filter to interesting rows (anomalies or recent records)
    anomalies = signals_df[signals_df["is_candidate_anomaly"]].copy()
    if anomalies.empty:
        st.write("No candidate statistical anomalies found in the selected sample.")
        display_df = signals_df.tail(15).copy()
    else:
        display_df = anomalies.copy()

    display_df["spread_pct"] = (display_df["spread"] * 100.0).round(2)
    display_df["z_score"] = display_df["z_score"].round(2)
    display_df["trading_date"] = pd.to_datetime(display_df["trading_date"]).dt.strftime("%Y-%m-%d")

    cols = [
        "trading_date",
        "pair_id",
        "q_A",
        "q_B",
        "spread_pct",
        "z_score",
        "signal_status",
        "explanation",
    ]
    renames = {
        "trading_date": "Date",
        "pair_id": "Pair",
        "q_A": "q_A (₹/g)",
        "q_B": "q_B (₹/g)",
        "spread_pct": "Spread (%)",
        "z_score": "Z-Score",
        "signal_status": "Classification",
        "explanation": "Algorithmic Explanation",
    }
    subset = display_df[[c for c in cols if c in display_df.columns]].rename(columns=renames)
    st.dataframe(
        subset.tail(max_rows).sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def render_trade_ledger_table(trades_df: pd.DataFrame) -> None:
    """Render strategy backtest trade execution ledger."""
    if trades_df.empty:
        st.info("No hypothetical trades executed by the strategy.")
        return

    display_df = trades_df.copy()
    display_df["entry_date"] = pd.to_datetime(display_df["entry_date"]).dt.strftime("%Y-%m-%d")
    display_df["exit_date"] = pd.to_datetime(display_df["exit_date"]).dt.strftime("%Y-%m-%d")
    display_df["gross_pnl"] = display_df["gross_pnl"].round(2)
    display_df["total_cost"] = display_df["total_cost"].round(2)
    display_df["net_pnl"] = display_df["net_pnl"].round(2)
    display_df["entry_z"] = display_df["entry_z"].round(2)
    display_df["exit_z"] = display_df["exit_z"].round(2)

    cols = [
        "trade_id",
        "direction",
        "entry_date",
        "exit_date",
        "holding_days",
        "entry_z",
        "exit_z",
        "gross_pnl",
        "total_cost",
        "net_pnl",
        "exit_reason",
    ]
    renames = {
        "trade_id": "ID",
        "direction": "Direction",
        "entry_date": "Entry (t+1)",
        "exit_date": "Exit Date",
        "holding_days": "Holding (Days)",
        "entry_z": "Entry Z",
        "exit_z": "Exit Z",
        "gross_pnl": "Gross PnL (₹)",
        "total_cost": "Costs & Taxes (₹)",
        "net_pnl": "Net PnL (₹)",
        "exit_reason": "Exit Rationale",
    }
    subset = display_df[[c for c in cols if c in display_df.columns]].rename(columns=renames)
    st.dataframe(subset, use_container_width=True, hide_index=True)


def render_data_quality_report(summary_dict: dict) -> None:
    """Render structured ingestion and validation summary cards and tables."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records Read", f"{summary_dict.get('records_read', 0):,}")
    with col2:
        st.metric("Records Accepted", f"{summary_dict.get('records_accepted', 0):,}")
    with col3:
        st.metric("Records Rejected", f"{summary_dict.get('records_rejected', 0):,}")
    with col4:
        val_rep = summary_dict.get("validation_report") or {}
        st.metric("Duplicates Filtered", f"{val_rep.get('duplicate_records', 0):,}")

    if val_rep.get("rejection_reasons"):
        st.subheader("Data Rejection Diagnostics")
        reasons_df = pd.DataFrame(
            list(val_rep["rejection_reasons"].items()),
            columns=["Rejection Reason", "Record Count"],
        )
        st.dataframe(reasons_df, use_container_width=True, hide_index=True)
