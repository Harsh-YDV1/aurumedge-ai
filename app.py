"""AurumEdge AI — Beyond Gold Prices. Into Market Intelligence.

Production-Quality Commodity Derivatives Intelligence Platform.
Relative-Value Analysis across MCX Gold Futures Contracts:
GOLDM, GOLDTEN, GOLDGUINEA, GOLDPETAL (and GOLD reference).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.alert_cards import (
    render_data_mode_badge,
    render_normalization_explainer_card,
    render_signal_alert_card,
)
from components.charts import (
    plot_backtest_equity,
    plot_liquidity_lifecycle,
    plot_normalized_prices,
    plot_relative_spread_bands,
    plot_spread_distribution,
    plot_z_score_series,
)
from components.tables import (
    render_contract_master_table,
    render_data_quality_report,
    render_signals_audit_table,
    render_trade_ledger_table,
)
from src.backtest import run_pairs_backtest, run_signal_study
from src.demo_data import load_synthetic_demo_data
from src.ingest import ingest_bhavcopy_files
from src.matching import find_available_expiries, match_contract_pair
from src.normalization import (
    load_contract_master,
    normalize_dataframe,
    normalize_single_observation,
)
from src.signals import generate_relative_value_signals
from src.storage import load_settings

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="AurumEdge AI — MCX Gold Intelligence",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Institutional Dark Navy Theme */
    .stApp {
        background-color: #0A1128;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    header[data-testid="stHeader"] {
        background-color: #0A1128;
    }
    section[data-testid="stSidebar"] {
        background-color: #060B1A;
        border-right: 1px solid #1E293B;
    }
    div[data-testid="stMetric"] {
        background-color: #132238;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetricValue"] {
        color: #D4AF37 !important;
        font-weight: 700;
        font-size: 24px;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0B132B;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1E293B;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #94A3B8;
        font-weight: 500;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #132238 !important;
        color: #D4AF37 !important;
        font-weight: 700;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_contract_master() -> pd.DataFrame:
    return load_contract_master()


@st.cache_data(show_spinner=False)
def get_demo_dataset() -> pd.DataFrame:
    return load_synthetic_demo_data()


# --- Load Base Settings & Master ---
settings = load_settings()
contract_master = get_contract_master()
all_symbols = sorted(contract_master["symbol"].unique().tolist())

# --- Sidebar Controls ---
st.sidebar.markdown(
    """
    <div style="padding: 10px 0; border-bottom: 1px solid #1E293B; margin-bottom: 15px;">
        <h2 style="color: #D4AF37; margin: 0; font-size: 20px; font-weight: 800; letter-spacing: -0.5px;">AurumEdge AI</h2>
        <div style="color: #94A3B8; font-size: 11px;">Commodity Derivatives Intelligence</div>
        <div style="color: #64748B; font-size: 10px; margin-top: 2px;">Team Proxy | Research Prototype</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.subheader("1. Data Ingestion Mode")
data_mode = st.sidebar.radio(
    "Data Source",
    options=["Demo Synthetic Data", "Upload MCX Bhavcopy CSV", "Ingest data/raw/ Directory"],
    index=0,
)

raw_df = pd.DataFrame()
is_synthetic = True
ingestion_summary = {}

if data_mode == "Demo Synthetic Data":
    raw_df = get_demo_dataset()
    is_synthetic = True
    ingestion_summary = {
        "records_read": len(raw_df),
        "records_accepted": len(raw_df),
        "records_rejected": 0,
        "contracts_found": sorted(raw_df["symbol"].unique().tolist()),
    }
elif data_mode == "Upload MCX Bhavcopy CSV":
    uploaded_files = st.sidebar.file_uploader(
        "Upload MCX Bhavcopy CSVs",
        type=["csv"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        temp_paths = []
        import tempfile
        from pathlib import Path

        for uf in uploaded_files:
            tfile = Path(tempfile.gettempdir()) / uf.name
            tfile.write_bytes(uf.getvalue())
            temp_paths.append(tfile)
        clean_df, rej_df, summary = ingest_bhavcopy_files(file_paths=temp_paths)
        raw_df = clean_df
        is_synthetic = False
        ingestion_summary = summary.to_dict()
    else:
        st.sidebar.info("Upload CSV files above, or revert to Demo Mode.")
        raw_df = get_demo_dataset()
        is_synthetic = True
else:
    clean_df, rej_df, summary = ingest_bhavcopy_files()
    if clean_df.empty:
        st.sidebar.warning("No files found in data/raw/. Falling back to Demo mode.")
        raw_df = get_demo_dataset()
        is_synthetic = True
    else:
        raw_df = clean_df
        is_synthetic = False
        ingestion_summary = summary.to_dict()

# Normalize dataset
normalized_market_df = normalize_dataframe(raw_df, contract_master=contract_master)

# --- Sidebar Contract Pairing ---
st.sidebar.subheader("2. Contract Pair Selection")
default_sym_a = "GOLDM" if "GOLDM" in all_symbols else all_symbols[0]
default_sym_b = "GOLDPETAL" if "GOLDPETAL" in all_symbols else all_symbols[1]

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    sym_a = st.selectbox("Symbol A", options=all_symbols, index=all_symbols.index(default_sym_a))
with col_s2:
    available_b = [s for s in all_symbols if s != sym_a]
    sym_b = st.selectbox(
        "Symbol B",
        options=available_b,
        index=available_b.index(default_sym_b) if default_sym_b in available_b else 0,
    )

# Detect common expiries
common_expiries = find_available_expiries(normalized_market_df, sym_a, sym_b)
chosen_expiry = None
if common_expiries:
    chosen_expiry = st.sidebar.selectbox("Contract Expiration Date", options=common_expiries, index=0)
else:
    st.sidebar.error(f"No shared expiry dates found for {sym_a} and {sym_b}.")

# --- Sidebar Quantitative Signal Settings ---
st.sidebar.subheader("3. Relative-Value Signal Engine")
rolling_window = st.sidebar.slider("Rolling Baseline Window (Bars)", min_value=10, max_value=60, value=20, step=1)
z_threshold = st.sidebar.slider("Anomaly Z-Score Threshold (σ)", min_value=1.0, max_value=4.0, value=2.0, step=0.1)
min_history = st.sidebar.slider("Minimum Overlap History (Bars)", min_value=5, max_value=30, value=15, step=1)

# Filters
with st.sidebar.expander("Eligibility & Liquidity Filters", expanded=False):
    min_vol = st.number_input("Min Volume (Contracts)", min_value=1, value=10, step=5)
    min_oi = st.number_input("Min Open Interest", min_value=1, value=20, step=10)
    expiry_buf = st.number_input("Tender Buffer (Days to Expiry)", min_value=1, value=3, step=1)

# Backtest Controls
with st.sidebar.expander("Backtest & Transaction Cost Settings", expanded=False):
    init_cash = st.number_input("Starting Capital (₹)", value=1_000_000.0, step=100_000.0)
    exposure_grams = st.number_input("Target Exposure (Pure Grams)", value=100.0, step=10.0)
    exit_z = st.number_input("Exit Z-Score", value=0.5, step=0.1)
    stop_z = st.number_input("Stop Loss Z-Score", value=3.5, step=0.1)
    apply_fees = st.checkbox("Include MCX Statutory Levies & Slippage", value=True)

# --- Match Contract Pair ---
paired_df, match_diag = match_contract_pair(
    normalized_market_df,
    symbol_a=sym_a,
    symbol_b=sym_b,
    target_expiry=chosen_expiry,
    min_overlap_bars=min_history,
    expiry_buffer_days=expiry_buf,
)

# --- Generate Signals ---
signals_df, signal_summary = generate_relative_value_signals(
    paired_df,
    rolling_window=rolling_window,
    z_score_threshold=z_threshold,
    min_history_bars=min_history,
    min_volume=min_vol,
    min_open_interest=min_oi,
    expiry_buffer_days=expiry_buf,
)

# --- Application Header ---
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 8px;">
        <div>
            <h1 style="color: #F8FAFC; margin: 0; font-size: 28px; font-weight: 800; letter-spacing: -0.5px;">
                AurumEdge AI <span style="color: #D4AF37; font-weight: 300;">| Bullion Intelligence</span>
            </h1>
            <div style="color: #94A3B8; font-size: 13px; margin-top: 3px;">
                Beyond Gold Prices. Into Market Intelligence. Relative-Value Analytics for MCX Gold Derivatives.
            </div>
        </div>
        <div style="text-align: right;">
            <span style="color: #64748B; font-size: 11px;">DEVELOPED BY TEAM PROXY</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_data_mode_badge(is_synthetic=is_synthetic)

# Research Disclaimer Callout
st.markdown(
    """
    <div style="background: rgba(19, 34, 56, 0.4); border-left: 3px solid #D4AF37; padding: 6px 12px; margin-bottom: 16px; font-size: 11px; color: #94A3B8;">
        <b>RESEARCH PROTOTYPE NOTICE:</b> AurumEdge AI is a quantitative decision-support system. A relative price difference is not automatically an opportunity. First, it must be measured correctly. Never imply an anomaly guarantees convergence, risk-free arbitrage, or executable profit.
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Tab Navigation ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Executive Overview",
    "Contract Master & Normalization",
    "Relative-Spread & Anomalies",
    "Liquidity & Lifecycle",
    "Alerts & Explainability",
    "Historical Backtesting",
    "Data Quality & Provenance",
    "Methodology & Integrity",
])

# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ==============================================================================
with tab1:
    if signals_df.empty:
        st.warning(f"Unable to align pair {sym_a} and {sym_b}. Reason: {', '.join(match_diag.rejection_reasons)}")
    else:
        latest = signals_df.iloc[-1]
        dte = int(latest["days_to_expiry"])
        cur_z = float(latest["z_score"])
        cur_spread = (float(latest["spread"])) * 100.0

        # KPI Metrics Row
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        with mcol1:
            st.metric("Pair Monitored", f"{sym_a} / {sym_b}")
        with mcol2:
            st.metric("Latest Session Date", str(latest["trading_date"])[:10])
        with mcol3:
            st.metric("Relative Spread", f"{cur_spread:+.2f}%")
        with mcol4:
            st.metric("Current Z-Score", f"{cur_z:+.2f}σ")
        with mcol5:
            st.metric("Days to Expiry", f"{dte} Days")

        # Active Anomaly Alert Banner
        render_signal_alert_card(latest, z_threshold)

        # Comparative overview charts
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(plot_normalized_prices(signals_df, sym_a, sym_b), use_container_width=True)
        with chart_col2:
            st.plotly_chart(plot_relative_spread_bands(signals_df, sym_a, sym_b, z_threshold), use_container_width=True)

# ==============================================================================
# TAB 2: CONTRACT MASTER & NORMALIZATION
# ==============================================================================
with tab2:
    st.markdown("### Official MCX Bullion Contract Master")
    st.markdown(
        "Verification status of quotation units, trading lot units, and purity standards across MCX gold contracts."
    )
    render_contract_master_table(contract_master)

    st.markdown("---")
    st.markdown("### Algorithmic Price Normalization Decomposition")

    # If pair data is available, show decomposition of latest prices
    if not signals_df.empty:
        latest_row = signals_df.iloc[-1]
        obs_a = normalize_single_observation(sym_a, float(latest_row["raw_price_A"]), contract_master)
        obs_b = normalize_single_observation(sym_b, float(latest_row["raw_price_B"]), contract_master)
        render_normalization_explainer_card(obs_a, obs_b)

    # Interactive Calculator Widget
    with st.expander("Live Interactive Normalization Calculator", expanded=True):
        st.markdown(
            "Test arbitrary quoted prices and verify the purity-adjusted fine-gold equivalent calculation live:"
        )
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            calc_sym = st.selectbox("Select Contract Specification", options=all_symbols, index=all_symbols.index(sym_a))
        with cc2:
            spec_row = contract_master[contract_master["symbol"] == calc_sym].iloc[0]
            default_p = float(spec_row["quotation_unit_grams"]) * 7300.0 * (float(spec_row["purity_per_mille"]) / 1000.0)
            test_price = st.number_input("Test Quoted Market Price (₹)", value=round(default_p, 2), step=100.0)
        with cc3:
            calc_obs = normalize_single_observation(calc_sym, test_price, contract_master)
            st.metric(
                label=f"Derived Pure Gold Price ({calc_sym})",
                value=f"₹{calc_obs['pure_gold_equivalent_price']:,.2f} / g",
            )
        st.code(calc_obs["explanation"], language="markdown")

# ==============================================================================
# TAB 3: SPREAD ANALYSIS & ANOMALY DIAGNOSTICS
# ==============================================================================
with tab3:
    if not signals_df.empty:
        st.plotly_chart(plot_relative_spread_bands(signals_df, sym_a, sym_b, z_threshold), use_container_width=True)
        st.plotly_chart(plot_z_score_series(signals_df, z_threshold), use_container_width=True)

        sc1, sc2 = st.columns([0.6, 0.4])
        with sc1:
            st.plotly_chart(plot_spread_distribution(signals_df), use_container_width=True)
        with sc2:
            st.markdown("#### Statistical Diagnostics")
            clean_s = signals_df["spread"].dropna()
            st.markdown(
                f"""
                - **Sample Observations**: `{len(clean_s)} trading sessions`
                - **Mean Log Spread**: `{clean_s.mean():+.5f}`
                - **Standard Deviation**: `{clean_s.std():.5f}`
                - **Median Absolute Deviation (MAD)**: `{signals_df['rolling_mad'].dropna().mean():.5f}`
                - **Skewness**: `{clean_s.skew():.2f}`
                - **Excess Kurtosis**: `{clean_s.kurt():.2f}`
                - **Anti-Look-Ahead**: Strictly computed over past $[t-W, t-1]$ sessions.
                """
            )
    else:
        st.info("No paired data to analyze.")

# ==============================================================================
# TAB 4: LIQUIDITY & LIFECYCLE DYNAMICS
# ==============================================================================
with tab4:
    if not signals_df.empty:
        st.plotly_chart(plot_liquidity_lifecycle(signals_df, sym_a, sym_b), use_container_width=True)

        st.markdown("#### Contract Lifecycle Status")
        lc_col1, lc_col2, lc_col3 = st.columns(3)
        with lc_col1:
            st.metric("Current Expiry Date", chosen_expiry or "N/A")
        with lc_col2:
            st.metric("Tender Period Buffer", f"{expiry_buf} Business Days")
        with lc_col3:
            latest_status = signals_df["lifecycle_state"].iloc[-1]
            st.metric("Lifecycle Phase", latest_status)
    else:
        st.info("No paired data to analyze.")

# ==============================================================================
# TAB 5: ALERTS & EXPLAINABILITY
# ==============================================================================
with tab5:
    st.markdown("### Algorithmic Anomaly Audit Ledger")
    st.markdown(
        "Complete historical record of candidate relative-value dislocations with deterministic filtering rationale."
    )

    if not signals_df.empty:
        anom_count = signal_summary.candidate_anomalies
        eligible_count = signal_summary.eligible_signals
        flt_liq = signal_summary.filtered_liquidity
        flt_exp = signal_summary.filtered_expiry
        flt_hist = signal_summary.filtered_history

        ac1, ac2, ac3, ac4, ac5 = st.columns(5)
        with ac1:
            st.metric("Total Anomalies", f"{anom_count}")
        with ac2:
            st.metric("Eligible Signals", f"{eligible_count}")
        with ac3:
            st.metric("Filtered (Liquidity)", f"{flt_liq}")
        with ac4:
            st.metric("Filtered (Near Expiry)", f"{flt_exp}")
        with ac5:
            st.metric("Filtered (History)", f"{flt_hist}")

        render_signals_audit_table(signals_df)

        # Download Signals
        csv_signals = signals_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Signal Ledger (CSV)",
            data=csv_signals,
            file_name=f"aurumedge_signals_{sym_a}_{sym_b}.csv",
            mime="text/csv",
        )
    else:
        st.info("No signals generated.")

# ==============================================================================
# TAB 6: HISTORICAL RESEARCH BACKTESTING
# ==============================================================================
with tab6:
    st.markdown("### Historical Research Evaluation")
    st.markdown(
        "Walk-forward chronological simulation with **strict next-bar (t+1) execution**, "
        "lot-size gram exposure matching, and official Indian MCX commodity transaction levies."
    )

    bt_mode = st.radio("Backtest Mode", options=["Executable Strategy Simulation", "Signal Mean-Reversion Study"], horizontal=True)

    if not signals_df.empty:
        cost_cfg = (
            {
                "brokerage_pct": float(settings["costs"]["brokerage_pct"]),
                "exchange_pct": float(settings["costs"]["exchange_turnover_pct"]),
                "sebi_pct": float(settings["costs"]["sebi_turnover_pct"]),
                "ctt_sell_pct": float(settings["costs"]["ctt_sell_pct"]),
                "stamp_duty_buy_pct": float(settings["costs"]["stamp_duty_buy_pct"]),
                "gst_pct": float(settings["costs"]["gst_pct"]),
                "slippage_per_leg": float(settings["costs"]["slippage_ticks"]),
            }
            if apply_fees
            else {
                "brokerage_pct": 0.0,
                "exchange_pct": 0.0,
                "sebi_pct": 0.0,
                "ctt_sell_pct": 0.0,
                "stamp_duty_buy_pct": 0.0,
                "gst_pct": 0.0,
                "slippage_per_leg": 0.0,
            }
        )

        bt_result = run_pairs_backtest(
            signals_df,
            initial_capital=init_cash,
            entry_z=z_threshold,
            exit_z=exit_z,
            stop_loss_z=stop_z,
            target_exposure_grams=exposure_grams,
            cost_config=cost_cfg,
        )

        if bt_mode == "Executable Strategy Simulation":
            # Display Warnings
            for w in bt_result.warnings:
                st.warning(f"**Research Caveat:** {w}")

            # Performance Metrics
            b1, b2, b3, b4, b5 = st.columns(5)
            with b1:
                st.metric("Total Net PnL", f"₹{bt_result.total_net_pnl:,.2f}", delta=f"{bt_result.strategy_return_pct:+.2f}%")
            with b2:
                st.metric("Gross PnL", f"₹{bt_result.total_gross_pnl:,.2f}")
            with b3:
                st.metric("Transaction Fees & CTT", f"₹{bt_result.total_costs:,.2f}")
            with b4:
                st.metric("Executed Trades", f"{bt_result.total_trades}")
            with b5:
                st.metric("Win Rate", f"{bt_result.win_rate_pct:.1f}%")

            b6, b7, b8, b9, b10 = st.columns(5)
            with b6:
                st.metric("Max Drawdown", f"{bt_result.max_drawdown_pct:.2f}%")
            with b7:
                st.metric("Sharpe Ratio (Annualized)", f"{bt_result.sharpe_ratio:.2f}")
            with b8:
                st.metric("Sortino Ratio", f"{bt_result.sortino_ratio:.2f}")
            with b9:
                st.metric("Benchmark Return", f"{bt_result.benchmark_return_pct:+.2f}%")
            with b10:
                cost_ratio = (bt_result.total_costs / abs(bt_result.total_gross_pnl) * 100) if bt_result.total_gross_pnl != 0 else 0
                st.metric("Cost Drag Ratio", f"{cost_ratio:.1f}%")

            st.plotly_chart(plot_backtest_equity(bt_result.equity_curve, init_cash), use_container_width=True)

            st.markdown("#### Completed Trade Execution Ledger")
            render_trade_ledger_table(bt_result.trade_ledger)

        else:
            # Signal Study Mode
            st.markdown("#### Forward Mean-Reversion Horizon Study")
            st.markdown(
                "Evaluates whether candidate anomalies empirically reverted toward the rolling mean over 1, 3, 5, and 10 days, without assuming trade execution."
            )
            study = run_signal_study(signals_df)
            if "horizon_reversion_results" in study:
                reversion_df = pd.DataFrame(
                    [
                        {
                            "Horizon": k,
                            "Anomalies Evaluated": v["eval_count"],
                            "Reversion Rate (%)": f"{v['reversion_rate_pct']:.1f}%",
                            "Avg Reversion Move": f"{v['avg_reversion_bps']:+.1f} bps",
                        }
                        for k, v in study["horizon_reversion_results"].items()
                    ]
                )
                st.dataframe(reversion_df, use_container_width=True, hide_index=True)
            else:
                st.info("No anomalies available to evaluate in the signal study.")

# ==============================================================================
# TAB 7: DATA QUALITY & PROVENANCE
# ==============================================================================
with tab7:
    st.markdown("### Data Ingestion & Quality Audit")
    render_data_quality_report(ingestion_summary)

    st.markdown("---")
    st.markdown("#### Data Schema & Source Provenance")
    st.markdown(
        f"""
        - **Current Data Mode**: `{'SYNTHETIC DEMO' if is_synthetic else 'OFFICIAL MCX DATA'}`
        - **Total Clean Observations**: `{len(raw_df):,} records`
        - **Active Universe**: `{', '.join(ingestion_summary.get('contracts_found', all_symbols))}`
        - **Storage Lineage**: Clean datasets cached as Parquet and CSV in `data/processed/`.
        """
    )

    with st.expander("Inspect Raw Ingested Sample (First 20 Records)"):
        st.dataframe(raw_df.head(20), use_container_width=True)

# ==============================================================================
# TAB 8: METHODOLOGY & INTEGRITY
# ==============================================================================
with tab8:
    st.markdown("### Quantitative Methodology & Research Integrity")
    st.markdown(
        """
        #### 1. Core Philosophy
        > **"A price difference is not automatically an opportunity. First, it must be measured correctly."**

        MCX gold futures are quoted in different weight bases (₹/10g, ₹/8g, ₹/1g) and purity grades (995 vs 999). Naive price comparisons generate misleading signals.

        #### 2. Normalization Formula
        $$q_i = \\frac{P_i}{g_i \\times \\left(\\frac{u_i}{1000}\\right)}$$
        * $P_i$: Quoted exchange price.
        * $g_i$: Quotation-unit weight in grams (e.g. 10g for GOLDM, 8g for GOLDGUINEA, 1g for GOLDPETAL).
        * $u_i$: Purity expressed per 1,000 (fineness, e.g. 995 for GOLDM, 999 for GOLDPETAL).
        * $q_i$: Purity-adjusted price per gram of 100.0% pure fine-gold equivalent.

        #### 3. Zero Look-Ahead Rolling Statistics
        All rolling means $\\mu_t$ and standard deviations $\\sigma_t$ are strictly computed over $[t-W, t-1]$, excluding bar $t$:
        $$s_t = \\ln(q_{A, t} / q_{B, t})$$
        $$z_t = \\frac{s_t - \\mu_t}{\\sigma_t}$$

        #### 4. The 5-Stage Signal Hierarchy
        1. **Valid Normalized Observation**: Validated quote converted to pure gold.
        2. **Statistically Unusual Relative Spread**: $|Z| > 2.0\\sigma$.
        3. **Candidate Research Signal**: Point-in-time verified anomaly.
        4. **Eligible Signal**: Passed minimum volume, open interest, and tender proximity buffers.
        5. **Historically Tested Result**: Walk-forward backtest after accounting for all statutory exchange costs.

        #### 5. Indian MCX Transaction Costs Schedule
        * **Brokerage**: 0.02% (2 bps)
        * **MCX Exchange Turnover Fee**: 0.0021%
        * **SEBI Regulatory Fee**: 0.0001%
        * **Commodity Transaction Tax (CTT)**: 0.01% on sell turnover
        * **Stamp Duty**: 0.002% on buy turnover
        * **GST**: 18% on (Brokerage + Exchange Fees)
        * **Slippage**: 1 tick per leg
        """
    )
