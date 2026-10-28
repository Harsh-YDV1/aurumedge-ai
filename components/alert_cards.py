"""AurumEdge AI - Metric Cards, Alert Banners & Explainability Cards."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_data_mode_badge(is_synthetic: bool = True) -> None:
    """Render prominent visual safety badge distinguishing real and demo datasets."""
    if is_synthetic:
        st.markdown(
            """
            <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid #F59E0B; border-radius: 6px; padding: 6px 12px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="background: #F59E0B; color: #000; font-weight: 700; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-right: 8px;">DEMO MODE</span>
                    <span style="color: #FCD34D; font-size: 13px; font-weight: 500;">Deterministic Synthetic Data Active — High-fidelity cointegrated simulation with injected anomalies.</span>
                </div>
                <span style="color: #FDE68A; font-size: 11px;">PROVENANCE: SYNTHETIC</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 6px; padding: 6px 12px; margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="background: #10B981; color: #000; font-weight: 700; padding: 2px 8px; border-radius: 4px; font-size: 11px; margin-right: 8px;">REAL DATA</span>
                    <span style="color: #6EE7B7; font-size: 13px; font-weight: 500;">MCX Official Bhavcopy Verified — Cleaned & validated market observations.</span>
                </div>
                <span style="color: #A7F3D0; font-size: 11px;">PROVENANCE: MCX_RAW</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_signal_alert_card(latest_row: pd.Series, z_thresh: float) -> None:
    """Render prominent visual alert banner for the current market state."""
    status = latest_row.get("signal_status", "NORMAL")
    z = float(latest_row.get("z_score", 0.0))
    spread_pct = (float(latest_row.get("spread", 0.0))) * 100.0
    explanation = str(latest_row.get("explanation", ""))
    dte = int(latest_row.get("days_to_expiry", 0))

    if status == "ELIGIBLE_SIGNAL":
        color = "#EF4444"
        bg = "rgba(239, 68, 68, 0.15)"
        border = "#EF4444"
        badge = "STATISTICALLY UNUSUAL SPREAD (ELIGIBLE)"
        header_text = f"Candidate Relative-Value Anomaly Detected ({abs(z):.2f}σ)"
    elif "FILTERED" in status:
        color = "#F59E0B"
        bg = "rgba(245, 158, 11, 0.15)"
        border = "#F59E0B"
        badge = f"STATISTICAL OUTLIER ({status})"
        header_text = f"Outlier Filtered by Operational Rules ({abs(z):.2f}σ)"
    else:
        color = "#10B981"
        bg = "rgba(16, 185, 129, 0.1)"
        border = "#10B981"
        badge = "NORMAL MARKET RELATIONSHIP"
        header_text = f"Spread within Expected Bounds ({z:+.2f}σ)"

    st.markdown(
        f"""
        <div style="background: {bg}; border: 1px solid {border}; border-radius: 8px; padding: 14px 18px; margin: 10px 0 20px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="color: {color}; font-weight: 700; font-size: 15px;">{header_text}</span>
                <span style="background: {color}; color: #000; font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 4px;">{badge}</span>
            </div>
            <p style="color: #E2E8F0; font-size: 13px; margin: 0 0 8px 0; line-height: 1.5;">{explanation}</p>
            <div style="display: flex; gap: 24px; font-size: 12px; color: #94A3B8;">
                <span>Spread: <b style="color: #FFF;">{spread_pct:+.2f}%</b></span>
                <span>Z-Score: <b style="color: #FFF;">{z:+.2f}σ</b> (Thresh: ±{z_thresh:.1f}σ)</span>
                <span>Days to Expiry: <b style="color: #FFF;">{dte} Days</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_normalization_explainer_card(obs_a: dict, obs_b: dict) -> None:
    """Render side-by-side comparison explaining the normalization mathematics."""
    st.markdown(
        """
        <div style="background: #132238; border: 1px solid #1E293B; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
            <h4 style="color: #D4AF37; margin: 0 0 10px 0; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;">
                Algorithmic Normalization Breakdown: q_i = P_i / [g_i × (u_i / 1000)]
            </h4>
            <p style="color: #94A3B8; font-size: 12px; margin-bottom: 14px;">
                Quotes are converted to fine-gold-equivalent prices per single gram of 100.0% pure gold.
                Lot size (trading unit) is strictly preserved for position sizing and never substituted for quotation unit.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="background: #0B132B; border: 1px solid #1E293B; border-radius: 6px; padding: 14px;">
                <div style="color: #D4AF37; font-weight: 700; font-size: 16px; margin-bottom: 8px;">{obs_a['symbol']}</div>
                <div style="font-size: 12px; color: #94A3B8; line-height: 1.8;">
                    Quoted Market Price (P_A): <b style="color: #FFF;">₹{obs_a['raw_price']:,.2f}</b><br>
                    Quotation Unit (g_A): <b style="color: #FFF;">{obs_a['quotation_unit_grams']}g</b><br>
                    Trading Unit / Lot (L_A): <b style="color: #FFF;">{obs_a['trading_unit_grams']}g</b><br>
                    Purity / Fineness (u_A): <b style="color: #FFF;">{obs_a['purity_per_mille']}‰ ({obs_a['purity_fraction']*100:.1f}%)</b><br>
                    Nominal Per-Gram: <b style="color: #FFF;">₹{obs_a['raw_per_gram']:,.2f}/g</b><br>
                    <hr style="border-color: #1E293B; margin: 8px 0;">
                    <span style="color: #D4AF37; font-size: 13px; font-weight: 700;">Pure Gold Equivalent (q_A): ₹{obs_a['pure_gold_equivalent_price']:,.3f} / g</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="background: #0B132B; border: 1px solid #1E293B; border-radius: 6px; padding: 14px;">
                <div style="color: #38BDF8; font-weight: 700; font-size: 16px; margin-bottom: 8px;">{obs_b['symbol']}</div>
                <div style="font-size: 12px; color: #94A3B8; line-height: 1.8;">
                    Quoted Market Price (P_B): <b style="color: #FFF;">₹{obs_b['raw_price']:,.2f}</b><br>
                    Quotation Unit (g_B): <b style="color: #FFF;">{obs_b['quotation_unit_grams']}g</b><br>
                    Trading Unit / Lot (L_B): <b style="color: #FFF;">{obs_b['trading_unit_grams']}g</b><br>
                    Purity / Fineness (u_B): <b style="color: #FFF;">{obs_b['purity_per_mille']}‰ ({obs_b['purity_fraction']*100:.1f}%)</b><br>
                    Nominal Per-Gram: <b style="color: #FFF;">₹{obs_b['raw_per_gram']:,.2f}/g</b><br>
                    <hr style="border-color: #1E293B; margin: 8px 0;">
                    <span style="color: #38BDF8; font-size: 13px; font-weight: 700;">Pure Gold Equivalent (q_B): ₹{obs_b['pure_gold_equivalent_price']:,.3f} / g</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
