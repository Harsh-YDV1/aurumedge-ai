"""AurumEdge AI - Interactive Plotly Financial Charts.

Institutional visual design: Dark navy surfaces (#0A1128, #132238),
rich gold accents (#D4AF37), accessible contrast, clean typography.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NAVY_BG = "#0A1128"
PANEL_BG = "#132238"
GOLD_ACCENT = "#D4AF37"
GOLD_LIGHT = "#F4D06F"
TEXT_COLOR = "#F8FAFC"
MUTED_TEXT = "#94A3B8"
GRID_COLOR = "#1E293B"
GREEN_COLOR = "#10B981"
RED_COLOR = "#EF4444"
BLUE_COLOR = "#38BDF8"


def apply_theme(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Apply consistent AurumEdge AI dark navy and gold institutional styling."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 16, "color": TEXT_COLOR, "family": "Inter, sans-serif"},
            "x": 0.02,
            "y": 0.95,
        },
        paper_bgcolor=NAVY_BG,
        plot_bgcolor=PANEL_BG,
        height=height,
        margin={"l": 50, "r": 30, "t": 50, "b": 40},
        legend={
            "font": {"color": TEXT_COLOR, "size": 11},
            "bgcolor": "rgba(19, 34, 56, 0.7)",
            "bordercolor": GRID_COLOR,
            "borderwidth": 1,
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
        hoverlabel={
            "bgcolor": "#1E293B",
            "font": {"color": "#FFFFFF", "size": 12},
            "bordercolor": GOLD_ACCENT,
        },
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=GRID_COLOR,
        tickfont={"color": MUTED_TEXT, "size": 10},
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=GRID_COLOR,
        tickfont={"color": MUTED_TEXT, "size": 10},
        zeroline=False,
    )
    return fig


def plot_normalized_prices(df: pd.DataFrame, sym_a: str, sym_b: str) -> go.Figure:
    """Plot comparative normalized pure gold prices (₹ per gram)."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["trading_date"],
            y=df["q_A"],
            mode="lines",
            name=f"{sym_a} (Normalized Fine Gold ₹/g)",
            line={"color": GOLD_ACCENT, "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>" + f"{sym_a}: ₹%{{y:,.2f}}/g<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["trading_date"],
            y=df["q_B"],
            mode="lines",
            name=f"{sym_b} (Normalized Fine Gold ₹/g)",
            line={"color": BLUE_COLOR, "width": 2, "dash": "dot"},
            hovertemplate="%{x|%Y-%m-%d}<br>" + f"{sym_b}: ₹%{{y:,.2f}}/g<extra></extra>",
        )
    )

    apply_theme(fig, title=f"Normalized Fine-Gold Equivalent Prices (₹/g): {sym_a} vs {sym_b}")
    fig.update_yaxes(title_text="₹ / Gram Pure Gold", title_font={"color": MUTED_TEXT})
    return fig


def plot_relative_spread_bands(df: pd.DataFrame, sym_a: str, sym_b: str, z_thresh: float) -> go.Figure:
    """Plot relative log spread with rolling mean and upper/lower deviation bands."""
    fig = go.Figure()

    # Percentage spread for intuitive reading: (exp(spread) - 1) * 100
    spread_pct = (np.exp(df["spread"]) - 1.0) * 100.0
    mean_pct = (np.exp(df["rolling_mean"]) - 1.0) * 100.0
    upper_pct = (np.exp(df["band_upper"]) - 1.0) * 100.0
    lower_pct = (np.exp(df["band_lower"]) - 1.0) * 100.0

    # Upper band
    fig.add_trace(
        go.Scatter(
            x=df["trading_date"],
            y=upper_pct,
            mode="lines",
            name=f"+{z_thresh:.1f}σ Upper Band",
            line={"color": "rgba(212, 175, 55, 0.4)", "width": 1, "dash": "dash"},
            showlegend=True,
        )
    )

    # Lower band with fill
    fig.add_trace(
        go.Scatter(
            x=df["trading_date"],
            y=lower_pct,
            mode="lines",
            name=f"-{z_thresh:.1f}σ Lower Band",
            line={"color": "rgba(212, 175, 55, 0.4)", "width": 1, "dash": "dash"},
            fill="tonexty",
            fillcolor="rgba(212, 175, 55, 0.05)",
            showlegend=True,
        )
    )

    # Rolling Mean
    fig.add_trace(
        go.Scatter(
            x=df["trading_date"],
            y=mean_pct,
            mode="lines",
            name="Point-in-Time Rolling Mean",
            line={"color": GOLD_LIGHT, "width": 1.5},
        )
    )

    # Spread
    fig.add_trace(
        go.Scatter(
            x=df["trading_date"],
            y=spread_pct,
            mode="lines+markers",
            name=f"Observed Spread ln({sym_a}/{sym_b})",
            line={"color": TEXT_COLOR, "width": 2},
            marker={"size": 4, "color": TEXT_COLOR},
            hovertemplate="%{x|%Y-%m-%d}<br>Spread: %{y:+.2f}%<extra></extra>",
        )
    )

    # Highlight Eligible Anomalies
    anomalies = df[df["signal_status"] == "ELIGIBLE_SIGNAL"]
    if not anomalies.empty:
        anom_spread_pct = (np.exp(anomalies["spread"]) - 1.0) * 100.0
        fig.add_trace(
            go.Scatter(
                x=anomalies["trading_date"],
                y=anom_spread_pct,
                mode="markers",
                name="Eligible Statistical Anomaly",
                marker={"size": 9, "color": RED_COLOR, "symbol": "diamond", "line": {"width": 1, "color": "#FFF"}},
                hovertemplate="%{x|%Y-%m-%d}<br>ANOMALY: %{y:+.2f}%<extra></extra>",
            )
        )

    apply_theme(fig, title=f"Relative-Value Spread & Deviation Bands: {sym_a} vs {sym_b}")
    fig.update_yaxes(title_text="Relative Spread (%)", title_font={"color": MUTED_TEXT})
    return fig


def plot_z_score_series(df: pd.DataFrame, z_thresh: float) -> go.Figure:
    """Plot point-in-time Z-score history with threshold bars."""
    fig = go.Figure()

    # Threshold horizontal lines
    fig.add_hline(y=z_thresh, line_dash="dash", line_color=RED_COLOR, opacity=0.8, annotation_text=f"+{z_thresh}σ Threshold", annotation_position="top right")
    fig.add_hline(y=-z_thresh, line_dash="dash", line_color=GREEN_COLOR, opacity=0.8, annotation_text=f"-{z_thresh}σ Threshold", annotation_position="bottom right")
    fig.add_hline(y=0.0, line_color=GRID_COLOR, line_width=1)

    # Z-Score curve
    fig.add_trace(
        go.Scatter(
            x=df["trading_date"],
            y=df["z_score"],
            mode="lines",
            name="Point-in-Time Z-Score",
            line={"color": BLUE_COLOR, "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Z-Score: %{y:+.2f}σ<extra></extra>",
        )
    )

    # Robust MAD comparison trace
    if "mad_score" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["trading_date"],
                y=df["mad_score"],
                mode="lines",
                name="Robust MAD Z-Score",
                line={"color": "rgba(244, 208, 111, 0.6)", "width": 1.5, "dash": "dot"},
                hovertemplate="%{x|%Y-%m-%d}<br>MAD Score: %{y:+.2f}<extra></extra>",
            )
        )

    # Mark filtered anomalies
    filtered = df[df["signal_status"].isin(["FILTERED_LIQUIDITY", "FILTERED_NEAR_EXPIRY"])]
    if not filtered.empty:
        fig.add_trace(
            go.Scatter(
                x=filtered["trading_date"],
                y=filtered["z_score"],
                mode="markers",
                name="Filtered Outlier (Non-Eligible)",
                marker={"size": 7, "color": "#F59E0B", "symbol": "x"},
                hovertemplate="%{x|%Y-%m-%d}<br>Filtered Z: %{y:+.2f}σ<extra></extra>",
            )
        )

    apply_theme(fig, title="Point-in-Time Anomaly Score History (Z-Score vs Robust MAD)")
    fig.update_yaxes(title_text="Standard Deviations (σ)", title_font={"color": MUTED_TEXT})
    return fig


def plot_spread_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot distribution of relative log spread with theoretical normal curve."""
    clean_spread = df["spread"].dropna()
    spread_pct = (np.exp(clean_spread) - 1.0) * 100.0

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=spread_pct,
            nbinsx=25,
            name="Historical Spread Observations",
            marker={"color": "rgba(212, 175, 55, 0.7)", "line": {"color": GOLD_ACCENT, "width": 1}},
            opacity=0.75,
        )
    )

    # Add vertical line for latest spread
    if not spread_pct.empty:
        latest_val = spread_pct.iloc[-1]
        fig.add_vline(
            x=latest_val,
            line_dash="dot",
            line_color=TEXT_COLOR,
            line_width=2,
            annotation_text=f"Current: {latest_val:+.2f}%",
            annotation_position="top left",
        )

    apply_theme(fig, title="Historical Relative Spread Distribution (%)", height=320)
    fig.update_xaxes(title_text="Relative Spread (%)", title_font={"color": MUTED_TEXT})
    fig.update_yaxes(title_text="Observation Frequency", title_font={"color": MUTED_TEXT})
    return fig


def plot_liquidity_lifecycle(df: pd.DataFrame, sym_a: str, sym_b: str) -> go.Figure:
    """Dual-axis plot of traded contract volume, open interest, and days-to-expiry."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Contract Traded Volume", "Open Interest & Days to Expiry"),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    # Row 1: Volume
    fig.add_trace(
        go.Bar(
            x=df["trading_date"],
            y=df["volume_A"],
            name=f"{sym_a} Volume",
            marker_color=GOLD_ACCENT,
            opacity=0.8,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df["trading_date"],
            y=df["volume_B"],
            name=f"{sym_b} Volume",
            marker_color=BLUE_COLOR,
            opacity=0.8,
        ),
        row=1,
        col=1,
    )

    # Row 2: Open Interest & DTE
    fig.add_trace(
        go.Scatter(
            x=df["trading_date"],
            y=df["oi_A"],
            name=f"{sym_a} OI",
            line={"color": GOLD_LIGHT, "width": 1.5},
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["trading_date"],
            y=df["oi_B"],
            name=f"{sym_b} OI",
            line={"color": "#38BDF8", "width": 1.5, "dash": "dot"},
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["trading_date"],
            y=df["days_to_expiry"],
            name="Days to Expiry",
            line={"color": "#94A3B8", "width": 1, "dash": "dash"},
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    apply_theme(fig, title="Contract Liquidity & Lifecycle Dynamics", height=450)
    fig.update_yaxes(title_text="Contracts Traded", row=1, col=1)
    fig.update_yaxes(title_text="Open Interest", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Days to Expiry", row=2, col=1, secondary_y=True)
    return fig


def plot_backtest_equity(equity_df: pd.DataFrame, initial_capital: float) -> go.Figure:
    """Plot cumulative strategy equity curve and underwater drawdown."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.08,
        subplot_titles=("Cumulative Portfolio Capital (₹)", "Drawdown (%)"),
    )

    # Capital curve
    fig.add_trace(
        go.Scatter(
            x=equity_df["trading_date"],
            y=equity_df["portfolio_value"],
            mode="lines",
            name="Strategy Portfolio Value",
            line={"color": GOLD_ACCENT, "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Capital: ₹%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Baseline cash line
    fig.add_hline(
        y=initial_capital,
        line_dash="dot",
        line_color=MUTED_TEXT,
        annotation_text="Initial Capital",
        row=1,
        col=1,
    )

    # Drawdown underwater trace
    if "drawdown_pct" in equity_df.columns:
        fig.add_trace(
            go.Scatter(
                x=equity_df["trading_date"],
                y=equity_df["drawdown_pct"],
                mode="lines",
                name="Drawdown (%)",
                line={"color": RED_COLOR, "width": 1},
                fill="tozeroy",
                fillcolor="rgba(239, 68, 68, 0.15)",
                hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

    apply_theme(fig, title="Historical Strategy Equity Curve & Drawdown Profile", height=450)
    fig.update_yaxes(title_text="Portfolio Value (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    return fig
