"""EnergyIQ — Enterprise Smart Energy Consumption Analytics Dashboard.

A SaaS-grade energy intelligence dashboard matching modern executive monitoring consoles:
- Automatic first-launch telemetry loading from data/sample_energy.csv so all KPIs and charts are visible immediately.
- Premium dark navy canvas (#070D18) with glassmorphic cards, vivid gradient accents, and sharp contrast.
- Fixed-style left sidebar navigation with active teal menu highlight and full telemetry controls.
- Compact executive header with Dashboard Overview title, search bar, and live telemetry status pill.
- Four colorful gradient KPI cards in one horizontal row:
    1. Total Energy Consumption (Blue gradient)
    2. Average Energy Consumption (Teal/Green gradient)
    3. Detected Anomalies (Purple gradient)
    4. Peak Consumption (Orange/Red gradient)
- Two-column primary grid:
    - Prominent smooth-spline energy consumption line chart with glowing teal line, visible markers, dashed diurnal baseline, and highlighted anomalies.
    - Zone-wise consumption distribution bar chart with alternating blue and purple bars.
- Secondary grid:
    - Hourly diurnal demand profile with peak-hour orange highlight.
    - Daily consumption trend chart.
- Detailed operational anomaly evidence table with native CSV download.
- Quantitative mathematical facts card & clean integration hook for downstream recommendation engines.
"""

from __future__ import annotations

import html
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import calculate_metrics, detect_anomalies, load_and_validate_data
from recommendations import generate_recommendations


# =============================================================================
# 1. Configuration & Constants
# =============================================================================
SAMPLE_CSV_PATH = Path(__file__).parent / "data" / "sample_energy.csv"

st.set_page_config(
    page_title="EnergyIQ — Smart Energy Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 2. Executive Dark Navy Design System & Contrast Stylesheet
# =============================================================================
DARK_NAVY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Base Reset & Palette Variables */
:root {
    --bg-base: #070D18;
    --bg-card: #0E172E;
    --bg-card-subtle: #132247;
    --border-card: rgba(255, 255, 255, 0.09);
    --border-card-hover: rgba(255, 255, 255, 0.18);
    --text-primary: #F8FAFC;
    --text-muted: #94A3B8;
    --text-dim: #64748B;
    --accent-teal: #14D6A5;
    --accent-teal-glow: rgba(20, 214, 165, 0.22);
    --accent-blue: #3B82F6;
    --accent-purple: #8B5CF6;
    --accent-orange: #F59E0B;
    --accent-red: #EF4444;
}

/* Force dark canvas background and eliminate white Streamlit header bar */
header[data-testid="stHeader"] {
    background-color: #070D18 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
}
header[data-testid="stHeader"] * {
    color: #E2E8F0 !important;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, Segoe UI, system-ui, sans-serif !important;
    background-color: #070D18 !important;
    color: #F8FAFC !important;
}

.stApp {
    background-image: radial-gradient(circle at 12% 10%, rgba(20, 214, 165, 0.08) 0%, transparent 45%),
                      radial-gradient(circle at 85% 20%, rgba(59, 130, 246, 0.06) 0%, transparent 40%) !important;
}

/* Critical Contrast Fixes: Prevent dark-on-dark text across all Streamlit components */
.stMarkdown, [data-testid="stMarkdownContainer"] {
    color: #F8FAFC !important;
}
.stMarkdown p, [data-testid="stMarkdownContainer"] p {
    color: #E2E8F0 !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #FFFFFF !important;
}
.stMarkdown span, [data-testid="stMarkdownContainer"] span {
    color: inherit;
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: #0A1124 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}
[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] strong {
    color: #FFFFFF !important;
}

/* Sidebar Brand Header */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 6px 0 16px 0;
}
.sidebar-logo-icon {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #14D6A5 0%, #3B82F6 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    color: #FFFFFF !important;
    font-weight: 800;
    box-shadow: 0 4px 14px rgba(20, 214, 165, 0.35);
}
.sidebar-brand-text {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: #FFFFFF !important;
    line-height: 1.15;
}
.sidebar-brand-tagline {
    font-size: 0.68rem;
    color: var(--accent-teal) !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Sidebar Navigation Items */
.sidebar-nav-container {
    margin-bottom: 20px;
}
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 0.83rem;
    font-weight: 500;
    color: #94A3B8 !important;
    margin-bottom: 4px;
    transition: all 0.15s ease;
    cursor: default;
}
.nav-item.active {
    background: rgba(20, 214, 165, 0.14) !important;
    border-left: 3.5px solid var(--accent-teal) !important;
    color: var(--accent-teal) !important;
    font-weight: 700 !important;
}
.nav-icon {
    font-size: 1rem;
    opacity: 0.9;
}

/* Fix File Uploader contrast & background */
[data-testid="stFileUploader"] section {
    background-color: #0E172E !important;
    border: 1px dashed rgba(20, 214, 165, 0.35) !important;
    border-radius: 12px !important;
    padding: 14px !important;
}
[data-testid="stFileUploader"] section * {
    color: #CBD5E1 !important;
}
[data-testid="stFileUploader"] button {
    background-color: #132247 !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
}

/* Fix Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0e7490 0%, #14D6A5 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 8px 16px !important;
    box-shadow: 0 4px 14px rgba(20, 214, 165, 0.25) !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(20, 214, 165, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* Fix Streamlit form inputs */
.stTextInput input, .stDateInput input, .stNumberInput input {
    background-color: #0E172E !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
}
[data-baseweb="select"] > div {
    background-color: #0E172E !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
}

/* Sidebar Active Telemetry Summary Card */
.sidebar-summary-box {
    background: #0E172E;
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 12px;
    padding: 14px 16px;
    margin-top: 18px;
    font-size: 0.8rem;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
}
.sidebar-summary-title {
    font-weight: 700;
    color: #FFFFFF !important;
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.sidebar-summary-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
    color: #94A3B8 !important;
}
.sidebar-summary-row strong {
    color: #FFFFFF !important;
}

/* Top Dashboard Header */
.dash-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 22px;
}
.dash-title-left {
    display: flex;
    flex-direction: column;
}
.dash-page-title {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    margin: 0 !important;
    letter-spacing: -0.025em !important;
    line-height: 1.2 !important;
}
.dash-page-subtitle {
    color: #94A3B8 !important;
    font-size: 0.88rem !important;
    margin-top: 4px !important;
    margin-bottom: 0 !important;
}
.dash-status-pill {
    background: #0E172E;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    padding: 8px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}
.pulse-indicator {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background-color: var(--accent-teal);
    box-shadow: 0 0 10px var(--accent-teal);
    flex-shrink: 0;
}

/* Four Horizontal Gradient KPI Cards Grid */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
    margin-bottom: 24px;
    width: 100%;
    box-sizing: border-box;
}
@media (max-width: 1100px) {
    .kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 600px) {
    .kpi-grid {
        grid-template-columns: 1fr;
    }
}

.gradient-kpi-card {
    border-radius: 16px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    border: 1px solid rgba(255, 255, 255, 0.09);
    min-width: 0;
    min-height: 146px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-sizing: border-box;
}
div[data-testid="column"] > div[data-testid="stMarkdownContainer"] {
    height: 100%;
}
.gradient-kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.45);
}

/* KPI 1: Blue Gradient */
.kpi-blue {
    background: linear-gradient(135deg, rgba(29, 78, 216, 0.45) 0%, rgba(14, 23, 46, 0.95) 100%);
    border-top: 3.5px solid #3B82F6;
    border-color: rgba(59, 130, 246, 0.35);
}
.kpi-blue .card-icon-wrap {
    background: rgba(59, 130, 246, 0.25);
    color: #93C5FD;
    border: 1px solid rgba(59, 130, 246, 0.4);
}

/* KPI 2: Teal/Green Gradient */
.kpi-teal {
    background: linear-gradient(135deg, rgba(13, 148, 136, 0.45) 0%, rgba(14, 23, 46, 0.95) 100%);
    border-top: 3.5px solid #14D6A5;
    border-color: rgba(20, 214, 165, 0.35);
}
.kpi-teal .card-icon-wrap {
    background: rgba(20, 214, 165, 0.25);
    color: #5EEAD4;
    border: 1px solid rgba(20, 214, 165, 0.4);
}

/* KPI 3: Purple Gradient */
.kpi-purple {
    background: linear-gradient(135deg, rgba(126, 34, 206, 0.45) 0%, rgba(14, 23, 46, 0.95) 100%);
    border-top: 3.5px solid #A855F7;
    border-color: rgba(168, 85, 247, 0.35);
}
.kpi-purple .card-icon-wrap {
    background: rgba(168, 85, 247, 0.25);
    color: #D8B4FE;
    border: 1px solid rgba(168, 85, 247, 0.4);
}

/* KPI 4: Orange/Red Gradient */
.kpi-orange {
    background: linear-gradient(135deg, rgba(194, 65, 12, 0.45) 0%, rgba(14, 23, 46, 0.95) 100%);
    border-top: 3.5px solid #F97316;
    border-color: rgba(249, 115, 22, 0.35);
}
.kpi-orange .card-icon-wrap {
    background: rgba(249, 115, 22, 0.25);
    color: #FDBA74;
    border: 1px solid rgba(249, 115, 22, 0.4);
}

.kpi-top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.card-label {
    font-size: 0.74rem;
    font-weight: 700;
    color: #CBD5E1 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.card-icon-wrap {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    backdrop-filter: blur(8px);
    flex-shrink: 0;
}
.kpi-main-val {
    font-size: 2rem;
    font-weight: 800;
    color: #FFFFFF !important;
    line-height: 1.1;
    letter-spacing: -0.025em;
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-bottom: 10px;
}
.kpi-val-unit {
    font-size: 0.88rem;
    font-weight: 600;
    color: #94A3B8 !important;
}
.kpi-sub-meta {
    font-size: 0.73rem;
    color: #E2E8F0 !important;
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    line-height: 1.3;
}
.badge-indicator {
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    flex-shrink: 0;
}
.badge-blue { background: rgba(59, 130, 246, 0.3); color: #93C5FD !important; }
.badge-teal { background: rgba(20, 214, 165, 0.3); color: #5EEAD4 !important; }
.badge-purple { background: rgba(168, 85, 247, 0.3); color: #D8B4FE !important; }
.badge-orange { background: rgba(249, 115, 22, 0.3); color: #FDBA74 !important; }

/* Glassmorphic Chart & Detail Panels */
.glass-panel {
    background: #0E172E;
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}
.glass-panel-title {
    font-size: 1.08rem !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.glass-panel-subtitle {
    font-size: 0.8rem !important;
    color: #94A3B8 !important;
    margin-bottom: 14px;
}

/* Health Quality Banner */
.quality-strip {
    border-radius: 12px;
    padding: 10px 18px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.82rem;
}
.quality-clean {
    background: rgba(20, 214, 165, 0.12);
    border: 1px solid rgba(20, 214, 165, 0.35);
    color: var(--accent-teal) !important;
}
.quality-warn {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.35);
    color: var(--accent-orange) !important;
}

/* Integration Callout Box */
.integration-callout {
    background: #0E172E;
    border: 1px dashed rgba(20, 214, 165, 0.45);
    border-radius: 14px;
    padding: 18px 22px;
    margin-top: 24px;
    margin-bottom: 24px;
}
.integration-head {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--accent-teal) !important;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.integration-desc {
    font-size: 0.82rem;
    color: #94A3B8 !important;
    line-height: 1.5;
    margin: 0;
}
</style>
"""
st.markdown(DARK_NAVY_CSS, unsafe_allow_html=True)


# =============================================================================
# 3. Plotly Chart Theming Helper (Dark Navy Palette)
# =============================================================================
def style_plotly_chart(fig: go.Figure) -> go.Figure:
    """Format Plotly charts to match dark navy energy dashboard palette."""
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(
            family="Inter, -apple-system, Segoe UI, system-ui, sans-serif",
            color="#E2E8F0",
            size=11,
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.07)",
            zeroline=False,
            tickfont=dict(color="#94A3B8", size=10),
            title_font=dict(color="#E2E8F0", size=11),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.07)",
            zeroline=False,
            tickfont=dict(color="#94A3B8", size=10),
            title_font=dict(color="#E2E8F0", size=11),
        ),
        hoverlabel=dict(
            bgcolor="#0E172E",
            bordercolor="#14D6A5",
            font=dict(
                family="Inter, -apple-system, Segoe UI, system-ui, sans-serif",
                color="#FFFFFF",
                size=11,
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#94A3B8", size=10),
        ),
        margin=dict(l=16, r=16, t=28, b=16),
    )
    return fig


# =============================================================================
# 4. Data Loading & Caching Helpers
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data_from_bytes(content: bytes) -> pd.DataFrame:
    """Read CSV byte contents, validate, and clean via analytics.py."""
    return load_and_validate_data(io.BytesIO(content))


@st.cache_data(show_spinner=False)
def load_data_from_path(path_str: str) -> pd.DataFrame:
    """Read CSV file from local path, validate, and clean via analytics.py."""
    return load_and_validate_data(path_str)


# =============================================================================
# 5. Visual Component & KPI Builders
# =============================================================================
def render_kpi_cards(metrics: dict[str, Any], anomalies_df: pd.DataFrame) -> None:
    """Render four large horizontal gradient KPI cards in a Streamlit 4-column row."""
    summary = metrics.get("summary", {})
    total_kwh = summary.get("total_energy_kwh", 0.0)
    avg_kwh = summary.get("average_interval_kwh")
    max_kwh = summary.get("maximum_interval_kwh")
    peak_time = summary.get("peak_timestamp")
    peak_zone = summary.get("peak_zone")
    total_readings = summary.get("reading_count", 0)

    anomaly_count = int(anomalies_df["anomaly_flag"].sum()) if not anomalies_df.empty else 0
    anomaly_pct = (anomaly_count / total_readings * 100) if total_readings > 0 else 0.0

    peak_time_str = (
        pd.to_datetime(peak_time).strftime("%b %d, %H:%M")
        if peak_time is not None and pd.notna(peak_time)
        else "N/A"
    )
    peak_zone_label = f"({html.escape(str(peak_zone))})" if peak_zone and peak_zone != "All" else ""
    avg_str = f"{avg_kwh:.2f}" if avg_kwh is not None else "N/A"
    max_str = f"{max_kwh:.2f}" if max_kwh is not None else "N/A"

    anom_badge_text = f"{anomaly_pct:.1f}% OF READINGS" if anomaly_count > 0 else "NORMAL"
    anom_meta_text = "Outliers exceeding 3× MAD" if anomaly_count > 0 else "No anomalies detected"
    anom_num_color = "#E9D5FF" if anomaly_count > 0 else "#FFFFFF"

    # KPI 1: Total Energy Consumption (Blue Gradient)
    card1_html = (
        '<div class="gradient-kpi-card kpi-blue">'
        '<div class="kpi-top-row">'
        '<span class="card-label">Total Energy Consumption</span>'
        '<div class="card-icon-wrap">⚡</div>'
        '</div>'
        '<div class="kpi-main-val">'
        f'<span>{total_kwh:,.1f}</span>'
        '<span class="kpi-val-unit">kWh</span>'
        '</div>'
        '<div class="kpi-sub-meta">'
        '<span class="badge-indicator badge-blue">AGGREGATE</span>'
        f'<span>Across {total_readings:,} recorded intervals</span>'
        '</div>'
        '</div>'
    )

    # KPI 2: Average Energy Consumption (Teal/Green Gradient)
    card2_html = (
        '<div class="gradient-kpi-card kpi-teal">'
        '<div class="kpi-top-row">'
        '<span class="card-label">Average Energy Consumption</span>'
        '<div class="card-icon-wrap">📊</div>'
        '</div>'
        '<div class="kpi-main-val">'
        f'<span>{avg_str}</span>'
        '<span class="kpi-val-unit">kWh/int</span>'
        '</div>'
        '<div class="kpi-sub-meta">'
        '<span class="badge-indicator badge-teal">INTERVAL MEAN</span>'
        '<span>Average power draw per interval</span>'
        '</div>'
        '</div>'
    )

    # KPI 3: Detected Anomalies (Purple Gradient)
    card3_html = (
        '<div class="gradient-kpi-card kpi-purple">'
        '<div class="kpi-top-row">'
        '<span class="card-label">Detected Anomalies</span>'
        '<div class="card-icon-wrap">🚨</div>'
        '</div>'
        '<div class="kpi-main-val">'
        f'<span style="color: {anom_num_color};">{anomaly_count}</span>'
        '<span class="kpi-val-unit">events</span>'
        '</div>'
        '<div class="kpi-sub-meta">'
        f'<span class="badge-indicator badge-purple">{anom_badge_text}</span>'
        f'<span>{anom_meta_text}</span>'
        '</div>'
        '</div>'
    )

    # KPI 4: Peak Consumption (Orange/Red Gradient)
    card4_html = (
        '<div class="gradient-kpi-card kpi-orange">'
        '<div class="kpi-top-row">'
        '<span class="card-label">Peak Consumption</span>'
        '<div class="card-icon-wrap">🔥</div>'
        '</div>'
        '<div class="kpi-main-val">'
        f'<span>{max_str}</span>'
        '<span class="kpi-val-unit">kWh</span>'
        '</div>'
        '<div class="kpi-sub-meta">'
        '<span class="badge-indicator badge-orange">MAX RECORD</span>'
        f'<span>{peak_time_str} {peak_zone_label}</span>'
        '</div>'
        '</div>'
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(card1_html, unsafe_allow_html=True)
    with col2:
        st.markdown(card2_html, unsafe_allow_html=True)
    with col3:
        st.markdown(card3_html, unsafe_allow_html=True)
    with col4:
        st.markdown(card4_html, unsafe_allow_html=True)


# =============================================================================
# 6. Chart Builders
# =============================================================================
def build_timeseries_chart(anomalies_df: pd.DataFrame) -> go.Figure:
    """Build energy consumption trend line chart with linear interpolation and zone safety."""
    fig = go.Figure()
    if anomalies_df.empty:
        return style_plotly_chart(fig)

    zones = sorted(anomalies_df["zone"].dropna().unique().tolist())
    is_multi_zone = len(zones) > 1

    zone_palette = {
        "Zone 1": "#14D6A5",
        "Zone 2": "#3B82F6",
        "Zone 3": "#8B5CF6",
        "Block A": "#14D6A5",
        "All": "#14D6A5",
    }
    fallback_colors = ["#14D6A5", "#3B82F6", "#8B5CF6", "#F59E0B", "#EC4899", "#06B6D4"]

    # When multiple zones exist, plot each zone series independently to avoid connecting different zones
    for i, z in enumerate(zones):
        z_df = anomalies_df[anomalies_df["zone"] == z].sort_values("timestamp")
        color = zone_palette.get(z, fallback_colors[i % len(fallback_colors)])
        # Use lines without markers for large datasets to guarantee high browser performance
        plot_mode = "lines" if len(z_df) > 500 else "lines+markers"

        fig.add_trace(
            go.Scatter(
                x=z_df["timestamp"],
                y=z_df["energy_kwh"],
                mode=plot_mode,
                name=f"{z}" if is_multi_zone else "Actual Consumption",
                line=dict(color=color, width=2.2, shape="linear"),
                marker=dict(size=4, color=color) if plot_mode == "lines+markers" else None,
                hovertemplate=f"<b>{html.escape(z)}</b><br>%{{x|%b %d, %Y %H:%M}}<br>Demand: <b>%{{y:.2f}} kWh</b><extra></extra>",
            )
        )

    # If single-zone, render the expected diurnal baseline trace
    if not is_multi_zone:
        baseline_subset = anomalies_df[anomalies_df["expected_baseline_kwh"].notna()].sort_values("timestamp")
        if not baseline_subset.empty:
            fig.add_trace(
                go.Scatter(
                    x=baseline_subset["timestamp"],
                    y=baseline_subset["expected_baseline_kwh"],
                    mode="lines",
                    name="Diurnal Baseline",
                    line=dict(color="#A855F7", width=1.8, dash="dash", shape="linear"),
                    hovertemplate="<b>Baseline</b><br>%{x|%b %d, %Y %H:%M}<br>Expected: <b>%{y:.2f} kWh</b><extra></extra>",
                )
            )

    # Highlight anomalies with visible markers (cap at top 300 to maintain browser fluidity on large ranges)
    anomalies_subset = anomalies_df[anomalies_df["anomaly_flag"] == True]
    if not anomalies_subset.empty:
        if len(anomalies_subset) > 300:
            anomalies_subset = anomalies_subset.nlargest(300, "deviation_kwh")

        fig.add_trace(
            go.Scatter(
                x=anomalies_subset["timestamp"],
                y=anomalies_subset["energy_kwh"],
                mode="markers",
                name="Critical Anomalies",
                marker=dict(
                    color="#EF4444",
                    size=8,
                    symbol="circle-open",
                    line=dict(color="#EF4444", width=2),
                ),
                hovertemplate=(
                    "<b>⚠️ ANOMALY DETECTED</b><br>"
                    "Timestamp: <b>%{x|%b %d, %Y %H:%M}</b><br>"
                    "Measured: <b>%{y:.2f} kWh</b><br>"
                    "Zone: <b>%{text}</b><extra></extra>"
                ),
                text=anomalies_subset["zone"],
            )
        )

    fig.update_layout(
        height=430,
        hovermode="x unified" if not is_multi_zone else "closest",
        xaxis=dict(
            title="Measurement Timestamp",
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1D", step="day", stepmode="backward"),
                    dict(count=3, label="3D", step="day", stepmode="backward"),
                    dict(count=7, label="7D", step="day", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]),
                bgcolor="#132247",
                activecolor="#14D6A5",
                font=dict(color="#FFFFFF", size=10),
            ),
            type="date",
        ),
        yaxis=dict(title="Energy (kWh)"),
    )
    return style_plotly_chart(fig)


def build_zone_chart(zone_df: pd.DataFrame) -> go.Figure:
    """Build zone-wise consumption bar chart with alternating blue and purple bars."""
    fig = go.Figure()
    if not zone_df.empty:
        sorted_zone_df = zone_df.sort_values("total_energy_kwh", ascending=True).reset_index(drop=True)
        # Alternating blue (#3B82F6) and purple (#8B5CF6) bar colors
        bar_colors = ["#3B82F6" if i % 2 == 0 else "#8B5CF6" for i in range(len(sorted_zone_df))]

        fig.add_trace(
            go.Bar(
                y=sorted_zone_df["zone"],
                x=sorted_zone_df["total_energy_kwh"],
                orientation="h",
                marker=dict(
                    color=bar_colors,
                    line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
                ),
                name="Total Energy",
                hovertemplate="Zone: <b>%{y}</b><br>Total Energy: <b>%{x:,.1f} kWh</b><extra></extra>",
            )
        )
        fig.update_layout(
            height=430,
            xaxis=dict(title="Total Consumption (kWh)"),
            yaxis=dict(title="Metered Zone"),
        )
    return style_plotly_chart(fig)


def build_hourly_chart(hourly_df: pd.DataFrame) -> go.Figure:
    """Build hourly diurnal load profile bar chart with peak hour highlighted in orange."""
    fig = go.Figure()
    if not hourly_df.empty:
        max_avg = hourly_df["average_energy_kwh"].max()
        bar_colors = [
            "#F59E0B" if val == max_avg else "#14D6A5"
            for val in hourly_df["average_energy_kwh"]
        ]
        fig.add_trace(
            go.Bar(
                x=hourly_df["hour"],
                y=hourly_df["average_energy_kwh"],
                marker=dict(
                    color=bar_colors,
                    line=dict(color="rgba(255, 255, 255, 0.15)", width=1),
                ),
                name="Hourly Avg kWh",
                hovertemplate="Hour: <b>%{x:02d}:00</b><br>Average Demand: <b>%{y:.2f} kWh</b><extra></extra>",
            )
        )
        fig.update_layout(
            height=310,
            xaxis=dict(
                title="Hour of Day (0–23)",
                tickmode="linear",
                tick0=0,
                dtick=2,
            ),
            yaxis=dict(title="Average kWh"),
        )
    return style_plotly_chart(fig)


def build_daily_chart(daily_df: pd.DataFrame) -> go.Figure:
    """Build daily aggregate consumption bar chart."""
    fig = go.Figure()
    if not daily_df.empty:
        fig.add_trace(
            go.Bar(
                x=daily_df["date"].astype(str),
                y=daily_df["total_energy_kwh"],
                marker=dict(
                    color="#3B82F6",
                    line=dict(color="rgba(59, 130, 246, 0.35)", width=1),
                ),
                name="Daily Total kWh",
                hovertemplate="Date: <b>%{x}</b><br>Daily Total: <b>%{y:.2f} kWh</b><extra></extra>",
            )
        )
        fig.update_layout(
            height=310,
            xaxis=dict(title="Date"),
            yaxis=dict(title="Total Daily Energy (kWh)"),
        )
    return style_plotly_chart(fig)


# =============================================================================
# 7. Main Application Logic
# =============================================================================
def main() -> None:
    """Main execution controller for the EnergyIQ Streamlit application."""

    # --- Sidebar: Brand Header ---
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo-icon">⚡</div>
            <div>
                <div class="sidebar-brand-text">EnergyIQ</div>
                <div class="sidebar-brand-tagline">Smart Energy Optimizer</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar: Executive Navigation Mockup ---
    st.sidebar.markdown(
        """
        <div class="sidebar-nav-container">
            <div class="nav-item active"><span class="nav-icon">📊</span> Dashboard Overview</div>
            <div class="nav-item"><span class="nav-icon">📈</span> Consumption Analytics</div>
            <div class="nav-item"><span class="nav-icon">📡</span> Live Monitoring</div>
            <div class="nav-item"><span class="nav-icon">💡</span> AI Recommendations</div>
            <div class="nav-item"><span class="nav-icon">📑</span> Reports & Exports</div>
            <div class="nav-item"><span class="nav-icon">⚙️</span> Settings & Sensitivity</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    # --- Sidebar: Telemetry Ingestion ---
    st.sidebar.markdown("##### 📥 Telemetry Ingestion")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Smart Meter CSV",
        type=["csv"],
        help="Expected columns: 'timestamp' and 'energy_kwh'. Optional: 'zone'.",
    )

    load_sample_clicked = st.sidebar.button(
        "⚡ Load Sample Telemetry",
        use_container_width=True,
        help="Loads sample_energy.csv with 72 hours of hourly meter data for Block A.",
    )

    # Automatic initial load behavior: default to sample data on first launch
    if "data_source" not in st.session_state:
        st.session_state["data_source"] = "sample"

    if uploaded_file is not None:
        st.session_state["data_source"] = "uploaded"
    elif load_sample_clicked:
        st.session_state["data_source"] = "sample"

    # --- Sidebar: Anomaly Sensitivity Controls ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### ⚙️ Outlier Sensitivity Tuner")
    st.sidebar.caption("Configures diurnal baseline windows and robust MAD spread limits.")

    param_window = st.sidebar.slider(
        "Diurnal Window (Days)",
        min_value=2,
        max_value=24,
        value=6,
        help="Preceding intervals at the same hour of day used for median baseline.",
    )
    param_min_history = st.sidebar.slider(
        "Min. History Required",
        min_value=1,
        max_value=10,
        value=2,
        help="Minimum required historical points before baseline calculation is activated.",
    )
    param_threshold_kwh = st.sidebar.number_input(
        "Absolute Deviation Floor (kWh)",
        min_value=0.1,
        max_value=50.0,
        value=2.0,
        step=0.5,
        help="Minimum absolute surge in kWh above baseline required to trigger an anomaly flag.",
    )

    # --- Telemetry Ingestion & Validation via analytics.py ---
    df_clean: pd.DataFrame | None = None
    source_label = "sample_energy.csv"

    if st.session_state.get("data_source") == "uploaded" and uploaded_file is not None:
        source_label = f"Uploaded: {uploaded_file.name}"
        try:
            file_bytes = uploaded_file.getvalue()
            df_clean = load_data_from_bytes(file_bytes)
        except Exception as exc:
            st.sidebar.error(f"Validation Error: {exc}")
            st.stop()
    else:
        # Automatic default to sample dataset
        source_label = "sample_energy.csv"
        if not SAMPLE_CSV_PATH.is_file():
            st.sidebar.error(f"Sample data file not found at {SAMPLE_CSV_PATH}")
            st.stop()
        try:
            df_clean = load_data_from_path(str(SAMPLE_CSV_PATH))
        except Exception as exc:
            st.sidebar.error(f"Error loading sample: {exc}")
            st.stop()

    if df_clean is None or df_clean.empty:
        st.error("No valid telemetry records loaded. Please verify the CSV format.")
        st.stop()

    # --- Telemetry Filters ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 🔍 Telemetry Filters")

    min_csv_date = df_clean["timestamp"].min().date()
    max_csv_date = df_clean["timestamp"].max().date()

    # Initial default view: Most recent 7 days for immediate responsiveness, full date boundaries selectable
    default_start_date = max(min_csv_date, max_csv_date - pd.Timedelta(days=7))

    date_filter = st.sidebar.date_input(
        "Date Range",
        value=(default_start_date, max_csv_date),
        min_value=min_csv_date,
        max_value=max_csv_date,
        help="Filter interval readings between selected dates. Initial view defaults to recent 7 days for fast responsiveness.",
    )

    filtered_df = df_clean.copy()
    if isinstance(date_filter, (list, tuple)):
        if len(date_filter) == 2:
            start_date, end_date = date_filter[0], date_filter[1]
            filtered_df = filtered_df[
                (filtered_df["timestamp"].dt.date >= start_date)
                & (filtered_df["timestamp"].dt.date <= end_date)
            ].copy()
        elif len(date_filter) == 1:
            st.sidebar.caption("ℹ️ Click an end date to complete the range filter.")

    # Zone Filter
    available_zones = sorted(df_clean["zone"].dropna().unique().tolist())
    if len(available_zones) > 1:
        selected_zones = st.sidebar.multiselect(
            "Filter Zones",
            options=available_zones,
            default=available_zones,
            help="Filter telemetry to specific facility zones.",
        )
        if selected_zones:
            filtered_df = filtered_df[filtered_df["zone"].isin(selected_zones)].copy()
        else:
            filtered_df = filtered_df.iloc[0:0].copy()
            st.sidebar.warning("⚠️ Select at least one zone to view analytics.")

    filtered_df.attrs["data_quality_warnings"] = df_clean.attrs.get("data_quality_warnings", [])
    filtered_df.attrs["interval_note"] = df_clean.attrs.get("interval_note", "")

    if filtered_df.empty:
        st.warning("⚠️ No records match the active date and zone filters. Please broaden your selection in the sidebar.")
        return

    # --- Sidebar Compact Data-Summary Card ---
    summary_rows_count = len(filtered_df)
    summary_start_str = filtered_df["timestamp"].min().strftime("%b %d, %Y")
    summary_end_str = filtered_df["timestamp"].max().strftime("%b %d, %Y")
    summary_zones_str = ", ".join(sorted(filtered_df["zone"].unique().tolist()))

    st.sidebar.markdown(
        f"""
        <div class="sidebar-summary-box">
            <div class="sidebar-summary-title"><span>📊</span> Active Telemetry</div>
            <div class="sidebar-summary-row"><span>Intervals:</span> <strong>{summary_rows_count:,}</strong></div>
            <div class="sidebar-summary-row"><span>Date Span:</span> <strong>{summary_start_str} – {summary_end_str}</strong></div>
            <div class="sidebar-summary-row"><span>Zones:</span> <strong title="{html.escape(summary_zones_str)}">{html.escape(summary_zones_str[:18])}{'...' if len(summary_zones_str) > 18 else ''}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Analytics Calculations ---
    try:
        metrics = calculate_metrics(filtered_df)
        anomalies_df = detect_anomalies(
            filtered_df,
            window=int(param_window),
            min_history=int(param_min_history),
            threshold_kwh=float(param_threshold_kwh),
        )
    except Exception as exc:
        st.error(f"Analytics Pipeline Error: {exc}")
        st.stop()

    summary = metrics.get("summary", {})
    total_kwh = summary.get("total_energy_kwh", 0.0)
    total_readings = summary.get("reading_count", 0)

    # --- Top Dashboard Header with Search Bar & Status Pill ---
    safe_source = html.escape(source_label)
    col_hdr_left, col_hdr_right = st.columns([1.6, 1])

    with col_hdr_left:
        st.markdown(
            """
            <div class="dash-title-left">
                <h1 class="dash-page-title">Dashboard Overview</h1>
                <p class="dash-page-subtitle">Continuous Energy Monitoring & Diurnal Outlier Intelligence</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_hdr_right:
        col_search, col_status = st.columns([1.2, 1])
        with col_search:
            st.text_input(
                "Search metrics",
                placeholder="🔍 Search telemetry...",
                label_visibility="collapsed",
            )
        with col_status:
            st.markdown(
                f"""
                <div class="dash-status-pill">
                    <div class="pulse-indicator"></div>
                    <div>
                        <div style="font-weight: 700; color: #14D6A5; font-size: 0.72rem; text-transform: uppercase;">● Telemetry Active</div>
                        <div style="font-size: 0.76rem; color: #E2E8F0;">{safe_source}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Data Quality Status Strip ---
    quality_warnings = filtered_df.attrs.get("data_quality_warnings", [])
    if quality_warnings:
        with st.expander(f"⚠️ Telemetry Quality Notices ({len(quality_warnings)} Items Flagged)", expanded=False):
            for w in quality_warnings:
                st.warning(w)
    else:
        st.markdown(
            """
            <div class="quality-strip quality-clean">
                <span style="font-weight: 600;">✓ Data looks healthy: Zero timestamp collisions, negative values, or missing intervals excluded.</span>
                <span style="font-size: 0.76rem; opacity: 0.85;">Native interval metering</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Four Large Horizontal Gradient KPI Cards ---
    render_kpi_cards(metrics, anomalies_df)

    # --- Primary Two-Column Grid: Energy Trend Line Chart (Left) + Zone Bar Chart (Right) ---
    col_grid_left, col_grid_right = st.columns([1.65, 1.0])

    with col_grid_left:
        st.markdown(
            """
            <div class="glass-panel">
                <div class="glass-panel-title"><span>📈</span> Energy Consumption Trend & Anomaly Surveillance</div>
                <div class="glass-panel-subtitle">Smooth-spline interval consumption with causal rolling diurnal baselines and highlighted statistical outliers.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_ts = build_timeseries_chart(anomalies_df)
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_grid_right:
        st.markdown(
            """
            <div class="glass-panel">
                <div class="glass-panel-title"><span>🏢</span> Zone-Wise Consumption Distribution</div>
                <div class="glass-panel-subtitle">Aggregate demand across facility sectors with alternating blue and purple bars.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        zone_df = metrics.get("zone_comparison", pd.DataFrame())
        if not zone_df.empty:
            fig_zone = build_zone_chart(zone_df)
            st.plotly_chart(fig_zone, use_container_width=True)
            if len(zone_df) == 1:
                sz_row = zone_df.iloc[0]
                st.caption(
                    f"Active Facility: **{html.escape(str(sz_row['zone']))}** "
                    f"({sz_row['total_energy_kwh']:,.1f} kWh total, {sz_row['reading_count']:,} readings, "
                    f"{sz_row['average_interval_kwh']:.2f} kWh/interval mean)."
                )
        else:
            st.caption("No zone distribution data available.")

    # --- Secondary Two-Column Grid: Hourly Diurnal Profile & Daily Aggregate Trend ---
    col_r2_left, col_r2_right = st.columns(2)
    hourly_df = metrics.get("hourly", pd.DataFrame())
    daily_df = metrics.get("daily", pd.DataFrame())

    with col_r2_left:
        st.markdown(
            """
            <div class="glass-panel">
                <div class="glass-panel-title"><span>⏱️</span> Hourly Diurnal Consumption Profile (0–23h)</div>
                <div class="glass-panel-subtitle">Average interval energy draw by hour of day with peak load hour highlighted in orange.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_hourly = build_hourly_chart(hourly_df)
        st.plotly_chart(fig_hourly, use_container_width=True)

    with col_r2_right:
        st.markdown(
            """
            <div class="glass-panel">
                <div class="glass-panel-title"><span>📅</span> Daily Energy Consumption Trends</div>
                <div class="glass-panel-subtitle">Day-over-day aggregate demand trajectories across the active monitoring timeframe.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        fig_daily = build_daily_chart(daily_df)
        st.plotly_chart(fig_daily, use_container_width=True)

    # --- Anomaly Evidence Table & CSV Download ---
    st.markdown(
        """
        <div class="glass-panel">
            <div class="glass-panel-title"><span>🚨</span> Operational Anomaly Audit Log & Evidence</div>
            <div class="glass-panel-subtitle">Mathematical proof for all flagged outliers. Baselines strictly reflect the median of prior readings for the identical zone and hour.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    flagged_rows = anomalies_df[anomalies_df["anomaly_flag"] == True].copy()
    if not flagged_rows.empty:
        evidence_df = flagged_rows[
            ["timestamp", "zone", "energy_kwh", "expected_baseline_kwh", "deviation_kwh"]
        ].copy()

        evidence_df["deviation_pct"] = (
            (evidence_df["deviation_kwh"] / evidence_df["expected_baseline_kwh"]) * 100
        )
        evidence_df["severity"] = evidence_df["deviation_pct"].map(
            lambda pct: "🔴 Critical Spike (>100%)" if pct >= 100 else "🟠 Moderate Surge"
        )
        evidence_df = evidence_df.sort_values("deviation_kwh", ascending=False).reset_index(drop=True)

        csv_export_data = evidence_df.copy()
        csv_export_data["timestamp"] = csv_export_data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        csv_bytes = csv_export_data.to_csv(index=False).encode("utf-8")

        total_flagged = len(evidence_df)
        col_tbl_left, col_tbl_right = st.columns([3.5, 1.5])
        with col_tbl_right:
            st.download_button(
                label=f"📥 Download Full Audit CSV ({total_flagged:,} events)",
                data=csv_bytes,
                file_name="energyiq_anomaly_audit.csv",
                mime="text/csv",
                use_container_width=True,
                help="Download the complete anomaly audit table with baselines and deviations.",
            )

        # Limit visible interactive table to top 100 rows to ensure snappy UI responsiveness
        evidence_display = evidence_df.head(100).copy()
        evidence_display["timestamp"] = evidence_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        evidence_display = evidence_display.rename(
            columns={
                "timestamp": "Timestamp",
                "zone": "Zone",
                "energy_kwh": "Measured (kWh)",
                "expected_baseline_kwh": "Baseline (kWh)",
                "deviation_kwh": "Surge (kWh)",
                "deviation_pct": "Deviation (%)",
                "severity": "Severity",
            }
        )

        st.dataframe(
            evidence_display.style.format(
                {
                    "Measured (kWh)": "{:,.2f}",
                    "Baseline (kWh)": "{:,.2f}",
                    "Surge (kWh)": "{:+,.2f}",
                    "Deviation (%)": "{:+,.1f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        if total_flagged > 100:
            st.caption(
                f"ℹ️ Displaying top 100 of {total_flagged:,} flagged outliers (ranked by highest surge). "
                f"Outlier Flag Criterion: Flagged when |Measured − Baseline| &gt; "
                f"max(3 × 1.4826 × MAD, {param_threshold_kwh:.1f} kWh). "
                "Download the complete audit CSV above for all interval records."
            )
        else:
            st.caption(
                f"ℹ️ Outlier Flag Criterion: Flagged when |Measured − Baseline| &gt; "
                f"max(3 × 1.4826 × MAD, {param_threshold_kwh:.1f} kWh). "
                "Historical baselines are causal (shift=1) to prevent lookahead bias or self-contamination."
            )
    else:
        st.markdown(
            """
            <div class="quality-strip quality-clean">
                <span style="font-weight: 700;">✓ Operational Normalcy Confirmed:</span>
                <span style="margin-left: 6px;">No interval consumption exceeded the rolling diurnal baseline by more than the configured sensitivity thresholds.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Row 4: Quantitative Demand Insights & Recommendation Handoff ---
    col_bot_left, col_bot_right = st.columns([1, 1.2])

    with col_bot_left:
        st.markdown(
            """
            <div class="glass-panel">
                <div class="glass-panel-title"><span>💡</span> Quantitative Demand Insights</div>
                <div class="glass-panel-subtitle">Mathematical facts derived strictly from validated interval data.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not hourly_df.empty:
            top_hr_row = hourly_df.loc[hourly_df["average_energy_kwh"].idxmax()]
            top_hr_val = int(top_hr_row["hour"])
            top_hr_avg = float(top_hr_row["average_energy_kwh"])
            fact_peak_hour = f"Hour {top_hr_val:02d}:00 with {top_hr_avg:.2f} kWh mean interval load"
        else:
            fact_peak_hour = "N/A"

        if not daily_df.empty:
            top_day_row = daily_df.loc[daily_df["total_energy_kwh"].idxmax()]
            fact_busiest_day = f"{top_day_row['date']} ({top_day_row['total_energy_kwh']:,.1f} kWh total demand)"
        else:
            fact_busiest_day = "N/A"

        zone_df = metrics.get("zone_comparison", pd.DataFrame())
        if not zone_df.empty:
            top_zn_row = zone_df.loc[zone_df["total_energy_kwh"].idxmax()]
            zn_pct = (top_zn_row["total_energy_kwh"] / total_kwh * 100) if total_kwh > 0 else 0.0
            fact_top_zone = f"{html.escape(str(top_zn_row['zone']))} ({top_zn_row['total_energy_kwh']:,.1f} kWh, {zn_pct:.1f}% share)"
        else:
            fact_top_zone = "N/A"

        baselined_count = int(anomalies_df["expected_baseline_kwh"].notna().sum())
        unbaselined_count = total_readings - baselined_count

        st.markdown(
            f"""
            <div style="background: #132247; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px 18px; font-size: 0.82rem; line-height: 1.6;">
                <div style="margin-bottom: 8px;">
                    <span style="color: #94A3B8; font-weight: 600;">• Peak Diurnal Demand:</span><br>
                    <strong style="color: #FFFFFF; margin-left: 12px;">{fact_peak_hour}</strong>
                </div>
                <div style="margin-bottom: 8px;">
                    <span style="color: #94A3B8; font-weight: 600;">• Highest Consumption Day:</span><br>
                    <strong style="color: #FFFFFF; margin-left: 12px;">{fact_busiest_day}</strong>
                </div>
                <div style="margin-bottom: 8px;">
                    <span style="color: #94A3B8; font-weight: 600;">• Highest Consuming Zone:</span><br>
                    <strong style="color: #FFFFFF; margin-left: 12px;">{fact_top_zone}</strong>
                </div>
                <div>
                    <span style="color: #94A3B8; font-weight: 600;">• Baselined Coverage:</span><br>
                    <strong style="color: #FFFFFF; margin-left: 12px;">{baselined_count:,} intervals active ({unbaselined_count} initial warm-up periods)</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_bot_right:
        st.markdown(
            """
            <div class="glass-panel">
                <div class="glass-panel-title"><span>💡</span> AI Energy Optimization Recommendations</div>
                <div class="glass-panel-subtitle">Rule-driven, data-backed energy conservation measures derived from telemetry patterns.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            raw_recs = generate_recommendations(metrics, anomalies_df)
        except Exception as exc:
            raw_recs = []
            st.caption(f"Recommendation engine note: {exc}")

        # Confirm returned dictionary keys before displaying
        valid_recs = [
            r for r in raw_recs
            if isinstance(r, dict) and "title" in r and "description" in r
        ]

        if valid_recs:
            for rec in valid_recs:
                priority = str(rec.get("priority", "Medium")).capitalize()
                category = str(rec.get("category", "Operational"))
                title = str(rec.get("title", "Recommendation"))
                desc = str(rec.get("description", ""))

                if priority == "High":
                    accent_color = "#EF4444"
                    badge_style = "background: rgba(239, 68, 68, 0.2); color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.4);"
                    icon = "🔴"
                elif priority == "Medium":
                    accent_color = "#F59E0B"
                    badge_style = "background: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.4);"
                    icon = "🟠"
                else:
                    accent_color = "#3B82F6"
                    badge_style = "background: rgba(59, 130, 246, 0.2); color: #93C5FD; border: 1px solid rgba(59, 130, 246, 0.4);"
                    icon = "🔵"

                st.markdown(
                    f"""
                    <div style="background: #0E172E; border: 1px solid rgba(255, 255, 255, 0.08); border-left: 4px solid {accent_color}; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 700; color: #FFFFFF; font-size: 0.88rem;">{icon} {html.escape(title)}</span>
                            <span style="padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; {badge_style}">{html.escape(priority)} • {html.escape(category)}</span>
                        </div>
                        <p style="color: #94A3B8; font-size: 0.8rem; margin: 0; line-height: 1.45;">{html.escape(desc)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div class="quality-strip quality-clean">
                    <span style="font-weight: 600;">✓ Optimization Normalcy: No critical efficiency anomalies or schedule conservation actions flagged for active telemetry window.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Footer ---
    st.markdown(
        """
        <div style="text-align: center; color: #64748B; font-size: 0.74rem; padding: 22px 0 10px 0; border-top: 1px solid rgba(255, 255, 255, 0.08); margin-top: 24px;">
            EnergyIQ v1.2.0 • Enterprise Energy Intelligence & Diurnal Demand Baselining Console • Hackathon Edition
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
