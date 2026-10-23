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

/* Sidebar Navigation Radio Component */
[data-testid="stSidebar"] [data-testid="stRadio"] {
    margin-top: -6px;
    margin-bottom: 16px;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
    margin-bottom: 2px !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    background: transparent !important;
    border-left: 3.5px solid transparent !important;
    width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    color: #E2E8F0 !important;
}
/* Hide the native radio circle element */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}
/* Active selected option */
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(20, 214, 165, 0.14) !important;
    border-left: 3.5px solid var(--accent-teal) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] span,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) span,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] div,
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) div {
    color: var(--accent-teal) !important;
    font-weight: 700 !important;
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


def render_generic_kpi_row(cards: list[dict[str, Any]]) -> None:
    """Render a row of styled gradient KPI cards."""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        color_class = card.get("color_class", "kpi-blue")
        badge_class = card.get("badge_class", "badge-blue")
        num_color = card.get("num_color", "#FFFFFF")
        val = card.get("value", "N/A")
        unit = card.get("unit", "")
        unit_html = f'<span class="kpi-val-unit">{html.escape(unit)}</span>' if unit else ""
        html_code = (
            f'<div class="gradient-kpi-card {color_class}">'
            '<div class="kpi-top-row">'
            f'<span class="card-label">{html.escape(str(card.get("title", "")))}</span>'
            f'<div class="card-icon-wrap">{card.get("icon", "⚡")}</div>'
            '</div>'
            '<div class="kpi-main-val">'
            f'<span style="color: {num_color};">{val}</span>'
            f'{unit_html}'
            '</div>'
            '<div class="kpi-sub-meta">'
            f'<span class="badge-indicator {badge_class}">{html.escape(str(card.get("badge", "")))}</span>'
            f'<span>{html.escape(str(card.get("meta", "")))}</span>'
            '</div>'
            '</div>'
        )
        with col:
            st.markdown(html_code, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str, source_label: str) -> None:
    """Render compact executive header with title, subtitle, search input, and status indicator."""
    safe_source = html.escape(source_label)
    search_key = f"search_{title.lower().replace(' ', '_').replace('&', 'and')}"
    col_hdr_left, col_hdr_right = st.columns([1.6, 1])

    with col_hdr_left:
        st.markdown(
            f"""
            <div class="dash-title-left">
                <h1 class="dash-page-title">{html.escape(title)}</h1>
                <p class="dash-page-subtitle">{html.escape(subtitle)}</p>
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
                key=search_key,
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
# 7. Page Views & Report Generators
# =============================================================================
def generate_executive_report(
    metrics: dict[str, Any],
    anomalies_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    raw_recs: list[dict[str, Any]],
) -> str:
    """Generate structured markdown executive report for export."""
    summary = metrics.get("summary", {})
    total_kwh = summary.get("total_energy_kwh", 0.0)
    avg_kwh = summary.get("average_interval_kwh", 0.0)
    max_kwh = summary.get("maximum_interval_kwh", 0.0)
    peak_ts = summary.get("peak_timestamp", "N/A")
    peak_zone = summary.get("peak_zone", "N/A")
    total_readings = summary.get("reading_count", 0)
    start_date = filtered_df["timestamp"].min().strftime("%Y-%m-%d %H:%M")
    end_date = filtered_df["timestamp"].max().strftime("%Y-%m-%d %H:%M")
    flagged_count = int(anomalies_df["anomaly_flag"].sum()) if not anomalies_df.empty else 0
    anom_pct = (flagged_count / total_readings * 100) if total_readings > 0 else 0.0
    zones_list = ", ".join(sorted(filtered_df["zone"].unique().tolist()))

    lines = [
        "# EnergyIQ — Executive Energy Intelligence & Anomaly Report",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Telemetry Timeframe:** {start_date} to {end_date}",
        f"**Monitored Facility Zones:** {zones_list}",
        f"**Total Meter Readings:** {total_readings:,} recorded intervals",
        "",
        "## 1. Executive Consumption Summary",
        f"- **Total Energy Consumed:** {total_kwh:,.2f} kWh",
        f"- **Average Interval Consumption:** {avg_kwh:.2f} kWh per interval",
        f"- **Peak Demand Record:** {max_kwh:.2f} kWh (Recorded at {peak_ts} in {peak_zone})",
        "",
        "## 2. Operational Outlier & Anomaly Surveillance",
        f"- **Flagged Outlier Events:** {flagged_count:,} ({anom_pct:.2f}% of monitored intervals)",
        "- **Surveillance Model:** Rolling Causal Diurnal Median Baseline (shift=1)",
        "- **Statistical Bounds:** 3.0 × 1.4826 × MAD (Median Absolute Deviation)",
        "",
        "## 3. Prioritized AI Energy Conservation Recommendations",
    ]
    if raw_recs:
        for i, r in enumerate(raw_recs, 1):
            lines.append(f"### {i}. [{str(r.get('priority', 'Medium')).upper()}] {r.get('title', 'Recommendation')}")
            lines.append(f"**Category:** {r.get('category', 'Operational')}")
            lines.append(f"{r.get('description', '')}")
            lines.append("")
    else:
        lines.append("No critical conservation measures flagged. Facility operating within optimal baseline tolerances.")
    return "\n".join(lines)


def render_dashboard_overview(
    metrics: dict[str, Any],
    anomalies_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    source_label: str,
    param_threshold_kwh: float,
) -> None:
    """Render executive overview dashboard with KPI cards, primary charts, outlier logs, and AI recommendations."""
    render_page_header(
        "Dashboard Overview",
        "Continuous Energy Monitoring & Diurnal Outlier Intelligence",
        source_label,
    )

    summary = metrics.get("summary", {})
    total_kwh = summary.get("total_energy_kwh", 0.0)
    total_readings = summary.get("reading_count", 0)

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
        else:
            st.info("Zone breakdown requires telemetry with a 'zone' column.")

    # --- Secondary Two-Column Grid: Hourly Diurnal Profile (Left) + Daily Trend (Right) ---
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
                key="download_audit_overview",
                help="Download the complete anomaly audit table with baselines and deviations.",
            )

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
                "severity": "Severity Level",
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

        valid_recs = [
            r for r in raw_recs
            if isinstance(r, dict) and "title" in r and "description" in r
        ]

        if valid_recs:
            for rec in valid_recs[:4]:
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


def render_consumption_analytics(
    metrics: dict[str, Any],
    filtered_df: pd.DataFrame,
    source_label: str,
) -> None:
    """Render deep-dive consumption analytics with diurnal profiles, zone metrics, and load quantiles."""
    render_page_header(
        "Consumption Analytics",
        "Deep-Dive Diurnal Demand Profiling, Zone Load Balances & Empirical Distributions",
        source_label,
    )

    summary = metrics.get("summary", {})
    total_kwh = summary.get("total_energy_kwh", 0.0)
    avg_kwh = summary.get("average_interval_kwh", 0.0)
    max_kwh = summary.get("maximum_interval_kwh", 0.0)
    hourly_df = metrics.get("hourly", pd.DataFrame())
    daily_df = metrics.get("daily", pd.DataFrame())
    zone_df = metrics.get("zone_comparison", pd.DataFrame())

    if not hourly_df.empty:
        top_hr_row = hourly_df.loc[hourly_df["average_energy_kwh"].idxmax()]
        top_hr_val = int(top_hr_row["hour"])
        top_hr_avg = float(top_hr_row["average_energy_kwh"])
        off_peak_rows = hourly_df[hourly_df["hour"].isin([0, 1, 2, 3, 4, 5])]
        daytime_rows = hourly_df[hourly_df["hour"].isin(range(8, 17))]
        off_peak_avg = float(off_peak_rows["average_energy_kwh"].mean()) if not off_peak_rows.empty else 0.0
        daytime_avg = float(daytime_rows["average_energy_kwh"].mean()) if not daytime_rows.empty else 0.0
        off_peak_ratio = (off_peak_avg / daytime_avg * 100) if daytime_avg > 0 else 0.0
    else:
        top_hr_val = 0
        top_hr_avg = 0.0
        off_peak_avg = 0.0
        off_peak_ratio = 0.0

    load_factor = (avg_kwh / max_kwh) if max_kwh and max_kwh > 0 else 0.0

    cards = [
        {
            "title": "Peak Diurnal Hour",
            "value": f"Hour {top_hr_val:02d}:00",
            "unit": "",
            "badge": "DIURNAL MAX",
            "meta": f"{top_hr_avg:.2f} kWh interval mean",
            "icon": "⚡",
            "color_class": "kpi-blue",
            "badge_class": "badge-blue",
        },
        {
            "title": "Off-Peak Base Load",
            "value": f"{off_peak_avg:,.1f}",
            "unit": "kWh/int",
            "badge": "00:00–05:00",
            "meta": "Average night-time baseline",
            "icon": "🌙",
            "color_class": "kpi-teal",
            "badge_class": "badge-teal",
        },
        {
            "title": "Off-Peak / Daytime Ratio",
            "value": f"{off_peak_ratio:.1f}",
            "unit": "%",
            "badge": "BASE RATIO",
            "meta": "Night load relative to daytime",
            "icon": "⚖️",
            "color_class": "kpi-purple",
            "badge_class": "badge-purple",
        },
        {
            "title": "Facility Load Factor",
            "value": f"{load_factor:.2f}",
            "unit": "ratio",
            "badge": "UTILIZATION",
            "meta": "Mean demand / Peak demand",
            "icon": "📈",
            "color_class": "kpi-orange",
            "badge_class": "badge-orange",
        },
    ]
    render_generic_kpi_row(cards)

    # Primary charts
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(
            """
            <div class="glass-panel">
                <div class="glass-panel-title"><span>⏱️</span> Diurnal Load Profile by Hour of Day</div>
                <div class="glass-panel-subtitle">Mean energy draw per hourly block (00:00 to 23:00) with peak hour highlighted in amber.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_hourly_chart(hourly_df), use_container_width=True)

    with col_c2:
        st.markdown(
            """
            <div class="glass-panel">
                <div class="glass-panel-title"><span>📅</span> Daily Total Energy Trajectory</div>
                <div class="glass-panel-subtitle">Day-over-day aggregate facility consumption across the active telemetry window.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_daily_chart(daily_df), use_container_width=True)

    # Zone Breakdown and Quantile Analysis
    col_z1, col_z2 = st.columns([1.2, 1.4])
    with col_z1:
        st.markdown(
            """
            <div class="glass-panel">
                <div class="glass-panel-title"><span>🏢</span> Zone Consumption Totals</div>
                <div class="glass-panel-subtitle">Comparative aggregate consumption across metered facility sectors.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not zone_df.empty:
            st.plotly_chart(build_zone_chart(zone_df), use_container_width=True)
        else:
            st.info("Single-zone telemetry loaded.")

    with col_z2:
        st.markdown(
            """
            <div class="glass-panel">
                <div class="glass-panel-title"><span>📋</span> Zone Demand Breakdown Table</div>
                <div class="glass-panel-subtitle">Parametric distribution and share of total facility consumption.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not zone_df.empty:
            z_tbl = zone_df.copy()
            z_tbl["share_pct"] = (z_tbl["total_energy_kwh"] / total_kwh * 100) if total_kwh > 0 else 0.0
            if "maximum_interval_kwh" not in z_tbl.columns and "zone" in filtered_df.columns:
                max_map = filtered_df.groupby("zone")["energy_kwh"].max().to_dict()
                z_tbl["maximum_interval_kwh"] = z_tbl["zone"].map(max_map).fillna(0.0)
            elif "maximum_interval_kwh" not in z_tbl.columns:
                z_tbl["maximum_interval_kwh"] = z_tbl.get("average_interval_kwh", 0.0)
            z_display = z_tbl[["zone", "total_energy_kwh", "share_pct", "average_interval_kwh", "maximum_interval_kwh"]].copy()
            z_display = z_display.rename(columns={
                "zone": "Zone / Subsystem",
                "total_energy_kwh": "Total Energy (kWh)",
                "share_pct": "Share (%)",
                "average_interval_kwh": "Mean Interval (kWh)",
                "maximum_interval_kwh": "Peak Record (kWh)",
            })
            st.dataframe(
                z_display.style.format({
                    "Total Energy (kWh)": "{:,.1f}",
                    "Share (%)": "{:.1f}%",
                    "Mean Interval (kWh)": "{:,.2f}",
                    "Peak Record (kWh)": "{:,.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    # Empirical Load Distribution Quantiles
    st.markdown(
        """
        <div class="glass-panel">
            <div class="glass-panel-title"><span>📊</span> Empirical Interval Load Percentiles</div>
            <div class="glass-panel-subtitle">Statistical distribution of interval power readings (quantiles P0 to P100).</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    q = filtered_df["energy_kwh"].quantile([0.0, 0.25, 0.50, 0.75, 0.95, 1.0])
    q_cols = st.columns(6)
    q_labels = [
        ("Min (P0)", q.loc[0.0], "Absolute floor"),
        ("25th % (P25)", q.loc[0.25], "Lower quartile"),
        ("Median (P50)", q.loc[0.50], "Central tendency"),
        ("75th % (P75)", q.loc[0.75], "Upper quartile"),
        ("95th % (P95)", q.loc[0.95], "Near-peak demand"),
        ("Max (P100)", q.loc[1.0], "Historic maximum"),
    ]
    for col, (lbl, val, desc) in zip(q_cols, q_labels):
        with col:
            st.markdown(
                f"""
                <div style="background: #0E172E; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 14px; text-align: center;">
                    <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 700; text-transform: uppercase;">{lbl}</div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF; margin: 4px 0;">{val:,.2f}</div>
                    <div style="font-size: 0.68rem; color: #14D6A5;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_live_monitoring(
    anomalies_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    source_label: str,
) -> None:
    """Render high-frequency interval surveillance with replay disclaimer, live metrics, and stream logs."""
    render_page_header(
        "Live Monitoring",
        "High-Resolution Telemetry Stream Surveillance (Active Dataset Replay)",
        source_label,
    )

    latest_ts = filtered_df["timestamp"].max()
    latest_ts_str = latest_ts.strftime("%b %d, %Y %H:%M") if pd.notna(latest_ts) else "N/A"

    st.markdown(
        f"""
        <div class="quality-strip quality-warn">
            <span>📡 <strong>Telemetry Stream Status: Active Dataset Replay Mode</strong> • Operating on active dataset window ({latest_ts_str}). Real-time IoT/BACnet field gateway is in simulated replay mode. All readings below reflect verified high-frequency intervals from data ingestion.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest_records = filtered_df[filtered_df["timestamp"] == latest_ts]
    latest_draw = float(latest_records["energy_kwh"].sum()) if not latest_records.empty else 0.0

    latest_anom = anomalies_df[anomalies_df["timestamp"] == latest_ts]
    is_latest_anom = bool(latest_anom["anomaly_flag"].any()) if not latest_anom.empty else False

    records_24h = filtered_df[filtered_df["timestamp"] >= latest_ts - pd.Timedelta(hours=24)]
    rolling_24h_avg = float(records_24h["energy_kwh"].mean()) if not records_24h.empty else 0.0

    status_val = "OUTLIER" if is_latest_anom else "NORMAL"
    status_badge = "SPIKE DETECTED" if is_latest_anom else "WITHIN BOUNDS"
    status_color = "kpi-orange" if is_latest_anom else "kpi-teal"
    status_badge_color = "badge-orange" if is_latest_anom else "badge-teal"
    status_num_color = "#FCA5A5" if is_latest_anom else "#FFFFFF"

    cards = [
        {
            "title": "Latest Telemetry Point",
            "value": latest_ts.strftime("%H:%M") if pd.notna(latest_ts) else "N/A",
            "unit": latest_ts.strftime("%b %d") if pd.notna(latest_ts) else "",
            "badge": "INTERVAL STAMP",
            "meta": "Most recent recorded timestamp",
            "icon": "⏱️",
            "color_class": "kpi-blue",
            "badge_class": "badge-blue",
        },
        {
            "title": "Current Active Demand",
            "value": f"{latest_draw:,.1f}",
            "unit": "kWh",
            "badge": "INSTANTANEOUS",
            "meta": f"Aggregate draw at {latest_ts.strftime('%H:%M') if pd.notna(latest_ts) else 'N/A'}",
            "icon": "⚡",
            "color_class": "kpi-teal",
            "badge_class": "badge-teal",
        },
        {
            "title": "Surveillance Status",
            "value": status_val,
            "unit": "",
            "badge": status_badge,
            "meta": "Threshold & MAD outlier check",
            "icon": "🚨" if is_latest_anom else "✓",
            "color_class": status_color,
            "badge_class": status_badge_color,
            "num_color": status_num_color,
        },
        {
            "title": "24h Rolling Mean Load",
            "value": f"{rolling_24h_avg:,.1f}",
            "unit": "kWh/int",
            "badge": "TRAILING 24H",
            "meta": "Moving window interval average",
            "icon": "📊",
            "color_class": "kpi-purple",
            "badge_class": "badge-purple",
        },
    ]
    render_generic_kpi_row(cards)

    # 48-Hour Surveillance Trajectory Chart
    st.markdown(
        """
        <div class="glass-panel">
            <div class="glass-panel-title"><span>📡</span> High-Resolution Surveillance Feed (Recent 48 Hours)</div>
            <div class="glass-panel-subtitle">Near-realtime continuous interval trajectory tracking actual power draw against baseline bounds.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    records_48h = anomalies_df[anomalies_df["timestamp"] >= latest_ts - pd.Timedelta(hours=48)].copy()
    if not records_48h.empty:
        fig_live = build_timeseries_chart(records_48h)
        st.plotly_chart(fig_live, use_container_width=True)
    else:
        st.info("Insufficient telemetry to plot 48-hour trajectory.")

    # Live Stream Telemetry Buffer (Latest 20 readings)
    st.markdown(
        """
        <div class="glass-panel">
            <div class="glass-panel-title"><span>📋</span> Real-Time Telemetry Event Stream Buffer (Latest 20 Intervals)</div>
            <div class="glass-panel-subtitle">Live stream buffer of high-frequency metered intervals in reverse chronological order.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    recent_20 = anomalies_df.sort_values("timestamp", ascending=False).head(20).copy()
    recent_20["timestamp"] = recent_20["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    recent_20["status"] = recent_20["anomaly_flag"].map(lambda flag: "🔴 Outlier Flagged" if flag else "✓ Normal")
    recent_display = recent_20[["timestamp", "zone", "energy_kwh", "expected_baseline_kwh", "deviation_kwh", "status"]].copy()
    recent_display = recent_display.rename(columns={
        "timestamp": "Timestamp",
        "zone": "Zone",
        "energy_kwh": "Measured (kWh)",
        "expected_baseline_kwh": "Baseline (kWh)",
        "deviation_kwh": "Deviation (kWh)",
        "status": "Surveillance Flag",
    })
    st.dataframe(
        recent_display.style.format({
            "Measured (kWh)": "{:,.2f}",
            "Baseline (kWh)": "{:,.2f}",
            "Deviation (kWh)": "{:+,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )


def render_ai_recommendations(
    metrics: dict[str, Any],
    anomalies_df: pd.DataFrame,
    source_label: str,
) -> None:
    """Render prioritized AI energy conservation measures with deep-dive mathematical evidence."""
    render_page_header(
        "AI Recommendations",
        "Evidence-Based Energy Conservation Measures & Diurnal Optimization Actions",
        source_label,
    )

    try:
        raw_recs = generate_recommendations(metrics, anomalies_df)
    except Exception as exc:
        raw_recs = []
        st.error(f"Recommendation Engine Error: {exc}")

    valid_recs = [
        r for r in raw_recs
        if isinstance(r, dict) and "title" in r and "description" in r
    ]

    high_count = sum(1 for r in valid_recs if str(r.get("priority", "")).lower() == "high")
    med_count = sum(1 for r in valid_recs if str(r.get("priority", "")).lower() == "medium")
    low_count = sum(1 for r in valid_recs if str(r.get("priority", "")).lower() == "low")
    total_count = len(valid_recs)

    cards = [
        {
            "title": "Total ECMs Flagged",
            "value": str(total_count),
            "unit": "actions",
            "badge": "IDENTIFIED",
            "meta": "Actionable energy conservation measures",
            "icon": "💡",
            "color_class": "kpi-blue",
            "badge_class": "badge-blue",
        },
        {
            "title": "High Priority Actions",
            "value": str(high_count),
            "unit": "critical",
            "badge": "URGENT",
            "meta": "Significant surges or peak demands",
            "icon": "🔴",
            "color_class": "kpi-orange",
            "badge_class": "badge-orange",
            "num_color": "#FCA5A5" if high_count > 0 else "#FFFFFF",
        },
        {
            "title": "Medium Priority Actions",
            "value": str(med_count),
            "unit": "operational",
            "badge": "OPERATIONAL",
            "meta": "Schedule tuning & zone imbalances",
            "icon": "🟠",
            "color_class": "kpi-purple",
            "badge_class": "badge-purple",
        },
        {
            "title": "Advisory Opportunities",
            "value": str(low_count),
            "unit": "advisories",
            "badge": "INFORMATIONAL",
            "meta": "General conservation opportunities",
            "icon": "🔵",
            "color_class": "kpi-teal",
            "badge_class": "badge-teal",
        },
    ]
    render_generic_kpi_row(cards)

    st.markdown(
        """
        <div class="glass-panel">
            <div class="glass-panel-title"><span>💡</span> Prioritized Conservation Measures & Root Evidence</div>
            <div class="glass-panel-subtitle">Rule-driven, data-backed interventions with mathematical evidence extracted directly from telemetry baselines.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if valid_recs:
        for idx, rec in enumerate(valid_recs):
            priority = str(rec.get("priority", "Medium")).capitalize()
            category = str(rec.get("category", "Operational"))
            title = str(rec.get("title", "Recommendation"))
            desc = str(rec.get("description", ""))
            evidence = rec.get("evidence", {})

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
                <div style="background: #0E172E; border: 1px solid rgba(255, 255, 255, 0.08); border-left: 4px solid {accent_color}; border-radius: 10px; padding: 16px 18px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; color: #FFFFFF; font-size: 0.92rem;">{icon} {html.escape(title)}</span>
                        <span style="padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 700; text-transform: uppercase; {badge_style}">{html.escape(priority)} • {html.escape(category)}</span>
                    </div>
                    <p style="color: #94A3B8; font-size: 0.83rem; margin: 0; line-height: 1.5;">{html.escape(desc)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if isinstance(evidence, dict) and evidence:
                with st.expander(f"🔍 Inspect Mathematical Evidence — {html.escape(title)}", expanded=False):
                    ev_items = []
                    for k, v in evidence.items():
                        ev_key = k.replace("_", " ").title()
                        ev_items.append({"Evidence Metric": ev_key, "Observed Value": str(v)})
                    st.table(pd.DataFrame(ev_items))
    else:
        st.markdown(
            """
            <div class="quality-strip quality-clean">
                <span style="font-weight: 600;">✓ Optimization Normalcy: No critical efficiency anomalies or schedule conservation actions flagged for active telemetry window.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_reports_exports(
    metrics: dict[str, Any],
    anomalies_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    source_label: str,
    param_threshold_kwh: float,
) -> None:
    """Render export packages for anomaly audits, telemetry data, and executive briefings."""
    render_page_header(
        "Reports & Exports",
        "Download Audit Evidence, Filtered Interval Data & Formatted Executive Summaries",
        source_label,
    )

    try:
        raw_recs = generate_recommendations(metrics, anomalies_df)
    except Exception:
        raw_recs = []

    # Prepare Anomaly Export
    flagged_rows = anomalies_df[anomalies_df["anomaly_flag"] == True].copy()
    if not flagged_rows.empty:
        evidence_df = flagged_rows[
            ["timestamp", "zone", "energy_kwh", "expected_baseline_kwh", "deviation_kwh"]
        ].copy()
        evidence_df["deviation_pct"] = (
            (evidence_df["deviation_kwh"] / evidence_df["expected_baseline_kwh"]) * 100
        )
        evidence_df["severity"] = evidence_df["deviation_pct"].map(
            lambda pct: "Critical Spike (>100%)" if pct >= 100 else "Moderate Surge"
        )
        evidence_df = evidence_df.sort_values("deviation_kwh", ascending=False).reset_index(drop=True)
        csv_anom_data = evidence_df.copy()
        csv_anom_data["timestamp"] = csv_anom_data["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        anom_bytes = csv_anom_data.to_csv(index=False).encode("utf-8")
        total_flagged = len(evidence_df)
    else:
        evidence_df = pd.DataFrame()
        anom_bytes = b"timestamp,zone,energy_kwh,expected_baseline_kwh,deviation_kwh\n"
        total_flagged = 0

    # Prepare Telemetry Export
    telemetry_export = filtered_df[["timestamp", "zone", "energy_kwh"]].copy()
    telemetry_export["timestamp"] = telemetry_export["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    telemetry_bytes = telemetry_export.to_csv(index=False).encode("utf-8")

    # Prepare Executive Markdown Report
    exec_report_md = generate_executive_report(metrics, anomalies_df, filtered_df, raw_recs)
    report_bytes = exec_report_md.encode("utf-8")

    # 3 Download Packages in 3 Columns
    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        st.markdown(
            f"""
            <div style="background: #0E172E; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px; height: 100%;">
                <div style="font-size: 1.1rem; margin-bottom: 8px;">🚨 Anomaly Audit Log</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 14px; line-height: 1.45;">
                    Complete record of all <strong>{total_flagged:,}</strong> flagged outlier intervals with historical median baselines, deviation surges, and severity levels.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label=f"📥 Download Audit CSV ({total_flagged:,} rows)",
            data=anom_bytes,
            file_name="energyiq_anomaly_audit.csv",
            mime="text/csv",
            key="btn_download_audit_page",
            use_container_width=True,
        )

    with col_exp2:
        st.markdown(
            f"""
            <div style="background: #0E172E; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px; height: 100%;">
                <div style="font-size: 1.1rem; margin-bottom: 8px;">📊 Processed Telemetry</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 14px; line-height: 1.45;">
                    Cleaned, regularized interval consumption dataset for active date filters across <strong>{len(filtered_df):,}</strong> records.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label=f"📥 Download Telemetry CSV ({len(filtered_df):,} rows)",
            data=telemetry_bytes,
            file_name="energyiq_filtered_telemetry.csv",
            mime="text/csv",
            key="btn_download_telemetry_page",
            use_container_width=True,
        )

    with col_exp3:
        st.markdown(
            f"""
            <div style="background: #0E172E; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px; height: 100%;">
                <div style="font-size: 1.1rem; margin-bottom: 8px;">📑 Executive Briefing (.md)</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 14px; line-height: 1.45;">
                    Formatted executive report summarizing facility consumption KPIs, peak demand records, outlier stats, and prioritised AI recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label="📥 Download Executive Summary (.md)",
            data=report_bytes,
            file_name="energyiq_executive_summary.md",
            mime="text/markdown",
            key="btn_download_summary_page",
            use_container_width=True,
        )

    # Interactive Inspection Tabs
    st.markdown("---")
    st.markdown(
        """
        <div class="glass-panel">
            <div class="glass-panel-title"><span>🔍</span> Interactive Data Preview & Audit Inspection</div>
            <div class="glass-panel-subtitle">Review sample rows before exporting or auditing external compliance.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tab1, tab2, tab3 = st.tabs([
        f"🚨 Outlier Audit Log ({total_flagged:,})",
        f"📊 Telemetry Sample ({len(filtered_df):,})",
        "📑 Executive Briefing Preview",
    ])

    with tab1:
        if not evidence_df.empty:
            ev_preview = evidence_df.head(100).copy()
            ev_preview["timestamp"] = ev_preview["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
            ev_preview = ev_preview.rename(columns={
                "timestamp": "Timestamp",
                "zone": "Zone",
                "energy_kwh": "Measured (kWh)",
                "expected_baseline_kwh": "Baseline (kWh)",
                "deviation_kwh": "Surge (kWh)",
                "deviation_pct": "Deviation (%)",
                "severity": "Severity Level",
            })
            st.dataframe(
                ev_preview.style.format({
                    "Measured (kWh)": "{:,.2f}",
                    "Baseline (kWh)": "{:,.2f}",
                    "Surge (kWh)": "{:+,.2f}",
                    "Deviation (%)": "{:+,.1f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(f"Showing top 100 rows. Complete {total_flagged:,} records available via CSV download.")
        else:
            st.info("No anomalies detected in active selection.")

    with tab2:
        tel_preview = filtered_df.head(100).copy()
        tel_preview["timestamp"] = tel_preview["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            tel_preview.rename(columns={"timestamp": "Timestamp", "zone": "Zone", "energy_kwh": "Demand (kWh)"})
            .style.format({"Demand (kWh)": "{:,.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Showing first 100 of {len(filtered_df):,} interval readings.")

    with tab3:
        st.markdown(exec_report_md)


def render_settings_sensitivity(
    param_window: int,
    param_min_history: int,
    param_threshold_kwh: float,
    filtered_df: pd.DataFrame,
    source_label: str,
) -> None:
    """Render baseline sensitivity configuration, mathematical explanations, and data validation rules."""
    render_page_header(
        "Settings & Sensitivity",
        "Diurnal Baseline Windows, Robust MAD Multipliers & Telemetry Rules",
        source_label,
    )

    cards = [
        {
            "title": "Baseline Diurnal Window",
            "value": f"{param_window}",
            "unit": "Days",
            "badge": "ROLLING WINDOW",
            "meta": "Prior daily intervals used for median",
            "icon": "📅",
            "color_class": "kpi-blue",
            "badge_class": "badge-blue",
        },
        {
            "title": "Min History Required",
            "value": f"{param_min_history}",
            "unit": "intervals",
            "badge": "WARM-UP CUTOFF",
            "meta": "Minimum historical observations",
            "icon": "⏳",
            "color_class": "kpi-teal",
            "badge_class": "badge-teal",
        },
        {
            "title": "Absolute Deviation Floor",
            "value": f"{param_threshold_kwh:.1f}",
            "unit": "kWh",
            "badge": "SURGE FLOOR",
            "meta": "Minimum absolute surge threshold",
            "icon": "🛡️",
            "color_class": "kpi-orange",
            "badge_class": "badge-orange",
        },
    ]
    render_generic_kpi_row(cards)

    # Methodology Explanations
    st.markdown(
        """
        <div class="glass-panel">
            <div class="glass-panel-title"><span>📐</span> Mathematical Baseline & Outlier Detection Methodology</div>
            <div class="glass-panel-subtitle">Transparent mathematical criteria applied to identify consumption spikes without lookahead bias.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.markdown(
            """
            <div style="background: #0E172E; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px; height: 100%;">
                <div style="font-weight: 700; color: #14D6A5; margin-bottom: 8px;">1. Causal Diurnal Baseline</div>
                <p style="font-size: 0.8rem; color: #94A3B8; line-height: 1.5; margin: 0;">
                    Telemetry is grouped by <code>(zone, hour-of-day)</code>. For any interval at time <em>t</em>, the expected baseline is calculated strictly from prior days:
                </p>
                <div style="background: #132247; border-radius: 6px; padding: 8px; margin: 10px 0; font-family: monospace; font-size: 0.76rem; color: #E2E8F0;">
                    B(t) = Median(E[t-1d], E[t-2d], ... E[t-Wd])
                </div>
                <p style="font-size: 0.78rem; color: #64748B; margin: 0;">
                    A causal shift (<code>shift=1</code>) is enforced to ensure the current reading never contaminates its own expected baseline.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            """
            <div style="background: #0E172E; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px; height: 100%;">
                <div style="font-weight: 700; color: #3B82F6; margin-bottom: 8px;">2. Robust Spread (MAD)</div>
                <p style="font-size: 0.8rem; color: #94A3B8; line-height: 1.5; margin: 0;">
                    Traditional standard deviation is distorted by extreme spikes. EnergyIQ uses the Median Absolute Deviation (MAD), scaled for normal distribution consistency:
                </p>
                <div style="background: #132247; border-radius: 6px; padding: 8px; margin: 10px 0; font-family: monospace; font-size: 0.76rem; color: #E2E8F0;">
                    Spread = 3.0 × 1.4826 × MAD(E - B)
                </div>
                <p style="font-size: 0.78rem; color: #64748B; margin: 0;">
                    The <code>1.4826</code> constant normalizes MAD to standard deviation equivalence for Gaussian distributions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m3:
        st.markdown(
            """
            <div style="background: #0E172E; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px; height: 100%;">
                <div style="font-weight: 700; color: #F59E0B; margin-bottom: 8px;">3. Dual-Criterion Gate</div>
                <p style="font-size: 0.8rem; color: #94A3B8; line-height: 1.5; margin: 0;">
                    An interval is flagged as an operational outlier if and only if the absolute deviation exceeds both the statistical spread and absolute floor:
                </p>
                <div style="background: #132247; border-radius: 6px; padding: 8px; margin: 10px 0; font-family: monospace; font-size: 0.76rem; color: #E2E8F0;">
                    |E(t) - B(t)| &gt; max(Spread, Floor)
                </div>
                <p style="font-size: 0.78rem; color: #64748B; margin: 0;">
                    The floor prevents false alarms during low-consumption night-time hours when small fluctuations might exceed narrow MAD bands.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Ingestion & Validation Rules
    st.markdown("---")
    st.markdown(
        """
        <div class="glass-panel">
            <div class="glass-panel-title"><span>🛡️</span> Telemetry Validation & Data Integrity Rules</div>
            <div class="glass-panel-subtitle">Pipeline checks executed automatically during telemetry ingestion.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    quality_warnings = filtered_df.attrs.get("data_quality_warnings", [])
    interval_note = filtered_df.attrs.get("interval_note", "Regular sampling frequency detected.")

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown(
            f"""
            <div style="background: #0E172E; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px;">
                <div style="font-weight: 600; color: #FFFFFF; font-size: 0.88rem; margin-bottom: 8px;">Active Sampling Cadence</div>
                <div style="color: #14D6A5; font-size: 0.82rem; margin-bottom: 8px;">{html.escape(interval_note)}</div>
                <p style="color: #94A3B8; font-size: 0.78rem; line-height: 1.45; margin: 0;">
                    Interval timestamps are automatically parsed, chronologically sorted, and validated for uniform cadence across metered facility zones.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_q2:
        st.markdown(
            f"""
            <div style="background: #0E172E; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px;">
                <div style="font-weight: 600; color: #FFFFFF; font-size: 0.88rem; margin-bottom: 8px;">Quality Notices & Sanitization</div>
                <div style="color: {'#F59E0B' if quality_warnings else '#14D6A5'}; font-size: 0.82rem; margin-bottom: 8px;">
                    {f"{len(quality_warnings)} advisory notices flagged" if quality_warnings else "✓ Zero anomalies in telemetry format"}
                </div>
                <p style="color: #94A3B8; font-size: 0.78rem; line-height: 1.45; margin: 0;">
                    Schema enforcement: Requires <code>timestamp</code> and <code>energy_kwh</code>. Negative readings are rejected.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# 8. Main Application Logic
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

    # --- Sidebar: Interactive Navigation Menu ---
    NAV_OPTIONS = [
        "📊 Dashboard Overview",
        "📈 Consumption Analytics",
        "📡 Live Monitoring",
        "💡 AI Recommendations",
        "📑 Reports & Exports",
        "⚙️ Settings & Sensitivity",
    ]

    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = NAV_OPTIONS[0]

    nav_selection = st.sidebar.radio(
        "Navigation Menu",
        options=NAV_OPTIONS,
        key="nav_selection",
        label_visibility="collapsed",
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
        help="Loads the Tetouan City power consumption dataset containing 157,248 readings at 10-minute intervals from 2017.",
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

    # --- Route to Active Page View ---
    if nav_selection == "📊 Dashboard Overview":
        render_dashboard_overview(metrics, anomalies_df, filtered_df, source_label, param_threshold_kwh)
    elif nav_selection == "📈 Consumption Analytics":
        render_consumption_analytics(metrics, filtered_df, source_label)
    elif nav_selection == "📡 Live Monitoring":
        render_live_monitoring(anomalies_df, filtered_df, source_label)
    elif nav_selection == "💡 AI Recommendations":
        render_ai_recommendations(metrics, anomalies_df, source_label)
    elif nav_selection == "📑 Reports & Exports":
        render_reports_exports(metrics, anomalies_df, filtered_df, source_label, param_threshold_kwh)
    elif nav_selection == "⚙️ Settings & Sensitivity":
        render_settings_sensitivity(param_window, param_min_history, param_threshold_kwh, filtered_df, source_label)

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
