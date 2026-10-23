# EnergyIQ — Smart Energy Consumption Optimizer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75.svg)](https://plotly.com/)
[![Tests Passing](https://img.shields.io/badge/pytest-70%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/Dataset%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **EnergyIQ** is an explainable, data-backed smart energy analytics and demand optimization platform. Built for facility managers, energy auditors, and grid operators, EnergyIQ transforms raw smart meter interval telemetry into actionable, prioritized conservation measures, causal diurnal baselines, and transparent anomaly audits.

---

## 📌 Problem Statement

Commercial facilities, industrial campuses, and municipal power distribution grids generate vast volumes of high-frequency smart meter readings. However, operational teams face persistent hurdles:

1. **Opaque Demand Spikes & Capacity Penalties**: Uncontrolled surges during peak tariff windows significantly inflate commercial demand charges.
2. **The "Black-Box" AI Trust Deficit**: Complex machine learning anomaly models produce false positives and lack explainability, making facility managers hesitant to enact expensive equipment shutdowns or setpoint adjustments.
3. **Off-Peak Baseload Waste**: Unoccupied buildings often consume excessive power overnight due to misconfigured HVAC schedules, lighting overrides, or malfunctioning equipment.
4. **Multi-Zone Imbalances**: Facilities with multiple submeters or distribution zones struggle to quickly isolate which sub-network is driving disproportionate demand.
5. **Inactionable Reporting**: Raw CSV logs and basic charts do not deliver concrete, prioritized steps for energy reduction.

**EnergyIQ resolves these challenges** with causal diurnal baselining, robust Median Absolute Deviation (MAD) spread boundaries, and a deterministic, evidence-based recommendation engine that cites exact timestamps, deviation magnitudes, and mathematical justifications.

---

## ⚡ Key Features

EnergyIQ offers a cohesive dark navy executive monitoring console structured across six interactive navigation modules:

### 1. 📊 Dashboard Overview
- **Vivid Gradient KPI Cards**: High-visibility metrics for Total Consumption (Blue), Average Interval Load (Teal), Detected Outliers (Purple), and Peak Recorded Demand (Orange).
- **Synchronized Time-Series Chart**: Prominent line visualization featuring smooth curves, distinct interval data points, dashed rolling diurnal baselines, and glowing red anomaly markers.
- **Zone Demand Comparison**: Horizontal bar chart comparing aggregate energy draw across metered facility zones with alternating color accents.
- **Top-100 Anomaly Audit Log**: Tabular record of top operational outliers ranked by absolute surge magnitude, complete with baseline comparisons, percentage deviation, and one-click CSV export.
- **Quantitative Demand Insights**: Key mathematical facts derived from telemetry (e.g., peak diurnal hour, busiest consumption day, highest-consuming zone, and warm-up coverage).

### 2. 📈 Consumption Analytics
- **24-Hour Diurnal Demand Profile**: Hourly mean energy draw (00:00 to 23:00) with peak hours highlighted to expose operational ramp-up patterns.
- **Daily Aggregate Trajectory**: Day-over-day aggregate energy trajectory revealing weekly work cycles.
- **Facility Load Factor**: Ratio of average interval load to maximum peak load to quantify demand uniformity.
- **Empirical Interval Load Percentiles**: Statistical distribution table providing quantile values (P0 Min, P25, P50 Median, P75, P95, and P100 Max).

### 3. 📡 Live Monitoring
- **Active Dataset Replay Mode**: High-frequency interval surveillance operating on verified dataset intervals with clear operational labeling (no fabricated real-time readings).
- **Instantaneous Demand Tracking**: Live interval readings, trailing 24-hour rolling averages, and immediate outlier threshold status.
- **48-Hour Continuous Trajectory Chart**: Near-realtime surveillance feed tracking measured power against dynamic baseline boundaries.
- **Live Stream Event Buffer**: Reverse-chronological table of the most recent 20 interval records with surveillance status flags.

### 4. 💡 AI Energy Recommendations
- **Rule-Driven Conservation Measures (ECMs)**: Explainable, deterministic recommendations categorized into *Peak Demand*, *Operational Anomaly*, *Off-Peak Baseline*, *Zone Imbalance*, and *High Overall Consumption*.
- **Strict Priority Hierarchy**: High (critical cost/load spikes), Medium (operational tuning & zone balance), and Low (general conservation advisories).
- **Mathematical Evidence Inspect**: Expandable drawers (`st.expander`) displaying exact metric inputs, observed values, and baseline numbers for full engineering auditability.

### 5. 📑 Reports & Exports
- **Anomaly Audit CSV (`energyiq_anomaly_audit.csv`)**: Full log of all flagged intervals, expected baselines, and deviation amounts.
- **Filtered Telemetry CSV (`energyiq_filtered_telemetry.csv`)**: Cleaned, validated interval data corresponding to active sidebar date and zone filters.
- **Executive Summary Briefing (`energyiq_executive_summary.md`)**: Markdown-formatted executive report containing key performance indicators, peak demand records, anomaly statistics, and top AI recommendations.
- **Interactive Inspection Tabs**: Preview up to 100 sample records directly within the UI before downloading.

### 6. ⚙️ Settings & Sensitivity
- **Interactive Sensitivity Tuner**: Adjust diurnal history windows (2–24 days), minimum warm-up intervals (1–10), and absolute deviation floors (0.1–50.0 kWh).
- **Transparent Methodological Documentation**: In-app mathematical formulation detailing causal baseline construction and MAD scaling.

---

## 📐 Mathematical & Algorithmic Foundation

EnergyIQ avoids non-causal contamination and distribution skew by adhering to rigorous statistical methods:

1. **Causal Diurnal Baseline**:
   For any interval at timestamp $t$ and hour-of-day $h$, the baseline is computed strictly from the same hour across preceding days:
   $$\text{Baseline}(t) = \text{Median}\big(E[t - 1\,\text{day}],\, E[t - 2\,\text{days}],\, \dots,\, E[t - W\,\text{days}]\big)$$
   Enforcing a causal lag (`shift=1`) guarantees that current consumption spikes never distort or contaminate their own expected baseline.

2. **Robust Spread Scaling (MAD)**:
   Standard deviation is heavily distorted by the very outliers it attempts to detect. EnergyIQ uses the Median Absolute Deviation (MAD), normalized to standard deviation equivalence for Gaussian distributions:
   $$\text{Spread}(t) = 3.0 \times 1.4826 \times \text{MAD}\big(E - \text{Baseline}\big)$$

3. **Dual-Criterion Outlier Gate**:
   An interval is flagged as an operational outlier if and only if:
   $$|E(t) - \text{Baseline}(t)| > \max\big(\text{Spread}(t),\; \text{Threshold Floor}\big)$$
   The floor parameter prevents false alerts during low-consumption overnight hours when tight MAD bands could otherwise trigger noise.

---

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Core programming runtime |
| **Web Framework** | Streamlit | SaaS-grade reactive dashboard interface |
| **Data Processing** | Pandas, NumPy | High-performance interval data cleaning and aggregation |
| **Visualizations** | Plotly | Dynamic, interactive charts with custom dark navy palettes |
| **Testing** | Pytest | Comprehensive test suite covering recommendation rules and contracts |

---

## 📂 Project Structure

```text
energyiq-smart-energy-optimizer/
│
├── README.md                          # Comprehensive project documentation
├── requirements.txt                   # External dependencies
├── app.py                             # Streamlit dashboard application & navigation router
├── analytics.py                       # Data validation, metrics calculation, and anomaly detection
├── recommendations.py                 # Evidence-based, explainable recommendation engine
│
├── data/
│   ├── README.md                      # Dataset documentation, unit conversions, and citations
│   ├── sample_energy.csv              # Primary demo dataset (UCI Tetouan City benchmark)
│   ├── sample_energy_synthetic.csv    # 72-hour synthetic test fixture
│   └── tetouan_source.zip             # Raw source archive
│
└── tests/
    ├── __init__.py
    └── test_recommendations.py        # 70 unit tests covering recommendation rules and contracts
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12+
- `pip` package manager
- Recommended: Virtual environment (`venv` or `conda`)

### Step 1: Clone the Repository
```bash
git clone https://github.com/samikshabhusare-hue/energyiq-smart-energy-optimizer.git
cd energyiq-smart-energy-optimizer
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Windows (PowerShell / Command Prompt)
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

Launch the Streamlit dashboard locally:

```bash
streamlit run app.py
```

The application will start and automatically open in your default browser at:
```text
http://localhost:8501
```

> **Note on Initial Launch**: EnergyIQ automatically loads `data/sample_energy.csv` (157,248 interval readings from the Tetouan City benchmark) on startup, rendering all KPI cards, Plotly charts, and anomaly logs immediately. You can upload custom telemetry CSVs at any time using the sidebar file uploader.

---

## 🧪 Running Tests & Code Quality Checks

Verify application compilation and run the full recommendation engine test suite:

### 1. Compile Check
```bash
python -m py_compile app.py
```

### 2. Run Unit Test Suite
```bash
python -m pytest tests/test_recommendations.py -v
```

All 70 test cases validate:
- Empty, missing, and malformed input handling
- Safe treatment of non-numeric readings (`NaN`, `Inf`, strings)
- Mathematical consistency of evidence metrics
- Strict priority sorting (High → Medium → Low)
- Benchmark contract validation against the Tetouan dataset

---

## 📊 Telemetry Data Schema

EnergyIQ supports both single-zone and multi-zone CSV telemetry. Required and optional columns are:

| Column Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | Timestamp | **Yes** | Parseable date/time (e.g. `YYYY-MM-DD HH:MM:SS` or `ISO 8601`) |
| `energy_kwh` | Float | **Yes** | Cumulative energy consumed in the interval ($\text{kWh} \ge 0$) |
| `zone` | String | *Optional* | Facility sector, zone, or submeter ID (defaults to `"All"` if omitted) |

---

## 📜 Dataset Attribution & License

- **Source**: [UCI Machine Learning Repository — Power Consumption of Tetouan City](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city) (DOI: [10.24432/C5B034](https://doi.org/10.24432/C5B034)).
- **Authors**: Abdulwahed Salam and Abdelaaziz El Hibaoui (Abdelmalek Essaadi University, Tetouan, Morocco).
- **License**: Distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
- **Unit Conversion**: Instantaneous power in kilowatts ($\text{kW}$) captured every 10 minutes was converted to interval energy in kilowatt-hours ($\text{kWh}$) via $\text{kWh} = \text{kW} / 6$. See [data/README.md](data/README.md) for full conversion derivations and academic citations.
