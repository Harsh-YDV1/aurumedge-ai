# AurumEdge AI — Beyond Gold Prices. Into Market Intelligence.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-24%20passed-brightgreen.svg)](https://docs.pytest.org/)

**Production-Quality Quantitative Derivatives Intelligence MVP | Multi Commodity Exchange of India (MCX)**  
*Developed by Team Proxy for Hackathon Demonstration*

---

## 1. Project Overview & Central Philosophy

> **"A price difference is not automatically an opportunity. First, it must be measured correctly."**

**AurumEdge AI** is a commodity derivatives intelligence platform designed for relative-value analysis across Multi Commodity Exchange of India (MCX) gold futures contracts:
* **GOLD** (Standard 1kg, 995 fineness)
* **GOLDM** (Gold Mini 100g, 995 fineness)
* **GOLDTEN** (Gold Ten 10g, 999 fineness)
* **GOLDGUINEA** (Gold Guinea 8g, 999 fineness)
* **GOLDPETAL** (Gold Petal 1g, 999 fineness)

Traders and jewellers routinely observe divergent prices across these contracts (e.g. ₹72,500 vs ₹58,000 vs ₹7,295). Naive comparisons fail because they overlook differences in **quotation units**, **physical purity**, **delivery lot sizes**, **expiration horizons (cost of carry)**, and **statutory Indian exchange levies**.

AurumEdge AI rigorously evaluates whether relative price differences between economically comparable contracts are statistically unusual after full physical and financial normalization.

---

## 2. The 5-Stage Analytical Hierarchy

AurumEdge AI explicitly enforces five distinct analytical stages:

```
[ Stage 1: Valid Normalized Observation ]
       q_i = P_i / [g_i * (u_i / 1000)]
                  │
                  ▼
[ Stage 2: Statistically Unusual Spread ]
       |z_t| > 2.0σ (Rolling Point-in-Time)
                  │
                  ▼
[ Stage 3: Candidate Research Signal ]
       Deterministic Audit Attribution
                  │
                  ▼
[ Stage 4: Eligible Signal ]
       Passes Liquidity (Vol, OI) & Tender Proximity Buffers
                  │
                  ▼
[ Stage 5: Historically Tested Result ]
       Walk-Forward Backtest with Next-Bar (t+1) Execution & MCX Fees
```

*Never imply that an anomaly guarantees convergence, risk-free arbitrage, or executable profit.*

---

## 3. Mathematical Foundations

### 3.1 Purity & Quotation Normalization
Quoted prices $P_i$ are converted to pure fine-gold equivalent price $q_i$ (₹ per gram of 100.0% pure gold):

$$q_i = \frac{P_i}{g_i \times \left(\frac{u_i}{1000}\right)}$$

where:
* $P_i$: Observed quoted exchange price.
* $g_i$: Quotation-unit weight in grams (e.g., 10g for GOLD/GOLDM/GOLDTEN, 8g for GOLDGUINEA, 1g for GOLDPETAL).
* $u_i$: Purity expressed per 1,000 (fineness: 995‰ for standard gold/mini, 999‰ for ten/guinea/petal).
* $q_i$: Derived fine-gold price per gram.

> **Crucial Invariant:** **Quotation Unit** ($g_i$) is strictly separated from **Trading Unit** ($L_i$). The trading unit represents deliverable lot exposure and is used exclusively for lot sizing, margin, and transaction fees.

### 3.2 Relative Spread & Point-in-Time Statistics
For two contracts with the **same contract expiry date** ($T_{E, A} == T_{E, B}$):

$$s_t = \ln\left(\frac{q_{A, t}}{q_{B, t}}\right)$$

To eliminate look-ahead bias, rolling statistics at observation session $t$ are calculated strictly over past sessions $[t-W, t-1]$ using a 1-period shift:

$$\mu_t = \frac{1}{W} \sum_{k=1}^W s_{t-k}, \quad \sigma_t = \sqrt{\frac{1}{W-1} \sum_{k=1}^W (s_{t-k} - \mu_t)^2}$$

$$z_t = \frac{s_t - \mu_t}{\sigma_t}$$

A robust Median / MAD anomaly metric is also computed alongside Gaussian Z-scores:

$$z_{\text{MAD}, t} = \frac{s_t - \text{median}(s)}{1.4826 \times \text{MAD}_t}$$

### 3.3 Indian MCX Transaction Cost Schedule
* **Brokerage**: 0.02% (2 bps) of turnover
* **MCX Exchange Turnover Charge**: 0.0021%
* **SEBI Regulatory Fee**: 0.0001% (₹10 per crore)
* **Commodities Transaction Tax (CTT)**: 0.01% on non-agri sell turnover
* **Stamp Duty**: 0.002% on buy turnover
* **GST**: 18% on (Brokerage + Exchange Fees)
* **Slippage**: Configurable (default 1 tick per leg)

---

## 4. Technology Stack & Repository Structure

* **Language & Runtime**: Python 3.12
* **Data Processing**: Pandas 2.2.3, NumPy 2.5
* **Storage**: PyArrow 26.0 (Columnar Parquet & CSV)
* **Presentation**: Streamlit 1.66
* **Interactive Charting**: Plotly 7.1
* **Testing & Quality**: Pytest 9.2, Ruff 0.17

```
aurumedge-ai/
├── app.py                     # Streamlit research cockpit entry point
├── requirements.txt           # Pinned dependencies
├── README.md                  # System manual and documentation
├── .gitignore                 # Version control ignores
├── config/
│   ├── contracts.csv          # Official MCX contract specifications master
│   └── settings.toml          # Operational defaults & fee schedules
├── data/
│   ├── raw/                   # Ingested MCX Bhavcopy CSV files
│   ├── processed/             # Cleaned canonical Parquet / CSV datasets
│   └── sample/                # Deterministic synthetic sample dataset
├── src/
│   ├── __init__.py
│   ├── storage.py             # Data I/O, Parquet caching, TOML loader
│   ├── normalization.py       # Purity and quotation unit normalizer
│   ├── ingest.py              # Multi-layout CSV parser & header mapper
│   ├── validation.py          # Data-quality rules, sanity & bound checks
│   ├── lifecycle.py           # Days-to-expiry & tender period metrics
│   ├── matching.py            # Same-expiry pair alignment engine
│   ├── signals.py             # Point-in-time Z-score & MAD anomaly engine
│   ├── metrics.py             # Financial risk metrics & MCX fee schedules
│   ├── backtest.py            # Chronological walk-forward strategy simulator
│   └── demo_data.py           # Cointegrated synthetic market generator
├── components/
│   ├── __init__.py
│   ├── charts.py              # Interactive Plotly dark navy & gold charts
│   ├── tables.py              # Formatted financial tables & ledgers
│   └── alert_cards.py         # Visual metric cards & explainability callouts
├── tests/
│   ├── test_normalization.py  # Hand-calculated math verification tests
│   ├── test_validation.py     # Data consistency & price bounds tests
│   ├── test_ingest.py         # Multi-format CSV ingestion tests
│   ├── test_matching.py       # Same-expiry & overlap tests
│   ├── test_signals.py        # Look-ahead prevention & filter tests
│   └── test_backtest.py       # t+1 execution & cost drag tests
└── docs/
    ├── architecture.md        # Architectural topology
    ├── methodology.md         # Quantitative research formulation
    ├── data_dictionary.md     # Field specifications & data provenance
    └── testing.md             # Automated test suite protocol
```

---

## 5. Quickstart & Installation (Windows PowerShell)

### Prerequisites
* Windows 10/11 with Python 3.12 installed.

### Setup Virtual Environment
```powershell
# 1. Clone repository and navigate to workspace
cd c:\Users\govin\OneDrive\Documents\H

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
.venv\Scripts\Activate.ps1

# 4. Install dependencies
.venv\Scripts\pip install -r requirements.txt
```

---

## 6. How to Run & Demonstrate

### 1. Launch Interactive Streamlit Cockpit
```powershell
.venv\Scripts\streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Run Automated Test Suite
```powershell
.venv\Scripts\pytest -v tests/
```
*Executes all 24 automated unit and integration tests verifying mathematical correctness, zero look-ahead bias, and t+1 execution.*

### 3. Run Static Analysis & Linter
```powershell
.venv\Scripts\ruff check src/ tests/ components/ app.py
```

### 4. Working with Real MCX Bhavcopy Data
1. Download daily MCX Bhavcopy CSV files from [MCX India](https://www.mcxindia.com/market-data/bhavcopy).
2. Place CSV files inside `data/raw/` or use the **Upload MCX Bhavcopy CSV** widget in the application sidebar.
3. The platform validates, deduplicates, and caches canonical clean records in `data/processed/`.

---

## 7. Research Disclaimer

*AurumEdge AI is an academic and quantitative research prototype developed for educational and decision-support purposes. It does not provide financial advice, broker integration, or automated trade execution. All commodity futures trading involves risk of capital loss.*
