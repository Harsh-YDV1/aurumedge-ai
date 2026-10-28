# AurumEdge AI — System Architecture

## 1. Architectural Overview

AurumEdge AI is designed around a modular, decoupled quantitative analytics pipeline. The design separates raw exchange ingestion, contract specification management, price normalization, contract matching, signal generation, and historical validation from the presentation layer.

```
+-------------------------------------------------------------------------+
|                               Data Sources                              |
|   +------------------------------+   +------------------------------+   |
|   |  MCX Bhavcopy CSVs (Manual)  |   |  Synthetic Generator (Demo)  |   |
|   +--------------+---------------+   +--------------+---------------+   |
+------------------|----------------------------------|-------------------+
                   v                                  v
+-------------------------------------------------------------------------+
|                             Ingestion Layer                             |
|               (src/ingest.py & src/demo_data.py)                        |
|  - Multi-file parsing & header normalization                            |
|  - Explicit price field mapping (Close vs Settlement)                   |
|  - Provenance tagging (MCX_OFFICIAL_RAW vs SYNTHETIC_DEMO)              |
+----------------------------------+--------------------------------------+
                                   v
+-------------------------------------------------------------------------+
|                            Validation Engine                            |
|                           (src/validation.py)                           |
|  - Schema conformance & duplicate elimination                           |
|  - High/Low violation & nonpositive price check                         |
|  - Missing expiry detection & data-quality reporting                    |
+----------------------------------+--------------------------------------+
                                   v
+-------------------------------------------------------------------------+
|                     Contract Master & Normalization                     |
|            (config/contracts.csv & src/normalization.py)                |
|  - Quotation unit vs. Trading unit segregation                          |
|  - Fineness adjustment: q_i = P_i / [g_i * (u_i / 1000)]                |
|  - Full intermediate breakdown for algorithmic explainability           |
+----------------------------------+--------------------------------------+
                                   v
+-------------------------------------------------------------------------+
|                       Contract Matching & Lifecycle                     |
|                   (src/matching.py & src/lifecycle.py)                  |
|  - Same-expiry constraint enforcement                                   |
|  - Joint calendar intersection & overlap verification                   |
|  - Tender period / expiry buffer proximity filtering                    |
+----------------------------------+--------------------------------------+
                                   v
+-------------------------------------------------------------------------+
|                       Relative-Value Signal Engine                      |
|                            (src/signals.py)                             |
|  - Scale-invariant log spread: s_t = ln(q_A / q_B)                      |
|  - Strictly point-in-time rolling statistics (Zero Look-Ahead Bias)     |
|  - Dual anomaly scoring: Gaussian Z-Score & Robust Median/MAD           |
|  - Multi-tier filtering (Liquidity, History, Expiry)                    |
|  - Deterministic human-readable explanation generation                  |
+----------------------------------+--------------------------------------+
                                   v
+-------------------------------------------------------------------------+
|                  Historical Backtesting & Research Suite                |
|                   (src/backtest.py & src/metrics.py)                    |
|  - Signal Study Mode (unconstrained forward mean reversion)             |
|  - Strategy Backtest Mode (next-bar execution t+1, margin, stops)       |
|  - Full MCX Transaction Cost Model (Brokerage, CTT, SEBI, GST, Slippage)|
|  - Performance attribution & risk metrics (Sharpe, Max DD, Turnover)    |
+----------------------------------+--------------------------------------+
                                   v
+-------------------------------------------------------------------------+
|                           Streamlit UI Suite                            |
|                       (app.py & components/)                            |
|  - Dark navy & gold institutional financial interface                   |
|  - Interactive Plotly relative spread & Z-score charts                  |
|  - Real vs. Demo mode toggle with safety boundaries                     |
|  - Expandable mathematical breakdown & audit trail                      |
+-------------------------------------------------------------------------+
```

---

## 2. Directory Structure

```
aurumedge-ai/
  app.py                     # Streamlit application entry point
  requirements.txt           # Python pinned dependencies
  README.md                  # Comprehensive documentation and quickstart
  config/
    contracts.csv            # Official MCX contract specifications master
    settings.toml            # Operational parameters & Indian tax/fee rates
  data/
    raw/                     # Ingested MCX Bhavcopy CSV files
    processed/               # Cleaned canonical Parquet / CSV datasets
    sample/                  # Offline deterministic synthetic samples
  src/
    __init__.py
    ingest.py                # Multi-format CSV ingestion & provenance
    validation.py            # Rule-based data quality & consistency validation
    normalization.py         # Quotation & purity fine-gold normalizer
    matching.py              # Cross-contract pairing & alignment
    lifecycle.py             # Days-to-expiry & tender period analytics
    signals.py               # Point-in-time relative-value signal engine
    backtest.py              # Chronological walk-forward strategy simulation
    metrics.py               # Financial risk, drawdown, and cost metrics
    storage.py               # Local caching, Parquet I/O, and data lineage
    demo_data.py             # Deterministic synthetic market simulator
  components/
    __init__.py
    charts.py                # Plotly financial visualizations
    tables.py                # Data tables & quality reports
    alert_cards.py           # Anomaly cards with explainable reasoning
  tests/
    test_ingest.py
    test_validation.py
    test_normalization.py
    test_matching.py
    test_signals.py
    test_backtest.py
  docs/
    architecture.md
    methodology.md
    data_dictionary.md
    testing.md
```

---

## 3. Technology Rationale

1. **Python 3.12**: Modern, performant runtime with standard typing and datetime enhancements.
2. **Pandas & NumPy**: High-performance vectorized tabular mathematics.
3. **PyArrow**: High-throughput columnar Parquet storage preserving strict datatypes.
4. **Streamlit**: Interactive rapid research cockpit without web server / build-chain complexity.
5. **Plotly**: Publication-grade interactive financial charting (zoom, pan, inspect tooltips).
6. **Pytest**: Deterministic unit and integration test verification.
7. **Ruff**: High-speed PEP-8 compliance and static analysis.
