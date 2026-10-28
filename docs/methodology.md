# AurumEdge AI — Quantitative Research Methodology

## 1. Executive Summary & Core Philosophy

> **"A price difference is not automatically an opportunity. First, it must be measured correctly."**

AurumEdge AI is a commodity derivatives intelligence platform designed for relative-value analysis across Multi Commodity Exchange of India (MCX) gold futures contracts:
* **GOLD** (Standard 1kg, 995 fineness)
* **GOLDM** (Gold Mini 100g, 995 fineness)
* **GOLDTEN** (Gold Ten 10g, 999 fineness)
* **GOLDGUINEA** (Gold Guinea 8g, 999 fineness)
* **GOLDPETAL** (Gold Petal 1g, 999 fineness)

The platform evaluates whether price differentials between economically comparable contracts are statistically anomalous after controlling for:
1. **Quotation unit differences** (₹/10g vs ₹/8g vs ₹/1g)
2. **Purity / Fineness differences** (995/1000 vs 999/1000)
3. **Contract maturity & carry costs** (Contango/backwardation)
4. **Liquidity & Open Interest constraints**
5. **Contract lifecycle & delivery tender period risks**
6. **Statutory & exchange transaction costs**

AurumEdge AI operates as a research and decision-support prototype, not an autonomous algorithmic execution system. It does not promise guaranteed arbitrage or risk-free profit.

---

## 2. Purity & Quotation Normalization

Exchange quotes are expressed in divergent physical dimensions and purity grades. Naive comparison of raw prices leads to false alarms.

### Mathematical Formulation
Let:
* $P_{i, t}$ = Observed market price of contract $i$ at time $t$
* $g_i$ = Quotation-unit weight in grams (e.g., 10g for GOLDM, 8g for GOLDGUINEA, 1g for GOLDPETAL)
* $u_i$ = Purity expressed per 1,000 (fineness, e.g. 995 for GOLDM, 999 for GOLDPETAL)
* $q_{i, t}$ = Derived normalized price per gram of pure (1,000 fineness) gold equivalent

$$q_{i, t} = \frac{P_{i, t}}{g_i \times \left(\frac{u_i}{1000}\right)}$$

### Separation of Quotation Unit and Trading Unit
* **Quotation Unit ($g_i$)**: The denominator for the price quote.
* **Trading Unit ($L_i$)**: The minimum lot size contractually deliverable (e.g. 100g for GOLDM, 8g for GOLDGUINEA). Lot size is used exclusively for capital exposure, position sizing, margin, and transaction fees.

---

## 3. Contract Matching & Lifecycle Policy

Futures contracts differing by expiration date incorporate interest rates, storage, insurance, and convenience yield (cost of carry). Comparing different expiries introduces term-structure risk rather than pure relative-value mispricing.

### Matching Constraints
1. **Exact Expiry Match**: For the core relative-value spread, contract $A$ and contract $B$ must share the exact expiration date $T_E$.
2. **Common Trading Date**: Both observations must occur on session date $t$.
3. **Minimum Historical Overlap**: The pair must possess at least $W_{\text{min}}$ (default: 15) concurrent trading days.
4. **Lifecycle Tender Proximity Filter**: Signals within $\delta$ days (default: 3 business days) of contract expiration are disqualified due to MCX physical delivery tender period margins, widening bid-ask spreads, and delivery settlement risks.

---

## 4. Relative-Value Signal Engine

### 1. Relative Spread
For normalized fine-gold prices $q_A$ and $q_B$:

$$s_t = \ln\left(\frac{q_{A, t}}{q_{B, t}}\right)$$

Log spread is scale-invariant and symmetric.

### 2. Point-in-Time Rolling Statistics (Zero Look-Ahead Bias)
Statistics are strictly calculated over past observations $[t - W, t - 1]$ to avoid look-ahead bias:

$$\mu_t = \frac{1}{W} \sum_{k=1}^W s_{t-k}$$

$$\sigma_t = \sqrt{\frac{1}{W-1} \sum_{k=1}^W (s_{t-k} - \mu_t)^2}$$

### 3. Z-Score Deviation
$$z_t = \frac{s_t - \mu_t}{\sigma_t}$$

* $|z_t| \le 2.0$: **Normal relationship**
* $|z_t| > 2.0$: **Statistically unusual spread** (Candidate signal)
* If $\sigma_t = 0$: Flagged as zero variance / illiquid, score set to 0.

### 4. Robust Dispersion (Median / MAD)
To protect against single-day quotation fat-tails:

$$\text{MAD}_t = \text{median}\left(|s_{t-k} - \text{median}(s)|\right)$$

$$z_{\text{MAD}, t} = \frac{s_t - \text{median}(s)}{1.4826 \times \text{MAD}_t}$$

---

## 5. Five-Stage Signal Hierarchy

1. **Valid Normalized Observation**: Clean data, verified contract specs, normalized to pure gold.
2. **Statistically Unusual Relative Spread**: Statistical outlier exceeding rolling standard deviation bands.
3. **Candidate Research Signal**: Anomaly with verified point-in-time baseline.
4. **Eligible Signal**: Passed minimum volume, open interest, lifecycle buffer, and data checks.
5. **Historically Tested Result**: Simulated in walk-forward backtest accounting for fees, slippage, and next-bar execution.

---

## 6. Historical Backtest & Execution Realism

### Execution Timing
Signals generated at session close $t$ **cannot** be executed at price $t$. Simulated fills occur at session $t+1$ (next available price).

### Indian MCX Transaction Cost Model
* **Brokerage**: 0.02% (2 bps) of turnover
* **MCX Exchange Turnover Fee**: 0.0021%
* **SEBI Regulatory Fee**: 0.0001% (₹10/crore)
* **Commodities Transaction Tax (CTT)**: 0.01% on non-agricultural sell turnover
* **Stamp Duty**: 0.002% on buy turnover
* **GST**: 18% on (Brokerage + Exchange Fees)
* **Execution Slippage**: Configurable (default 1 tick per leg)

### Signal Study vs. Strategy Backtest
* **Signal Study Mode**: Analyzes empirical forward mean-reversion horizons (5, 10, 15 days) without execution assumptions.
* **Strategy Backtest Mode**: Enforces explicit entry ($|z| \ge 2.0$), exit ($|z| \le 0.5$ or sign flip), stop-loss ($|z| \ge 3.5$), and mandatory expiration close.
