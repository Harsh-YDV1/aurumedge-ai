# AurumEdge AI — Data Dictionary & Provenance Specification

## 1. Overview
AurumEdge AI enforces a strict multi-tier data pipeline that distinguishes between raw market ingest, validated canonical records, normalized fine-gold price equivalents, and relative-value analytical candidates.

At no point are Close and Settlement prices conflated, nor are synthetic observations mixed into historical performance datasets.

---

## 2. Ingestion & Raw Data Mapping

The ingestion engine maps varied exchange files (MCX Daily Bhavcopy formats, historical CSVs, and vendor exports) into a standardized internal representation:

| Canonical Field | Description | Expected Types | Acceptable Source Headers |
| :--- | :--- | :--- | :--- |
| `trading_date` | Date of market session | `Date` (YYYY-MM-DD) | `TIMESTAMP`, `Date`, `Trading Date`, `Trade_Date` |
| `symbol` | Gold contract variant | `String` (uppercase) | `SYMBOL`, `Instrument Name`, `Contract` |
| `expiry_date` | Contract expiration date | `Date` (YYYY-MM-DD) | `EXPIRY_DT`, `Expiry Date`, `Expiry` |
| `open_price` | Session opening trade price | `Float64` (> 0) | `OPEN`, `Open Price`, `Open` |
| `high_price` | Session high trade price | `Float64` (>= Low) | `HIGH`, `High Price`, `High` |
| `low_price` | Session low trade price | `Float64` (> 0) | `LOW`, `Low Price`, `Low` |
| `close_price` | Last traded price or session close | `Float64` (> 0) | `CLOSE`, `Close Price`, `LTP`, `Last Price` |
| `settle_price` | Official daily settlement price | `Float64` (> 0, optional) | `SETTLE_PR`, `Settle Price`, `Settlement Price` |
| `volume` | Traded volume (contracts) | `Int64` (>= 0) | `CONTRACTS`, `VOLUME`, `Traded Qty`, `Vol` |
| `open_interest` | Outstanding open contracts | `Int64` (>= 0) | `OPEN_INT`, `Open Interest`, `OI` |
| `turnover_lakhs` | Session turnover in ₹ Lakhs | `Float64` (>= 0) | `VAL_INLAKH`, `Turnover`, `Value Lakhs` |
| `provenance` | Data lineage indicator | `String` | `MCX_OFFICIAL_RAW`, `SYNTHETIC_DEMO` |

---

## 3. Contract Master Specifications (`config/contracts.csv`)

| Symbol | Contract Name | Trading Unit (Lot Size) | Quotation Unit | Purity (per 1,000) | Tick Size | Delivery Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GOLD** | Standard Gold (1kg) | 1,000 grams | 10 grams | 995.0 | ₹1.00 / 10g | Compulsory Delivery |
| **GOLDM** | Gold Mini | 100 grams | 10 grams | 995.0 | ₹1.00 / 10g | Compulsory Delivery |
| **GOLDTEN** | Gold Ten | 10 grams | 10 grams | 999.0 | ₹1.00 / 10g | Compulsory Delivery |
| **GOLDGUINEA** | Gold Guinea | 8 grams | 8 grams | 999.0 | ₹1.00 / 8g | Compulsory Delivery |
| **GOLDPETAL** | Gold Petal | 1 gram | 1 gram | 999.0 | ₹1.00 / 1g | Compulsory Delivery |

---

## 4. Normalization Fields

When converting quoted market prices to pure fine-gold equivalent price:

$$q_i = \frac{P_i}{g_i \times (u_i / 1000)}$$

* `raw_price` ($P_i$): Raw price quoted on the exchange.
* `quotation_unit_grams` ($g_i$): Gram weight basis of the quote.
* `purity_per_mille` ($u_i$): Fine gold fineness (e.g. 995 = 99.5%, 999 = 99.9%).
* `pure_gold_equivalent_price` ($q_i$): Derived price per single gram of 100.0% pure gold.
* `trading_unit_grams`: Used exclusively for lot-level physical delivery exposure, capital requirements, and transaction costs. It is never substituted for $g_i$.

---

## 5. Signal & Pair Analytical Fields

* `pair_id`: Unique identifier formatted as `SYMBOL_A-SYMBOL_B` (e.g., `GOLDM-GOLDPETAL`).
* `relative_spread` ($s_t$): Log price ratio $\ln(q_A / q_B)$.
* `rolling_mean` ($\mu_t$): Rolling arithmetic mean computed over past $W$ days strictly prior to observation bar $t$.
* `rolling_std` ($\sigma_t$): Rolling standard deviation over window $W$.
* `z_score` ($z_t$): Point-in-time normalized deviation $(s_t - \mu_t) / \sigma_t$.
* `mad_score`: Robust anomaly z-metric based on median absolute deviation.
* `signal_status`:
  - `NORMAL`: Spread within $[-Z_{\text{threshold}}, +Z_{\text{threshold}}]$.
  - `CANDIDATE_ANOMALY`: $|Z| > Z_{\text{threshold}}$, subject to eligibility filtering.
  - `ELIGIBLE_SIGNAL`: Statistically unusual and passed all liquidity, lifecycle, and quality filters.
  - `FILTERED_LIQUIDITY`: Disqualified due to low volume or open interest.
  - `FILTERED_EXPIRY`: Disqualified due to expiry proximity (< 3 days).
  - `FILTERED_HISTORY`: Insufficient rolling history (< minimum bars).
  - `FILTERED_INCOMPATIBLE`: Mismatched contract expiries.
