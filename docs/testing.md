# AurumEdge AI — Verification & Testing Protocol

## 1. Testing Philosophy
The testing framework for AurumEdge AI verifies the core quantitative invariant:

> **"Mathematical correctness, point-in-time integrity, and deterministic execution must precede any trading evaluation."**

Automated tests cover all pipeline stages:
1. File loading and malformed input resilience
2. Purity and quotation unit arithmetic against manual hand-calculations
3. Look-ahead bias prevention in rolling calculations
4. Lifecycle edge cases (negative days-to-expiry, tender buffer violations)
5. Execution timing ($t+1$ fill checks) and cost model accuracy

---

## 2. Test Suites

### `tests/test_ingest.py`
* Standard MCX bhavcopy parsing.
* Alternative column header detection (e.g. `Traded Qty` vs `CONTRACTS`, `Date` vs `TIMESTAMP`).
* Malformed date string handling.
* Empty file and missing required header rejection.
* Provenance metadata stamping.

### `tests/test_validation.py`
* Non-positive price detection (0 or negative prices rejected).
* High < Low price contradiction detection.
* Duplicate `(trading_date, symbol, expiry_date)` detection.
* Handling of missing volume or open interest.
* Production of structured validation reports (accepted, rejected, warnings).

### `tests/test_normalization.py`
* Hand-calculated verification:
  - GOLDM @ ₹72,500/10g (995 fineness) $\rightarrow$ ₹7,286.432 / pure gram.
  - GOLDPETAL @ ₹7,295/1g (999 fineness) $\rightarrow$ ₹7,302.302 / pure gram.
* Proper separation of Quotation Unit (price denominator) and Trading Unit (lot size).
* Handling of unknown symbols or corrupted specification lookups.

### `tests/test_matching.py`
* Strict same-expiry pairing enforcement.
* Rejection of pairs with disparate expiration dates.
* Handling of calendar gaps (holidays, missing sessions).
* Insufficient overlapping history rejection.

### `tests/test_signals.py`
* Look-ahead bias verification: ensuring signal at bar $t$ depends strictly on $[t-W, t-1]$.
* Controlled synthetic anomaly test (injected 3-sigma dislocation triggers candidate alert).
* Zero variance handling (flat price spread gracefully results in $z = 0$).
* Filtering of candidates when volume < threshold, open interest < threshold, or days-to-expiry < buffer.
* Deterministic explanation generation.

### `tests/test_backtest.py`
* Chronological order invariant: trades cannot be executed prior to signal bar.
* Next-bar execution rule ($t+1$ execution, never executing at signal bar close $t$).
* Cost deduction verification: net PnL is strictly lower than gross PnL when fees are non-zero.
* Correct handling of forced contract expiry exits.
* Reproducibility under identical inputs.

---

## 3. Execution Commands

```bash
# Run all automated tests
.venv/Scripts/pytest tests/ -v

# Run with test coverage
.venv/Scripts/pytest --cov=src tests/

# Run static analysis and linting
.venv/Scripts/ruff check src/ tests/
```
