# Post-Earnings Announcement Drift: Evidence of Increasing Market Efficiency (2015–2026)

An event study testing whether Post-Earnings Announcement Drift (PEAD) has weakened as algorithmic trading has grown, using 21,878 earnings events across 499 S&P 500 companies. The interaction term between earnings surprise and a post-2020 period dummy is negative and statistically significant (p = 0.039), confirming that markets have become measurably more efficient at pricing earnings information.

---

## Key Findings

### 1. PEAD has weakened post-2020, and the regression confirms it

The 3-day abnormal return for earnings beats fell from **+1.17%** (2015–2019) to **+0.90%** (2020–2026). The surprise-to-return correlation declined from **0.187 to 0.150**. An OLS regression with sector fixed effects confirms the weakening is statistically significant:

| Variable | Coefficient | p-value |
|---|:---:|:---:|
| Surprise magnitude | 0.0364 | <0.001 *** |
| Post-2020 dummy | −0.0022 | 0.009 ** |
| **Surprise × Post-2020** | **−0.0074** | **0.039 \*\*** |

*N = 21,878. R² = 0.028. HC3 robust SEs. Sector fixed effects included.*

### 2. The 40-day drift horizon shows the sharpest compression

| Horizon | 2015–2019 | 2020–2026 | Δ |
|---|:---:|:---:|:---:|
| 3 days | +1.17% | +0.90% | −0.27pp |
| 10 days | +1.32% | +1.02% | −0.30pp |
| **40 days** | **+1.83%** | **+0.97%** | **−0.86pp (−47%)** |
| 60 days | +1.78% | +1.29% | −0.49pp |

The 40-day signal has compressed by nearly half, consistent with faster medium-term repricing as algorithmic strategies absorb post-announcement information more efficiently.

### 3. Beat rate has risen — analyst sandbagging has intensified

The rate at which S&P 500 companies beat EPS estimates rose from **71.5%** to **77.2%**, while mean surprise widened from **7.8%** to **11.3%**. This suggests analysts are setting lower bars rather than improving forecast accuracy, independently diluting the drift signal quality.

### 4. PEAD still exists — the anomaly has weakened, not disappeared

The baseline surprise-return relationship remains highly significant (p < 0.001). The 40-day drift for beats is still +0.97% post-2020. Markets have moved toward efficiency, but have not reached it.

---

## Methodology

### Event Study Design

```
Earnings Surprise = (Actual EPS − Estimated EPS) / |Estimated EPS|
Winsorized at 1st / 99th percentile

Cumulative Abnormal Return (CAR) = Stock Return − SPY Return
Window: close(t-1) → close(t+h), where t = announcement date
Horizons h: 3, 5, 10, 20, 40, 60 trading days
```

### OLS Specification

```
CAR(3d) = β₀ + β₁(surprise) + β₂(post2020)
        + β₃(surprise × post2020) + Σβₙ(sector_FE) + ε
```

The interaction term β₃ directly measures whether the surprise-to-return relationship changed post-2020. Sector fixed effects control for cross-sector composition differences between the two periods.

### Benchmark

SPY (SPDR S&P 500 ETF Trust) is used as the market return benchmark. Abnormal return = stock return minus SPY return over the identical window.

---

## Project Structure

```
earnings-surprise-analysis/
├── data/
│   └── raw/
│       ├── earnings_full.csv          # 22,436 rows: EPS actuals + estimates, 2015+
│       ├── prices_full.csv            # 1,401,511 rows: daily adjusted close, 2015–2026
│       ├── earnings_batch1.csv        # Validation batch (50 tickers)
│       ├── prices_batch1.csv          # Validation batch prices
│       └── failed_tickers.csv         # 3 tickers with no yfinance data
├── analysis/
│   ├── test_fmp_data.py               # FMP API validation (deprecated, FMP not used)
│   ├── collect_batch1.py              # Batch 1 data collection (50 tickers)
│   ├── calculate_pead_batch1.py       # Batch 1 PEAD validation
│   ├── collect_full_sp500.py          # Full S&P 500 data collection pipeline
│   └── calculate_pead_full.py         # Full PEAD analysis (primary script)
├── output/
│   ├── pead_full_results.csv          # Event-level results (21,880 rows)
│   ├── pead_period_comparison.csv     # Pre vs post 2020 summary (2 rows)
│   ├── pead_drift_curves.csv          # CAR at each horizon × period × direction (36 rows)
│   └── tableau_ready_summary.csv      # Combined Tableau-ready flat file (50 rows)
├── docs/
│   └── executive_brief.md             # CFO/CIO-level summary of findings
└── README.md
```

---

## How to Run

**Requirements:** Python 3.9+ with `pandas`, `numpy`, `statsmodels`, `yfinance`, `requests`, `lxml`.

```bash
pip install pandas numpy statsmodels yfinance requests lxml
```

**Step 1: Collect full S&P 500 data** (~20–30 minutes)

```bash
python3 analysis/collect_full_sp500.py
```

Fetches earnings (EPS actuals + analyst estimates) and daily prices for all S&P 500 tickers via yfinance. Outputs `earnings_full.csv` and `prices_full.csv`.

**Step 2: Run the PEAD analysis** (~2–3 minutes)

```bash
python3 analysis/calculate_pead_full.py
```

Builds the price matrix, computes cumulative abnormal returns at six drift horizons for all 21,878 events, runs the OLS regression with sector fixed effects, and exports all three output files.

**Step 3: Validate on batch 1** (optional, ~30 seconds)

```bash
python3 analysis/calculate_pead_batch1.py
```

Runs the same methodology on the 50-ticker validation batch. Useful for confirming the pipeline before the full run.

---

## Data Sources

| Source | What | How |
|---|---|---|
| Yahoo Finance (yfinance) | EPS actuals, analyst estimates, daily prices | `Ticker.get_earnings_dates(limit=80)`, `yf.download()` |
| Wikipedia | S&P 500 constituent list, GICS sector | `pd.read_html()` |
| SPY ETF | Market benchmark returns | `yf.download('SPY')` |

**Note on data source selection:** Financial Modeling Prep (FMP) was evaluated as a primary earnings source but its free tier returned 402 errors for ~86% of S&P 500 tickers after a recent API restructuring. yfinance with `limit=80` provides analyst consensus estimates back to ~2002 with 0% null rate in the 2015–2026 study window, at no cost and no rate limits.

### Coverage

| Metric | Value |
|---|---|
| S&P 500 tickers attempted | 503 |
| Tickers with earnings data | 499 |
| Tickers with price data | 503 |
| Failed tickers | 3 (CRH, DXCM, HONA — no yfinance data) |
| Earnings events (2015–2026) | 22,436 |
| PEAD-ready events (both EPS fields) | 21,878 (98%) |
| Price rows (daily close) | 1,401,511 |
| Study window | 2015-01-01 → 2026-07-01 |

---

## Output Files for Tableau

Three flat files are exported for visualization:

| File | Rows | Description |
|---|:---:|---|
| `pead_full_results.csv` | 21,880 | One row per earnings event with surprise, CAR at all horizons, sector, period |
| `pead_period_comparison.csv` | 2 | Pre vs post 2020 summary metrics |
| `pead_drift_curves.csv` | 36 | Long format: CAR by period × surprise direction × horizon. Primary visualization source |
| `tableau_ready_summary.csv` | 50 | Combined flat file with `chart_type` filter column |

---

## Limitations

- **R² = 0.028.** Earnings surprise explains ~3% of 3-day return variance. The model is an anomaly detector, not a return predictor.
- **Correlation, not causation.** PEAD compression coincides with algorithmic trading growth but cannot be attributed to it without instrument variables or a natural experiment.
- **Survivorship bias.** Only current S&P 500 constituents are included; de-listed companies are underrepresented in the pre-2020 period.
- **Analyst estimate quality.** Yahoo Finance consensus estimates vary in coverage and timeliness by company. Institutional-grade data (Bloomberg, FactSet) would improve precision.
- **Beat rate confound.** Rising beat rates (71.5% → 77.2%) independently reduce the informativeness of positive surprises, making it difficult to fully isolate the algorithmic efficiency channel.

---

## Tools & Stack

| Tool | Purpose |
|---|---|
| Python 3.9 | Analysis runtime |
| pandas | Data loading, cleaning, pivoting, event study windows |
| NumPy | Vectorized CAR computation, winsorization |
| statsmodels | OLS regression with HC3 robust SEs |
| yfinance | Earnings data (EPS actuals + estimates) and daily prices |
| requests + lxml | Wikipedia S&P 500 constituent and sector data |

---

*Study period: January 2015 – July 2026. Data current as of July 2026.*
