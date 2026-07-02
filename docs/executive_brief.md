# Post-Earnings Announcement Drift in the Algorithmic Trading Era: Evidence of Increasing Market Efficiency

**Dylan Jeon** | Quantitative Finance Research | July 2026

---

## Overview

Post-Earnings Announcement Drift (PEAD) is one of the most well-documented anomalies in financial markets. First described by Ball and Brown (1968), PEAD refers to the tendency for stock prices to continue drifting in the direction of an earnings surprise for weeks or months after the announcement — positive surprises followed by positive drift, negative surprises by negative drift. In an efficient market, earnings information should be fully priced in immediately. PEAD implies it isn't.

This study tests whether PEAD has weakened in the post-2020 period, when algorithmic trading and real-time information processing have become increasingly dominant in U.S. equity markets. Using 21,878 earnings events across 499 S&P 500 companies from 2015 to 2026, we find robust evidence that it has.

---

## Key Finding 1: PEAD Still Exists, But Has Weakened Significantly Post-2020

The core PEAD signal — stocks that beat earnings estimates continue to outperform the market in the weeks that follow — is present in both periods, but measurably smaller after 2020.

The 3-day cumulative abnormal return (CAR) around earnings announcements for stocks that beat estimates fell from **+1.17%** (2015–2019) to **+0.90%** (2020–2026), a decline of 27 basis points. For misses, the negative drift deepened slightly: **−2.05%** to **−2.20%**, suggesting markets are also faster to reprice disappointments.

The surprise-return correlation — the relationship between how much a company beat or missed estimates and how much its stock moved — declined from **0.187 to 0.150** across the two periods. Both the magnitude and predictability of the drift have compressed.

---

## Key Finding 2: The 40-Day Horizon Shows the Sharpest Compression

The most striking compression appears at the 40-trading-day horizon — approximately two months post-announcement.

| Drift Horizon | 2015–2019 | 2020–2026 | Change |
|:---|:---:|:---:|:---:|
| 3 days | +1.17% | +0.90% | −0.27pp |
| 5 days | +1.21% | +0.95% | −0.25pp |
| 10 days | +1.32% | +1.02% | −0.30pp |
| 20 days | +1.62% | +1.24% | −0.37pp |
| **40 days** | **+1.83%** | **+0.97%** | **−0.86pp** |
| 60 days | +1.78% | +1.29% | −0.49pp |

*Cumulative abnormal return from close(t−1) to close(t+h), beats only. Benchmark: SPY.*

The 40-day decline of **−0.86 percentage points** represents a **47% compression** in the medium-term drift signal. The fact that compression is largest at intermediate horizons — rather than at the immediate 3-day window — is consistent with algorithms absorbing the initial announcement reaction quickly while the market converges on fair value faster over the following weeks.

The 60-day horizon shows partial reversion relative to 40 days, likely due to noise accumulation and sector-specific mean reversion at longer time horizons.

---

## Key Finding 3: Regression Confirms Statistical Significance Across 21,878 Events

To isolate the post-2020 weakening from concurrent structural changes — sector composition shifts, macroeconomic regime differences, and the post-COVID earnings volatility spike — we run an OLS regression with sector fixed effects:

```
CAR(3d) = β₀ + β₁(surprise) + β₂(post2020) + β₃(surprise × post2020) + sector FE + ε
```

The interaction term β₃ directly tests whether the surprise-to-return relationship changed post-2020.

| Variable | Coefficient | p-value | |
|---|:---:|:---:|:---:|
| Intercept | 0.0023 | 0.379 | |
| Surprise magnitude | 0.0364 | <0.001 | *** |
| Post-2020 dummy | −0.0022 | 0.009 | *** |
| **Surprise × Post-2020** | **−0.0074** | **0.039** | **\*\*** |
| Sector fixed effects (10) | — | — | |

*N = 21,878 events. R² = 0.028. HC3 heteroskedasticity-robust standard errors.*

The interaction term is **negative and significant at the 5% level** (p = 0.039), confirming that earnings surprises generated smaller abnormal returns post-2020 even after controlling for sector composition. Both the surprise-return sensitivity and the unconditional post-announcement return declined — the post-2020 dummy is also significant (p = 0.009), indicating that post-2020 earnings windows carry lower average drift regardless of surprise magnitude.

---

## Key Finding 4: Beat Rate Has Risen — Analyst Sandbagging Has Increased

A secondary but notable finding: the rate at which companies beat analyst EPS estimates rose from **71.5%** (2015–2019) to **77.2%** (2020–2026), while the mean earnings surprise widened from **7.8%** to **11.3%**.

This is not evidence that companies became more profitable relative to expectations — it is evidence of **analyst sandbagging**: analysts systematically lowering estimates to make beats easier. The pattern is well-documented in the behavioral finance literature and has intensified as companies and analysts have engaged in more active guidance management.

The implication for PEAD research: the average "beat" in the post-2020 period is a lower-quality signal. A 77% beat rate with a larger average surprise suggests the market has learned to discount small positive surprises, which would independently compress the drift signal even without algorithmic efficiency improvements.

---

## Enterprise Implications

**For quantitative long/short equity strategies:**
PEAD-based strategies that were reliably generating alpha at the 20–60 day horizon in the 2015–2019 period have seen that edge compress by roughly one-third to one-half post-2020. Strategies relying solely on earnings surprise direction without additional signal filtering face meaningfully lower expected returns.

**For active fundamental managers:**
The compression is largest at intermediate horizons (40 days), not at the immediate reaction window. This suggests the fastest-moving capital — algorithmic traders reacting to earnings prints in milliseconds — has become more efficient at the announcement, while slower fundamental capital is closing the medium-term gap faster than before. Active managers who historically generated returns by holding post-earnings beats for 1–2 months face a compressed opportunity.

**For risk models:**
The rising beat rate and wider average surprise magnitude mean earnings windows are generating more market noise than directional signal post-2020. Factor models that use earnings surprise as an input should be recalibrated to weight recent observations less heavily or apply beat-rate-adjusted surprise scoring.

**The anomaly is not dead:**
PEAD still exists. The 40-day drift for beats remains positive at +0.97%, and the regression confirms the baseline surprise-return relationship is still highly significant (p < 0.001). The finding is that the anomaly has weakened, not disappeared — consistent with a market that is becoming more efficient but has not reached the semi-strong form efficiency described by Fama (1970).

---

## Methodology Note

**Data:** 21,878 earnings events across 499 S&P 500 companies, 2015–2026. Earnings data (EPS actuals and analyst consensus estimates) sourced from Yahoo Finance via yfinance. Price data from Yahoo Finance adjusted for splits and dividends. Benchmark: SPY (SPDR S&P 500 ETF Trust).

**Abnormal return calculation:** Cumulative abnormal return = stock return − SPY return over the same window. Three-day window: close(t−1) to close(t+1). Drift windows: close(t−1) to close(t+h) for h ∈ {5, 10, 20, 40, 60} trading days.

**Winsorization:** Earnings surprise ratios winsorized at 1st and 99th percentile to limit distortion from near-zero EPS denominators.

**Honest limitations:**

- **R² = 0.028.** Earnings surprise explains approximately 3% of the variance in 3-day abnormal returns. The vast majority of stock price movement around earnings is explained by factors not captured here: guidance, revenue dynamics, margin trends, and market sentiment.

- **Correlation, not causation.** We observe that PEAD weakened concurrent with the growth of algorithmic trading. We cannot establish causation. Other explanations for post-2020 PEAD compression include increased retail participation, COVID-era earnings volatility distorting estimates, and broader sector rotation effects.

- **Analyst estimate quality varies.** Yahoo Finance consensus estimates aggregate analyst forecasts with varying coverage and timeliness. Institutional databases (Bloomberg, FactSet) would provide more precise consensus histories, particularly for small-cap constituents.

- **S&P 500 survivorship.** This study covers current S&P 500 constituents, introducing survivorship bias. Companies that were removed from the index due to poor performance are underrepresented in the 2015–2019 pre-period.

---

*Analysis conducted in Python (pandas, statsmodels, yfinance). Full methodology, code, and replication data available in the project repository.*
