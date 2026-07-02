"""
PEAD (Post-Earnings Announcement Drift) calculation — batch 1 validation.

Computes 3-day abnormal returns around each earnings announcement using SPY
as the market benchmark, then compares pre-2020 (2015–2019) vs post-2020
(2020–2026) to test whether PEAD has weakened as algorithmic trading grew.

3-day window convention: close(t-1) → close(t+1) where t is the normalized
calendar date of the announcement. This symmetrically brackets the event
regardless of whether the announcement was pre- or post-market.
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from pathlib import Path
from datetime import date

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent.parent
EARN_PATH   = BASE_DIR / "data" / "raw" / "earnings_batch1.csv"
PRICE_PATH  = BASE_DIR / "data" / "raw" / "prices_batch1.csv"
OUTPUT_PATH = BASE_DIR / "output" / "pead_batch1_results.csv"

PRE_START  = "2015-01-01"
PRE_END    = "2019-12-31"
POST_START = "2020-01-01"
POST_END   = str(date.today())

# Drop events where |surprise_pct| > 200% — near-zero EPS denominator distorts ratio
SURPRISE_CAP = 2.0


# ── 1. Load data ──────────────────────────────────────────────────────────────

def load_earnings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["earnings_date"])
    # Strip intraday time / timezone so dates align with price index
    df["earnings_date"] = df["earnings_date"].dt.normalize()
    # Keep only announced quarters (actual EPS present) with a valid estimate
    df = df.dropna(subset=["eps_estimate", "eps_actual"]).copy()
    # Recompute surprise from raw EPS (transparent; ignore yfinance pre-computed %)
    df["surprise_pct"] = (
        (df["eps_actual"] - df["eps_estimate"]) / df["eps_estimate"].abs()
    )
    # Drop events with nonsense surprise ratios (near-zero denominator)
    df = df[df["surprise_pct"].abs() <= SURPRISE_CAP].copy()
    return df.reset_index(drop=True)


def load_prices(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def fetch_spy(start: str, end: str) -> pd.Series:
    print("Fetching SPY benchmark prices from yfinance...")
    # Fetch a few extra days before start so window calculations work at edges
    raw = yf.download("SPY", start="2014-12-15", end=end,
                      auto_adjust=True, progress=False)
    spy = raw["Close"].squeeze()
    spy.index = pd.to_datetime(spy.index).normalize()
    spy.name = "SPY"
    print("  SPY: {} trading days ({} → {})".format(
        len(spy), spy.index.min().date(), spy.index.max().date()))
    return spy


# ── 2. Build price matrix ─────────────────────────────────────────────────────

def build_price_matrix(prices_df: pd.DataFrame, spy: pd.Series) -> pd.DataFrame:
    pivot = prices_df.pivot_table(index="date", columns="ticker", values="close")
    pivot.index = pd.to_datetime(pivot.index).normalize()
    pivot = pivot.sort_index()
    pivot["SPY"] = spy.reindex(pivot.index)
    return pivot


# ── 3. 3-day abnormal return ──────────────────────────────────────────────────

def calc_3day_returns(
    ann_date: pd.Timestamp,
    ticker: str,
    pivot: pd.DataFrame,
) -> tuple:
    """
    Return (stock_ret, spy_ret, abnormal_ret) over [t-1, t+1] window.
    t is the first trading day in the price matrix on or after ann_date.
    """
    idx = pivot.index
    pos = int(idx.searchsorted(ann_date, side="left"))

    # Clamp: need at least one day before and one day after
    if pos < 1 or pos + 1 >= len(idx):
        return np.nan, np.nan, np.nan

    if ticker not in pivot.columns:
        return np.nan, np.nan, np.nan

    p_pre  = pivot.iat[pos - 1, pivot.columns.get_loc(ticker)]
    p_post = pivot.iat[pos + 1, pivot.columns.get_loc(ticker)]
    s_pre  = pivot.iat[pos - 1, pivot.columns.get_loc("SPY")]
    s_post = pivot.iat[pos + 1, pivot.columns.get_loc("SPY")]

    if any(pd.isna(v) or v <= 0 for v in [p_pre, p_post, s_pre, s_post]):
        return np.nan, np.nan, np.nan

    stock_ret = p_post / p_pre - 1
    spy_ret   = s_post / s_pre - 1
    ab_ret    = stock_ret - spy_ret
    return stock_ret, spy_ret, ab_ret


def add_event_returns(df: pd.DataFrame, pivot: pd.DataFrame) -> pd.DataFrame:
    print("Computing 3-day abnormal returns for {} events...".format(len(df)))
    rows = []
    for _, row in df.iterrows():
        sr, mr, ar = calc_3day_returns(row["earnings_date"], row["ticker"], pivot)
        rows.append((sr, mr, ar))

    ret_df = pd.DataFrame(rows, columns=["stock_return_3d", "spy_return_3d", "abnormal_return_3d"])
    result = pd.concat([df.reset_index(drop=True), ret_df], axis=1)
    n_total = len(result)
    result = result.dropna(subset=["abnormal_return_3d"]).reset_index(drop=True)
    print("  {} / {} events have valid 3-day return windows.".format(len(result), n_total))
    return result


# ── 4. Period labels ──────────────────────────────────────────────────────────

def add_period(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period"] = np.where(df["earnings_date"] <= PRE_END, "pre_2020", "post_2020")
    df["post2020"] = (df["period"] == "post_2020").astype(int)
    df["surprise_x_post2020"] = df["surprise_pct"] * df["post2020"]
    return df


# ── 5. Summary stats ──────────────────────────────────────────────────────────

def compute_period_stats(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for period, g in df.groupby("period"):
        beats   = g[g["surprise_pct"] > 0]
        misses  = g[g["surprise_pct"] < 0]
        corr    = g["surprise_pct"].corr(g["abnormal_return_3d"])

        records.append({
            "period":                   period,
            "n_events":                 len(g),
            "n_tickers":                g["ticker"].nunique(),
            "mean_surprise_pct":        round(g["surprise_pct"].mean() * 100, 2),
            "median_surprise_pct":      round(g["surprise_pct"].median() * 100, 2),
            "beat_rate_pct":            round((g["surprise_pct"] > 0).mean() * 100, 1),
            "mean_abnormal_ret_beats":  round(beats["abnormal_return_3d"].mean() * 100, 3),
            "mean_abnormal_ret_misses": round(misses["abnormal_return_3d"].mean() * 100, 3),
            "mean_abnormal_ret_all":    round(g["abnormal_return_3d"].mean() * 100, 3),
            "surprise_abnret_corr":     round(corr, 4),
        })

    return pd.DataFrame(records).set_index("period")


# ── 6. OLS regression ─────────────────────────────────────────────────────────

def run_ols(df: pd.DataFrame):
    """
    3d_abnormal_return ~ const + surprise_pct + post2020 + surprise_pct*post2020

    The interaction term tests whether the surprise→return relationship
    changed in the post-2020 period (negative = weaker PEAD post-2020).
    """
    X = sm.add_constant(df[["surprise_pct", "post2020", "surprise_x_post2020"]])
    y = df["abnormal_return_3d"]
    return sm.OLS(y, X).fit(cov_type="HC3")   # HC3: robust to heteroskedasticity


# ── 7. Print summary ──────────────────────────────────────────────────────────

def print_summary(stats: pd.DataFrame, ols_result, df: pd.DataFrame) -> None:
    W = 70
    print("\n" + "=" * W)
    print("  PEAD VALIDATION — BATCH 1 (50 S&P 500 tickers, 2015–2026)")
    print("=" * W)

    # Period comparison table
    pre  = stats.loc["pre_2020"]
    post = stats.loc["post_2020"]

    print("\n  PERIOD COMPARISON")
    print("  {:<34} {:>12} {:>12}".format("Metric", "2015–2019", "2020–2026"))
    print("  " + "-" * 60)

    def row(label, pre_val, post_val, fmt="{}", delta_sign=1):
        pv = fmt.format(pre_val)
        ov = fmt.format(post_val)
        if isinstance(pre_val, (int, float)) and isinstance(post_val, (int, float)):
            delta = post_val - pre_val
            arrow = "▲" if delta * delta_sign > 0 else "▼"
            delta_str = "  ({}{:.2f})".format(arrow, abs(delta))
        else:
            delta_str = ""
        print("  {:<34} {:>12} {:>12}{}".format(label, pv, ov, delta_str))

    row("Events (n)",               int(pre.n_events),               int(post.n_events),               "{}")
    row("Tickers (n)",              int(pre.n_tickers),              int(post.n_tickers),              "{}")
    row("Mean surprise (%)",        pre.mean_surprise_pct,           post.mean_surprise_pct,           "{:.2f}%")
    row("Median surprise (%)",      pre.median_surprise_pct,         post.median_surprise_pct,         "{:.2f}%")
    row("Beat rate (%)",            pre.beat_rate_pct,               post.beat_rate_pct,               "{:.1f}%")
    row("Avg 3d abn. ret — beats", pre.mean_abnormal_ret_beats,     post.mean_abnormal_ret_beats,     "{:.3f}%")
    row("Avg 3d abn. ret — misses",pre.mean_abnormal_ret_misses,    post.mean_abnormal_ret_misses,    "{:.3f}%")
    row("Avg 3d abn. ret — all",   pre.mean_abnormal_ret_all,       post.mean_abnormal_ret_all,       "{:.3f}%")
    row("Surprise-return corr (r)", pre.surprise_abnret_corr,        post.surprise_abnret_corr,        "{:.4f}")

    # OLS results
    p = ols_result.params
    pv = ols_result.pvalues
    ci = ols_result.conf_int()

    print("\n  OLS REGRESSION")
    print("  Dependent var:  3-day abnormal return")
    print("  Specification:  abn_ret ~ surprise_pct + post2020 + surprise_pct×post2020")
    print("  N = {}    R² = {:.4f}    Adj R² = {:.4f}    (HC3 robust SEs)".format(
        int(ols_result.nobs), ols_result.rsquared, ols_result.rsquared_adj))
    print()
    print("  {:<28} {:>10} {:>10} {:>10}".format("Variable", "Coef", "p-value", "Sig"))
    print("  " + "-" * 62)

    def stars(p_val):
        if p_val < 0.01: return "***"
        if p_val < 0.05: return "**"
        if p_val < 0.10: return "*"
        return ""

    var_labels = {
        "const":               "Intercept",
        "surprise_pct":        "Surprise magnitude",
        "post2020":            "Post-2020 dummy",
        "surprise_x_post2020": "Surprise × Post-2020",
    }
    for var, label in var_labels.items():
        print("  {:<28} {:>10.4f} {:>10.4f} {:>10}".format(
            label, p[var], pv[var], stars(pv[var])))

    # Key finding callout
    print("\n  KEY FINDINGS")
    print("  " + "-" * 60)

    delta_corr = post.surprise_abnret_corr - pre.surprise_abnret_corr
    interaction_coef = p["surprise_x_post2020"]
    interaction_sig  = pv["surprise_x_post2020"]
    beat_ret_change  = post.mean_abnormal_ret_beats - pre.mean_abnormal_ret_beats

    direction = "weaker" if interaction_coef < 0 else "stronger"
    sig_str   = "significant (p={:.3f})".format(interaction_sig) if interaction_sig < 0.10 \
                else "not statistically significant (p={:.3f})".format(interaction_sig)

    print("  • Surprise-return correlation: {:.4f} → {:.4f} ({:+.4f})".format(
        pre.surprise_abnret_corr, post.surprise_abnret_corr, delta_corr))
    print("  • Avg 3d abn. return on beats: {:+.3f}% → {:+.3f}% ({:+.3f}pp)".format(
        pre.mean_abnormal_ret_beats, post.mean_abnormal_ret_beats, beat_ret_change))
    print("  • Interaction term (Surprise×Post-2020): {:.4f} — {} {}".format(
        interaction_coef, direction, sig_str))
    print("  • Interpretation: PEAD appears {} post-2020 on this batch.".format(
        direction + (" (supports efficiency hypothesis)" if direction == "weaker" else
                     " (contradicts efficiency hypothesis)")))

    print("\n  NOTE: Batch 1 = 50 tickers. Scale to 500 for robust inference.")
    print("=" * W + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    earnings = load_earnings(EARN_PATH)
    prices   = load_prices(PRICE_PATH)
    spy      = fetch_spy(PRE_START, POST_END)

    print("Loaded {} earnings events across {} tickers.".format(
        len(earnings), earnings["ticker"].nunique()))

    pivot = build_price_matrix(prices, spy)

    events = add_event_returns(earnings, pivot)
    events = add_period(events)

    stats      = compute_period_stats(events)
    ols_result = run_ols(events)

    # Export: one row per event with all computed fields
    out_cols = [
        "ticker", "earnings_date", "period",
        "eps_estimate", "eps_actual", "surprise_pct",
        "stock_return_3d", "spy_return_3d", "abnormal_return_3d",
        "post2020",
    ]
    events[out_cols].to_csv(OUTPUT_PATH, index=False)
    print("Results exported → {} ({} rows)".format(OUTPUT_PATH, len(events)))

    print_summary(stats, ols_result, events)


if __name__ == "__main__":
    main()
