"""
Full PEAD analysis — complete S&P 500 dataset (2015–2026).

Tests whether PEAD has weakened post-2020 as algorithmic trading grew.

Drift window convention:
  car_3d  = [t-1, t+1]  — 2 trading-day intervals (matches batch 1)
  car_Xd  = [t-1, t+X]  — X trading days post-announcement, from pre-announcement close
  where t = first trading day on or after the earnings announcement date.

Outputs:
  output/pead_full_results.csv      — one row per event
  output/pead_period_comparison.csv — pre vs post 2020 summary
  output/pead_drift_curves.csv      — CAR at each horizon by period × beat/miss
"""

import sys
import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from pathlib import Path
from datetime import date
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent.parent
EARN_PATH  = BASE_DIR / "data" / "raw" / "earnings_full.csv"
PRICE_PATH = BASE_DIR / "data" / "raw" / "prices_full.csv"
EVENTS_OUT = BASE_DIR / "output" / "pead_full_results.csv"
PERIOD_OUT = BASE_DIR / "output" / "pead_period_comparison.csv"
DRIFT_OUT  = BASE_DIR / "output" / "pead_drift_curves.csv"

PRE_END    = "2019-12-31"
POST_START = "2020-01-01"
STUDY_END  = str(date.today())

WINSOR_LOW  = 0.01
WINSOR_HIGH = 0.99

# Drift horizons: {label: offset_from_pre_pos}
# offset_from_pre_pos = number of positions forward from pos-1 in the trading day index
# car_3d: pos-1 → pos+1  (offset 2)
# car_Xd: pos-1 → pos+X  (offset X+1)
HORIZONS: Dict[str, int] = {
    "car_3d":  2,
    "car_5d":  6,
    "car_10d": 11,
    "car_20d": 21,
    "car_40d": 41,
    "car_60d": 61,
}


# ── 1. Sector map ─────────────────────────────────────────────────────────────

def get_sector_map() -> Dict[str, str]:
    print("Fetching GICS sector data from Wikipedia...")
    headers = {"User-Agent": "Mozilla/5.0 (research project; educational use)"}
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=headers, timeout=15,
    )
    resp.raise_for_status()
    df = pd.read_html(resp.text)[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    sector_map = dict(zip(df["Symbol"], df["GICS Sector"]))
    print("  {} tickers mapped to {} sectors.".format(
        len(sector_map), df["GICS Sector"].nunique()))
    return sector_map


# ── 2. Load and clean earnings ────────────────────────────────────────────────

def load_earnings(path: Path, sector_map: Dict[str, str]) -> pd.DataFrame:
    print("Loading earnings data...")
    df = pd.read_csv(path, parse_dates=["earnings_date"])
    df["earnings_date"] = df["earnings_date"].dt.normalize()

    # Keep only announced quarters with both fields
    df = df.dropna(subset=["eps_estimate", "eps_actual"]).copy()

    # Recompute surprise from raw EPS (transparent, ignore pre-computed column)
    df["surprise_pct"] = (
        (df["eps_actual"] - df["eps_estimate"]) / df["eps_estimate"].abs()
    )

    # Drop division-by-near-zero artefacts before winsorizing
    df = df[np.isfinite(df["surprise_pct"])].copy()

    # Winsorize at 1st / 99th percentile
    lo = df["surprise_pct"].quantile(WINSOR_LOW)
    hi = df["surprise_pct"].quantile(WINSOR_HIGH)
    df["surprise_pct"] = df["surprise_pct"].clip(lo, hi)

    # Sector
    df["sector"] = df["ticker"].map(sector_map).fillna("Unknown")

    # Period
    df["period"]  = np.where(df["earnings_date"] <= PRE_END, "pre_2020", "post_2020")
    df["post2020"] = (df["period"] == "post_2020").astype(int)

    print("  {:,} events loaded ({:,} pre-2020, {:,} post-2020) across {} tickers.".format(
        len(df),
        (df["period"] == "pre_2020").sum(),
        (df["period"] == "post_2020").sum(),
        df["ticker"].nunique(),
    ))
    return df.reset_index(drop=True)


# ── 3. Price matrix + SPY ────────────────────────────────────────────────────

def build_price_matrix(price_path: Path) -> pd.DataFrame:
    print("Building price matrix...")
    prices = pd.read_csv(price_path, parse_dates=["date"])
    pivot = prices.pivot_table(index="date", columns="ticker", values="close")
    pivot.index = pd.to_datetime(pivot.index).normalize()
    pivot = pivot.sort_index()

    print("  Fetching SPY benchmark...")
    spy_raw = yf.download("SPY", start="2014-12-15", end=STUDY_END,
                          auto_adjust=True, progress=False)
    spy = spy_raw["Close"].squeeze()
    spy.index = pd.to_datetime(spy.index).normalize()
    pivot["SPY"] = spy.reindex(pivot.index)

    print("  Price matrix: {} trading days × {} columns (incl. SPY).".format(
        len(pivot), len(pivot.columns)))
    return pivot


# ── 4. CAR computation ────────────────────────────────────────────────────────

def compute_all_cars(events: pd.DataFrame, pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised-ish event study: for each row in events compute cumulative
    abnormal returns at every HORIZONS entry.
    """
    print("Computing CAR at {} drift horizons for {:,} events...".format(
        len(HORIZONS), len(events)))

    idx       = pivot.index.values                 # numpy array of Timestamps
    col_index = {c: i for i, c in enumerate(pivot.columns)}
    mat       = pivot.values                        # (dates × tickers) numpy float array
    spy_col   = col_index.get("SPY")

    if spy_col is None:
        raise ValueError("SPY not found in price matrix.")

    horizon_labels  = list(HORIZONS.keys())
    horizon_offsets = list(HORIZONS.values())
    n_horizons      = len(horizon_labels)

    results = np.full((len(events), n_horizons), np.nan)

    for i, (_, row) in enumerate(events.iterrows()):
        ticker  = row["ticker"]
        ann_ts  = row["earnings_date"]
        stk_col = col_index.get(ticker)
        if stk_col is None:
            continue

        # Find first trading day on or after announcement date
        pos = int(np.searchsorted(idx, np.datetime64(ann_ts), side="left"))
        pre_pos = pos - 1
        if pre_pos < 0:
            continue

        p_pre = mat[pre_pos, stk_col]
        s_pre = mat[pre_pos, spy_col]
        if np.isnan(p_pre) or p_pre <= 0 or np.isnan(s_pre) or s_pre <= 0:
            continue

        for j, offset in enumerate(horizon_offsets):
            post_pos = pre_pos + offset
            if post_pos >= len(idx):
                continue
            p_post = mat[post_pos, stk_col]
            s_post = mat[post_pos, spy_col]
            if np.isnan(p_post) or p_post <= 0 or np.isnan(s_post) or s_post <= 0:
                continue
            results[i, j] = (p_post / p_pre - 1) - (s_post / s_pre - 1)

        if (i + 1) % 5000 == 0:
            print("  {}/{} events processed...".format(i + 1, len(events)))
            sys.stdout.flush()

    car_df = pd.DataFrame(results, columns=horizon_labels, index=events.index)
    out    = pd.concat([events, car_df], axis=1)

    valid_3d = out["car_3d"].notna().sum()
    print("  Done. {}/{} events have valid car_3d.".format(valid_3d, len(out)))
    return out


# ── 5. Period summary stats ───────────────────────────────────────────────────

def compute_period_stats(df: pd.DataFrame) -> pd.DataFrame:
    core = df.dropna(subset=["car_3d", "surprise_pct"])
    records = []

    for period, g in core.groupby("period"):
        beats  = g[g["surprise_pct"] > 0]
        misses = g[g["surprise_pct"] < 0]
        corr   = g["surprise_pct"].corr(g["car_3d"])

        rec = {
            "period":                   period,
            "n_events":                 len(g),
            "n_tickers":                g["ticker"].nunique(),
            "mean_surprise_pct":        round(g["surprise_pct"].mean() * 100, 3),
            "median_surprise_pct":      round(g["surprise_pct"].median() * 100, 3),
            "beat_rate_pct":            round((g["surprise_pct"] > 0).mean() * 100, 1),
            "mean_car3d_all":           round(g["car_3d"].mean() * 100, 4),
            "mean_car3d_beats":         round(beats["car_3d"].mean() * 100, 4),
            "mean_car3d_misses":        round(misses["car_3d"].mean() * 100, 4),
            "surprise_return_corr":     round(corr, 4),
        }
        # Mean CAR at each drift horizon (beats only — this is PEAD signal)
        for h in HORIZONS:
            rec["beats_" + h] = round(beats[h].mean() * 100, 4) if h in g.columns else np.nan
            rec["misses_" + h] = round(misses[h].mean() * 100, 4) if h in g.columns else np.nan
        records.append(rec)

    return pd.DataFrame(records).set_index("period")


# ── 6. Drift curves (long format for Tableau) ─────────────────────────────────

def build_drift_curves(df: pd.DataFrame) -> pd.DataFrame:
    core = df.dropna(subset=["car_3d", "surprise_pct"])
    core = core.copy()
    core["surprise_direction"] = np.where(
        core["surprise_pct"] > 0, "beats",
        np.where(core["surprise_pct"] < 0, "misses", "inline")
    )

    rows = []
    for period, pg in core.groupby("period"):
        for direction, dg in pg.groupby("surprise_direction"):
            for label, _ in HORIZONS.items():
                if label not in dg.columns:
                    continue
                vals = dg[label].dropna()
                if len(vals) == 0:
                    continue
                rows.append({
                    "period":             period,
                    "surprise_direction": direction,
                    "horizon":            label,
                    "n_events":           len(vals),
                    "mean_car":           round(vals.mean() * 100, 4),
                    "median_car":         round(vals.median() * 100, 4),
                    "std_car":            round(vals.std() * 100, 4),
                    "stderr_car":         round(vals.sem() * 100, 4),
                })

    return pd.DataFrame(rows)


# ── 7. OLS with sector fixed effects ─────────────────────────────────────────

def run_ols(df: pd.DataFrame):
    """
    car_3d ~ surprise_pct + post2020 + surprise_pct×post2020 + sector_FE
    Reference sector: Communication Services (alphabetically first, dropped).
    HC3 heteroskedasticity-robust SEs.
    """
    core = df.dropna(subset=["car_3d", "surprise_pct", "sector"]).copy()
    core["surprise_x_post2020"] = core["surprise_pct"] * core["post2020"]

    sector_dummies = pd.get_dummies(core["sector"], prefix="sec", drop_first=True)
    X = pd.concat([
        core[["surprise_pct", "post2020", "surprise_x_post2020"]],
        sector_dummies,
    ], axis=1).astype(float)
    X = sm.add_constant(X)
    y = core["car_3d"].astype(float)

    return sm.OLS(y, X).fit(cov_type="HC3"), len(core)


# ── 8. Print summary ──────────────────────────────────────────────────────────

def print_summary(
    stats: pd.DataFrame,
    ols_res,
    n_ols: int,
    drift_df: pd.DataFrame,
) -> None:
    W = 72
    print("\n" + "=" * W)
    print("  PEAD FULL ANALYSIS — S&P 500 (2015–2026)")
    print("=" * W)

    pre  = stats.loc["pre_2020"]
    post = stats.loc["post_2020"]

    def fmt_row(label, pv, ov, fmt="{:.3f}", unit=""):
        ps, os_ = fmt.format(pv) + unit, fmt.format(ov) + unit
        if isinstance(pv, float) and isinstance(ov, float):
            delta = ov - pv
            arrow = "▲" if delta > 0 else "▼"
            d = "  ({}{:.3f}{})".format(arrow, abs(delta), unit)
        else:
            d = ""
        print("  {:<38} {:>12} {:>12}{}".format(label, ps, os_, d))

    print("\n  PERIOD COMPARISON")
    print("  {:<38} {:>12} {:>12}".format("Metric", "2015-2019", "2020-2026"))
    print("  " + "-" * 66)
    fmt_row("Events (n)",              float(pre.n_events),          float(post.n_events),        "{:.0f}")
    fmt_row("Tickers (n)",             float(pre.n_tickers),         float(post.n_tickers),       "{:.0f}")
    fmt_row("Mean surprise (%)",       pre.mean_surprise_pct,        post.mean_surprise_pct,      "{:.3f}", "%")
    fmt_row("Median surprise (%)",     pre.median_surprise_pct,      post.median_surprise_pct,    "{:.3f}", "%")
    fmt_row("Beat rate",               pre.beat_rate_pct,            post.beat_rate_pct,          "{:.1f}", "%")
    fmt_row("Mean CAR 3d — beats",     pre.mean_car3d_beats,         post.mean_car3d_beats,       "{:.3f}", "%")
    fmt_row("Mean CAR 3d — misses",    pre.mean_car3d_misses,        post.mean_car3d_misses,      "{:.3f}", "%")
    fmt_row("Mean CAR 3d — all",       pre.mean_car3d_all,           post.mean_car3d_all,         "{:.3f}", "%")
    fmt_row("Surprise-return corr",    pre.surprise_return_corr,     post.surprise_return_corr,   "{:.4f}")

    # Drift horizon table (beats only)
    print("\n  PEAD DRIFT — BEATS (mean cumulative abnormal return, % from close t-1)")
    print("  {:<10} {:>12} {:>12} {:>12}".format("Horizon", "2015-2019", "2020-2026", "Δ (pp)"))
    print("  " + "-" * 50)
    for h in HORIZONS:
        pv = getattr(pre, "beats_" + h, np.nan)
        ov = getattr(post, "beats_" + h, np.nan)
        delta = ov - pv if not (np.isnan(pv) or np.isnan(ov)) else np.nan
        d_str = "{:+.3f}".format(delta) if not np.isnan(delta) else "N/A"
        print("  {:<10} {:>11.3f}% {:>11.3f}% {:>12}".format(h, pv, ov, d_str))

    print("\n  PEAD DRIFT — MISSES (mean cumulative abnormal return, %)")
    print("  {:<10} {:>12} {:>12} {:>12}".format("Horizon", "2015-2019", "2020-2026", "Δ (pp)"))
    print("  " + "-" * 50)
    for h in HORIZONS:
        pv = getattr(pre, "misses_" + h, np.nan)
        ov = getattr(post, "misses_" + h, np.nan)
        delta = ov - pv if not (np.isnan(pv) or np.isnan(ov)) else np.nan
        d_str = "{:+.3f}".format(delta) if not np.isnan(delta) else "N/A"
        print("  {:<10} {:>11.3f}% {:>11.3f}% {:>12}".format(h, pv, ov, d_str))

    # OLS
    p   = ols_res.params
    pv  = ols_res.pvalues
    def stars(x):
        return "***" if x < 0.01 else ("**" if x < 0.05 else ("*" if x < 0.10 else ""))

    print("\n  OLS REGRESSION  (HC3 robust SEs)")
    print("  Dep. var: car_3d  |  N = {:,}  |  R² = {:.4f}  |  Adj R² = {:.4f}".format(
        n_ols, ols_res.rsquared, ols_res.rsquared_adj))
    print("  Spec: car_3d ~ surprise + post2020 + surprise×post2020 + sector_FE\n")
    print("  {:<30} {:>10} {:>10} {:>6}".format("Variable", "Coef", "p-value", "Sig"))
    print("  " + "-" * 60)

    key_vars = {
        "const":               "Intercept",
        "surprise_pct":        "Surprise magnitude",
        "post2020":            "Post-2020 dummy",
        "surprise_x_post2020": "Surprise × Post-2020  ←",
    }
    for var, label in key_vars.items():
        if var in p:
            print("  {:<30} {:>10.4f} {:>10.4f} {:>6}".format(
                label, p[var], pv[var], stars(pv[var])))

    sector_params = [(v, label) for v, label in zip(p.index, p.index)
                     if v.startswith("sec_")]
    if sector_params:
        print("  {:<30} {:>10}   {:>10}".format("Sector FEs", "({})" .format(
            len(sector_params)), "omitted"))

    # Key finding
    int_coef = p.get("surprise_x_post2020", np.nan)
    int_pval = pv.get("surprise_x_post2020", np.nan)
    corr_chg = post.surprise_return_corr - pre.surprise_return_corr
    beat_chg = post.mean_car3d_beats - pre.mean_car3d_beats
    drift60_chg = (getattr(post, "beats_car_60d", np.nan)
                   - getattr(pre,  "beats_car_60d", np.nan))

    print("\n  KEY FINDINGS")
    print("  " + "-" * 66)
    print("  • Surprise-return corr:       {:.4f} → {:.4f}  ({:+.4f})".format(
        pre.surprise_return_corr, post.surprise_return_corr, corr_chg))
    print("  • Mean 3d CAR on beats:       {:+.3f}% → {:+.3f}%  ({:+.3f}pp)".format(
        pre.mean_car3d_beats, post.mean_car3d_beats, beat_chg))
    if not np.isnan(drift60_chg):
        print("  • 60d drift (beats):          {:+.3f}% → {:+.3f}%  ({:+.3f}pp)".format(
            getattr(pre, "beats_car_60d", np.nan),
            getattr(post, "beats_car_60d", np.nan),
            drift60_chg))
    print("  • Interaction coef:           {:.4f}  p={:.4f}  {}".format(
        int_coef, int_pval, stars(int_pval)))

    direction = "weaker" if int_coef < 0 else "stronger"
    sig_str   = ("statistically significant" if int_pval < 0.05
                 else "not statistically significant")
    print("  • PEAD is {} post-2020 — {} at α=0.05.".format(direction, sig_str))
    print("\n" + "=" * W + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    sector_map = get_sector_map()
    earnings   = load_earnings(EARN_PATH, sector_map)
    pivot      = build_price_matrix(PRICE_PATH)

    events     = compute_all_cars(earnings, pivot)

    stats      = compute_period_stats(events)
    drift_df   = build_drift_curves(events)
    ols_res, n = run_ols(events)

    # Export
    BASE_DIR.joinpath("output").mkdir(exist_ok=True)

    out_cols = (
        ["ticker", "sector", "earnings_date", "period", "post2020",
         "eps_estimate", "eps_actual", "surprise_pct"]
        + list(HORIZONS.keys())
    )
    events[[c for c in out_cols if c in events.columns]].to_csv(EVENTS_OUT, index=False)

    stats.reset_index().to_csv(PERIOD_OUT, index=False)
    drift_df.to_csv(DRIFT_OUT, index=False)

    print("Saved: {}  ({:,} rows)".format(EVENTS_OUT, len(events)))
    print("Saved: {}  ({} rows)".format(PERIOD_OUT, len(stats)))
    print("Saved: {}  ({} rows)".format(DRIFT_OUT, len(drift_df)))

    print_summary(stats, ols_res, n, drift_df)


if __name__ == "__main__":
    main()
