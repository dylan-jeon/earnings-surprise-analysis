"""
FMP earnings surprise data validation test.
Pulls historical EPS actuals + analyst estimates for 5 sample tickers
and validates date range, estimate quality, and coverage.
"""

import sys
import requests
import pandas as pd

API_KEY  = "oehZWoU7rOhgf9IYQjaAvh1MqHWMPBW4"
BASE_URL = "https://financialmodelingprep.com/stable"
TICKERS  = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]


def fetch_earnings_surprises(ticker: str) -> pd.DataFrame:
    url = f"{BASE_URL}/earnings?symbol={ticker}&apikey={API_KEY}"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and "Error Message" in data:
        raise ValueError(f"{ticker}: {data['Error Message']}")
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.rename(columns={"epsActual": "actualEarningResult", "epsEstimated": "estimatedEarning"})
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"])
    df["actualEarningResult"] = pd.to_numeric(df["actualEarningResult"], errors="coerce")
    df["estimatedEarning"]    = pd.to_numeric(df["estimatedEarning"],    errors="coerce")
    df["surprise_pct"] = (
        (df["actualEarningResult"] - df["estimatedEarning"])
        / df["estimatedEarning"].abs()
        * 100
    )
    return df.sort_values("date").reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    W = 65
    print(f"\n{'=' * W}")
    print("  FMP EARNINGS SURPRISE — DATA VALIDATION")
    print(f"{'=' * W}")

    # ── Per-ticker summary ────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  PER-TICKER COVERAGE")
    print(f"{'─' * W}")
    print(f"  {'Ticker':<6}  {'Rows':>4}  {'Earliest':>12}  {'Latest':>12}  "
          f"{'Est null%':>9}  {'Act null%':>9}  {'Pre-2020':>8}")
    print(f"  {'─'*6}  {'─'*4}  {'─'*12}  {'─'*12}  {'─'*9}  {'─'*9}  {'─'*8}")

    for ticker, g in df.groupby("ticker"):
        est_null = g["estimatedEarning"].isna().mean() * 100
        act_null = g["actualEarningResult"].isna().mean() * 100
        pre2020  = (g["date"] < "2020-01-01").sum()
        print(
            f"  {ticker:<6}  {len(g):>4}  "
            f"{str(g['date'].min().date()):>12}  "
            f"{str(g['date'].max().date()):>12}  "
            f"{est_null:>8.1f}%  {act_null:>8.1f}%  {pre2020:>8}"
        )

    # ── Aggregate stats ───────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  AGGREGATE STATS")
    print(f"{'─' * W}")

    total        = len(df)
    est_null_tot = df["estimatedEarning"].isna().mean() * 100
    act_null_tot = df["actualEarningResult"].isna().mean() * 100
    pre2015      = (df["date"] < "2015-01-01").sum()
    in_window    = df[df["date"].between("2015-01-01", "2019-12-31")]
    post2020     = df[df["date"] >= "2020-01-01"]

    print(f"  Total rows (all tickers):    {total}")
    print(f"  Estimate null rate:          {est_null_tot:.1f}%")
    print(f"  Actual null rate:            {act_null_tot:.1f}%")
    print(f"  Rows before 2015:            {pre2015}")
    print(f"  Rows 2015–2019 (pre-period): {len(in_window)}")
    print(f"  Rows 2020–present:           {len(post2020)}")

    # ── Surprise distribution ─────────────────────────────────────────────────
    clean = df.dropna(subset=["surprise_pct"])
    # Cap display at ±200% to avoid distortion from near-zero estimate outliers
    capped = clean[clean["surprise_pct"].abs() <= 200]["surprise_pct"]

    print(f"\n{'─' * W}")
    print("  SURPRISE % DISTRIBUTION  (capped at ±200% for display)")
    print(f"{'─' * W}")
    print(f"  Rows with valid surprise:    {len(clean)} / {total}")
    print(f"  Mean surprise:               {capped.mean():+.1f}%")
    print(f"  Median surprise:             {capped.median():+.1f}%")
    print(f"  Std dev:                     {capped.std():.1f}%")
    print(f"  Beat estimates (>0%):        {(capped > 0).sum()} "
          f"({(capped > 0).mean()*100:.0f}%)")
    print(f"  Missed estimates (<0%):      {(capped < 0).sum()} "
          f"({(capped < 0).mean()*100:.0f}%)")

    # ── Sample rows ───────────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  SAMPLE ROWS — AAPL (most recent 6 quarters)")
    print(f"{'─' * W}")
    aapl = df[df["ticker"] == "AAPL"].tail(6)[
        ["date", "actualEarningResult", "estimatedEarning", "surprise_pct"]
    ].copy()
    aapl.columns = ["date", "actual_eps", "est_eps", "surprise_pct"]
    aapl["date"] = aapl["date"].dt.date
    print(aapl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * W}")
    print("  VERDICT")
    print(f"{'─' * W}")
    issues = []
    if est_null_tot > 10:
        issues.append(f"High estimate null rate ({est_null_tot:.1f}%)")
    if len(in_window) < 15:
        issues.append("Insufficient 2015–2019 coverage (<15 rows across 5 tickers)")
    if pre2015 == 0:
        issues.append("No data before 2015 — check endpoint depth")

    if issues:
        print("  CONCERNS:")
        for i in issues:
            print(f"    • {i}")
    else:
        print("  PASS — data looks suitable for PEAD pipeline.")
        print(f"  Avg {total // len(TICKERS)} quarters/ticker, "
              f"{est_null_tot:.1f}% estimate null rate, "
              f"good 2015+ coverage.")

    print(f"\n{'=' * W}\n")


def main() -> None:
    frames = []
    for ticker in TICKERS:
        print(f"Fetching {ticker}...", end=" ", flush=True)
        try:
            df = fetch_earnings_surprises(ticker)
            print(f"{len(df)} rows")
            frames.append(df)
        except Exception as e:
            print(f"ERROR — {e}")

    if not frames:
        print("No data retrieved. Check API key and connectivity.", file=sys.stderr)
        sys.exit(1)

    all_df = pd.concat(frames, ignore_index=True)

    out = "data/raw/fmp_test_earnings.csv"
    all_df.to_csv(out, index=False)
    print(f"\nRaw data saved → {out}")

    validate(all_df)


if __name__ == "__main__":
    main()
