"""
Data collection pipeline — batch 1 (first 50 S&P 500 tickers).

Uses yfinance for both earnings (EPS estimates + actuals) and daily prices.
No API key required.

Outputs:
  data/raw/earnings_batch1.csv  — quarterly EPS actual + estimate, 2015+
  data/raw/prices_batch1.csv    — daily adjusted close prices, 2015–today
"""

import time
import warnings
import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import date
from typing import Optional, Tuple, List

warnings.filterwarnings("ignore")

START_DATE = "2015-01-01"
END_DATE   = str(date.today())
BATCH_SIZE = 50
DELAY_SECS = 0.5    # polite delay between yfinance calls
EPS_LIMIT  = 80     # ~20 years of quarters; safely covers 2015+

BASE_DIR   = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw"


# ── 1. S&P 500 ticker list ────────────────────────────────────────────────────

def get_sp500_tickers() -> List[str]:
    print("Fetching S&P 500 ticker list from Wikipedia...")
    headers = {"User-Agent": "Mozilla/5.0 (research project; educational use)"}
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=headers, timeout=15,
    )
    resp.raise_for_status()
    tickers = (
        pd.read_html(resp.text)[0]["Symbol"]
        .str.replace(".", "-", regex=False)   # BRK.B → BRK-B for yfinance
        .tolist()
    )
    print("  {} tickers found.".format(len(tickers)))
    return tickers


# ── 2. Earnings fetch (yfinance) ──────────────────────────────────────────────

def fetch_one_earnings(ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        raw = yf.Ticker(ticker).get_earnings_dates(limit=EPS_LIMIT)
    except Exception as e:
        return None, str(e)

    if raw is None or raw.empty:
        return None, "empty response"

    df = raw.copy()
    df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
    df.index.name = "earnings_date"
    df = df.reset_index()
    df = df.rename(columns={
        "EPS Estimate": "eps_estimate",
        "Reported EPS": "eps_actual",
        "Surprise(%)":  "surprise_pct",
    })
    df["ticker"] = ticker
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df = df[df["earnings_date"] >= START_DATE].copy()
    df = df[["ticker", "earnings_date", "eps_estimate", "eps_actual", "surprise_pct"]]
    df = df.sort_values("earnings_date").reset_index(drop=True)
    return df, None


def collect_earnings(tickers: List[str]) -> Tuple[pd.DataFrame, list, list]:
    print("\nCollecting yfinance earnings for {} tickers...".format(len(tickers)))

    frames, failed, warned = [], [], []

    for i, ticker in enumerate(tickers, 1):
        df, err = fetch_one_earnings(ticker)
        if err:
            print("  [{:>2}/{}] {:<6}  FAILED — {}".format(i, len(tickers), ticker, err))
            failed.append((ticker, err))
        elif df is None or df.empty:
            print("  [{:>2}/{}] {:<6}  WARNING — 0 rows in study window".format(i, len(tickers), ticker))
            warned.append(ticker)
        else:
            # Count rows with both estimate and actual present (usable for PEAD)
            complete = df.dropna(subset=["eps_estimate", "eps_actual"])
            est_null = df["eps_estimate"].isna().mean() * 100
            print("  [{:>2}/{}] {:<6}  {:>3} rows  complete={:>2}  est_null={:.0f}%  {} → {}".format(
                i, len(tickers), ticker, len(df), len(complete), est_null,
                df["earnings_date"].min().date(), df["earnings_date"].max().date()
            ))
            frames.append(df)

        time.sleep(DELAY_SECS)

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_df, failed, warned


# ── 3. Price fetch (yfinance batch download) ──────────────────────────────────

def collect_prices(tickers: List[str]) -> Tuple[pd.DataFrame, list]:
    print("\nDownloading yfinance prices for {} tickers ({} → {})...".format(
        len(tickers), START_DATE, END_DATE))

    raw = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # With multiple tickers, columns are MultiIndex: (field, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers

    long = (
        close
        .reset_index()
        .melt(id_vars="Date", var_name="ticker", value_name="close")
        .dropna(subset=["close"])
        .rename(columns={"Date": "date"})
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    succeeded  = set(long["ticker"].unique())
    failed     = [t for t in tickers if t not in succeeded]
    return long, failed


# ── 4. Summary ────────────────────────────────────────────────────────────────

def print_summary(
    earnings_df: pd.DataFrame,
    earn_failed: list,
    earn_warn: list,
    prices_df: pd.DataFrame,
    price_failed: list,
    batch: List[str],
) -> None:
    W = 65
    print("\n" + "=" * W)
    print("  BATCH 1 COLLECTION SUMMARY")
    print("=" * W)

    earn_ok = len(batch) - len(earn_failed) - len(earn_warn)
    print("\n  EARNINGS (yfinance get_earnings_dates)")
    print("  Tickers succeeded:   {} / {}".format(earn_ok, len(batch)))
    print("  Tickers with 0 rows: {}".format(len(earn_warn)))
    print("  Tickers failed:      {}".format(len(earn_failed)))

    if not earnings_df.empty:
        per_ticker  = earnings_df.groupby("ticker").size()
        complete_df = earnings_df.dropna(subset=["eps_estimate", "eps_actual"])
        est_null    = earnings_df["eps_estimate"].isna().mean() * 100
        act_null    = earnings_df["eps_actual"].isna().mean() * 100
        print("  Total rows:          {}".format(len(earnings_df)))
        print("  Complete rows (PEAD-ready): {} ({:.0f}%)".format(
            len(complete_df), len(complete_df) / len(earnings_df) * 100))
        print("  Avg rows / ticker:   {:.1f}  (min {}, max {})".format(
            per_ticker.mean(), per_ticker.min(), per_ticker.max()))
        print("  Date range:          {} → {}".format(
            earnings_df["earnings_date"].min().date(),
            earnings_df["earnings_date"].max().date()))
        print("  eps_estimate null:   {:.1f}%".format(est_null))
        print("  eps_actual null:     {:.1f}%".format(act_null))

    if earn_failed:
        print("\n  Failed tickers (earnings):")
        for t, e in earn_failed:
            print("    {}: {}".format(t, e[:80]))
    if earn_warn:
        print("\n  Zero-row tickers: {}".format(", ".join(earn_warn)))

    price_ok = len(batch) - len(price_failed)
    print("\n  PRICES (yfinance)")
    print("  Tickers succeeded:   {} / {}".format(price_ok, len(batch)))
    print("  Tickers failed:      {}".format(len(price_failed)))

    if not prices_df.empty:
        per_ticker_p = prices_df.groupby("ticker").size()
        print("  Total rows:          {}".format(len(prices_df)))
        print("  Avg trading days:    {:.0f}  (min {}, max {})".format(
            per_ticker_p.mean(), per_ticker_p.min(), per_ticker_p.max()))
        print("  Date range:          {} → {}".format(
            prices_df["date"].min().date(), prices_df["date"].max().date()))

    if price_failed:
        print("\n  Failed tickers (prices): {}".format(", ".join(price_failed)))

    print("\n" + "=" * W + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    all_tickers = get_sp500_tickers()
    batch       = all_tickers[:BATCH_SIZE]
    print("Batch 1: {} → {}".format(batch[0], batch[-1]))

    earnings_df, earn_failed, earn_warn = collect_earnings(batch)
    prices_df,   price_failed           = collect_prices(batch)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    earn_out  = OUTPUT_DIR / "earnings_batch1.csv"
    price_out = OUTPUT_DIR / "prices_batch1.csv"

    earnings_df.to_csv(earn_out,  index=False)
    prices_df.to_csv(price_out, index=False)

    print("Saved: {}  ({} rows)".format(earn_out,  len(earnings_df)))
    print("Saved: {}  ({} rows)".format(price_out, len(prices_df)))

    print_summary(earnings_df, earn_failed, earn_warn, prices_df, price_failed, batch)


if __name__ == "__main__":
    main()
