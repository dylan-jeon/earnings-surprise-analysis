"""
Full S&P 500 data collection pipeline.

Skips the first 50 tickers already collected in batch 1, fetches earnings and
prices for the remaining ~453 tickers via yfinance, then merges batch 1 data
into complete output files.

Outputs:
  data/raw/earnings_full.csv  — all 500 tickers, EPS actual + estimate, 2015+
  data/raw/prices_full.csv    — all 500 tickers, daily adjusted close, 2015–today
  data/raw/failed_tickers.csv — any tickers that errored, with reason
"""

import sys
import time
import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import date
from typing import List, Tuple, Optional

warnings.filterwarnings("ignore")

START_DATE  = "2015-01-01"
END_DATE    = str(date.today())
BATCH1_SIZE = 50
EPS_LIMIT   = 80        # ~20 years of quarters per ticker
EARN_DELAY  = 0.5       # seconds between earnings calls
PRICE_CHUNK = 50        # tickers per yfinance batch price download
LOG_EVERY   = 50        # print progress every N tickers

BASE_DIR    = Path(__file__).parent.parent
RAW_DIR     = BASE_DIR / "data" / "raw"
BATCH1_EARN = RAW_DIR / "earnings_batch1.csv"
BATCH1_PRIC = RAW_DIR / "prices_batch1.csv"
FULL_EARN   = RAW_DIR / "earnings_full.csv"
FULL_PRIC   = RAW_DIR / "prices_full.csv"
FAILED_PATH = RAW_DIR / "failed_tickers.csv"


# ── Ticker list ───────────────────────────────────────────────────────────────

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
        .str.replace(".", "-", regex=False)
        .tolist()
    )
    print("  {} total S&P 500 tickers found.".format(len(tickers)))
    return tickers


# ── Earnings (yfinance, one ticker at a time) ─────────────────────────────────

def fetch_one_earnings(ticker: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        raw = yf.Ticker(ticker).get_earnings_dates(limit=EPS_LIMIT)
    except Exception as e:
        return None, "earnings fetch error: {}".format(str(e)[:120])

    if raw is None or raw.empty:
        return None, "empty earnings response"

    df = raw.copy()
    df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
    df.index.name = "earnings_date"
    df = df.reset_index()
    df = df.rename(columns={
        "EPS Estimate": "eps_estimate",
        "Reported EPS": "eps_actual",
        "Surprise(%)":  "surprise_pct_yf",
    })
    df["ticker"] = ticker
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df = df[df["earnings_date"] >= START_DATE].copy()
    df = df[["ticker", "earnings_date", "eps_estimate", "eps_actual"]].copy()
    df = df.sort_values("earnings_date").reset_index(drop=True)
    return df, None


# ── Prices (yfinance batch download) ─────────────────────────────────────────

def fetch_prices_chunk(tickers: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    try:
        raw = yf.download(
            tickers,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        return pd.DataFrame(), tickers

    if raw.empty:
        return pd.DataFrame(), tickers

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers[:1]

    long = (
        close
        .reset_index()
        .melt(id_vars="Date", var_name="ticker", value_name="close")
        .dropna(subset=["close"])
        .rename(columns={"Date": "date"})
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    succeeded = set(long["ticker"].unique())
    failed    = [t for t in tickers if t not in succeeded]
    return long, failed


# ── Main collection loop ──────────────────────────────────────────────────────

def collect_remaining(tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    n          = len(tickers)
    earn_parts = []
    fail_log   = []
    earn_failed, earn_ok = 0, 0

    print("\nCollecting earnings for {} tickers (skipped first {} from batch 1)...".format(n, BATCH1_SIZE))
    print("ETA: ~{:.0f} minutes at {}s delay/call.\n".format(n * EARN_DELAY / 60, EARN_DELAY))

    for i, ticker in enumerate(tickers, 1):
        df, err = fetch_one_earnings(ticker)

        if err:
            fail_log.append({"ticker": ticker, "source": "earnings", "error": err})
            earn_failed += 1
        elif df is None or df.empty:
            fail_log.append({"ticker": ticker, "source": "earnings", "error": "0 rows in 2015+ window"})
            earn_failed += 1
        else:
            earn_parts.append(df)
            earn_ok += 1

        if i % LOG_EVERY == 0 or i == n:
            pct = i / n * 100
            print("  [{:>4}/{}]  {:.0f}%  earn_ok={:>4}  earn_failed={:>3}  ticker={}".format(
                i, n, pct, earn_ok, earn_failed, ticker))
            sys.stdout.flush()

        time.sleep(EARN_DELAY)

    # Batch price downloads in chunks
    print("\nDownloading prices in chunks of {}...".format(PRICE_CHUNK))
    price_parts = []
    chunks = [tickers[j:j + PRICE_CHUNK] for j in range(0, n, PRICE_CHUNK)]

    for k, chunk in enumerate(chunks, 1):
        df_p, chunk_failed = fetch_prices_chunk(chunk)
        if not df_p.empty:
            price_parts.append(df_p)
        for t in chunk_failed:
            fail_log.append({"ticker": t, "source": "prices", "error": "no price data returned"})
        print("  Chunk {}/{}: {} tickers, {} rows returned, {} failed".format(
            k, len(chunks), len(chunk),
            len(df_p), len(chunk_failed)))
        sys.stdout.flush()

    earn_df  = pd.concat(earn_parts,  ignore_index=True) if earn_parts  else pd.DataFrame()
    price_df = pd.concat(price_parts, ignore_index=True) if price_parts else pd.DataFrame()
    return earn_df, price_df, fail_log


# ── Merge with batch 1 ────────────────────────────────────────────────────────

def merge_with_batch1(
    new_earn: pd.DataFrame,
    new_price: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    b1_earn  = pd.read_csv(BATCH1_EARN,  parse_dates=["earnings_date"])
    b1_price = pd.read_csv(BATCH1_PRIC,  parse_dates=["date"])

    full_earn = (
        pd.concat([b1_earn, new_earn], ignore_index=True)
        .drop_duplicates(subset=["ticker", "earnings_date"])
        .sort_values(["ticker", "earnings_date"])
        .reset_index(drop=True)
    )
    full_price = (
        pd.concat([b1_price, new_price], ignore_index=True)
        .drop_duplicates(subset=["ticker", "date"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    return full_earn, full_price


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(
    full_earn: pd.DataFrame,
    full_price: pd.DataFrame,
    fail_log: List[dict],
    n_attempted: int,
) -> None:
    W = 65
    print("\n" + "=" * W)
    print("  FULL S&P 500 COLLECTION — FINAL SUMMARY")
    print("=" * W)

    earn_tickers  = full_earn["ticker"].nunique()
    price_tickers = full_price["ticker"].nunique()
    n_failed      = len(set(r["ticker"] for r in fail_log))

    print("\n  Tickers attempted (new):    {}".format(n_attempted))
    print("  Tickers with earnings data: {} (incl. batch 1)".format(earn_tickers))
    print("  Tickers with price data:    {} (incl. batch 1)".format(price_tickers))
    print("  Tickers with any failure:   {}".format(n_failed))

    print("\n  EARNINGS (earnings_full.csv)")
    print("  Total rows:     {:,}".format(len(full_earn)))
    complete = full_earn.dropna(subset=["eps_estimate", "eps_actual"])
    print("  Complete rows:  {:,} ({:.0f}%)".format(
        len(complete), len(complete) / len(full_earn) * 100))
    print("  Date range:     {} → {}".format(
        full_earn["earnings_date"].min().date(),
        full_earn["earnings_date"].max().date()))
    per_t = full_earn.groupby("ticker").size()
    print("  Avg rows/ticker: {:.1f}  (min {}, max {})".format(
        per_t.mean(), per_t.min(), per_t.max()))

    print("\n  PRICES (prices_full.csv)")
    print("  Total rows:     {:,}".format(len(full_price)))
    print("  Date range:     {} → {}".format(
        full_price["date"].min().date(),
        full_price["date"].max().date()))
    per_tp = full_price.groupby("ticker").size()
    print("  Avg trading days/ticker: {:.0f}  (min {}, max {})".format(
        per_tp.mean(), per_tp.min(), per_tp.max()))

    if fail_log:
        fail_df = pd.DataFrame(fail_log)
        earn_fails  = fail_df[fail_df["source"] == "earnings"]
        price_fails = fail_df[fail_df["source"] == "prices"]
        print("\n  FAILURES")
        print("  Earnings: {}  |  Prices: {}".format(len(earn_fails), len(price_fails)))
        if len(fail_log) <= 30:
            for r in fail_log:
                print("    [{source}] {ticker}: {error}".format(**r))
        else:
            print("  (See {} for full list)".format(FAILED_PATH))

    print("\n" + "=" * W + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_tickers      = get_sp500_tickers()
    remaining        = all_tickers[BATCH1_SIZE:]
    print("Remaining tickers to collect: {} (indices {}–{})".format(
        len(remaining), BATCH1_SIZE, len(all_tickers) - 1))

    new_earn, new_price, fail_log = collect_remaining(remaining)

    print("\nMerging with batch 1 data...")
    full_earn, full_price = merge_with_batch1(new_earn, new_price)

    full_earn.to_csv(FULL_EARN,   index=False)
    full_price.to_csv(FULL_PRIC,  index=False)

    if fail_log:
        pd.DataFrame(fail_log).to_csv(FAILED_PATH, index=False)

    print("Saved: {}  ({:,} rows)".format(FULL_EARN,  len(full_earn)))
    print("Saved: {}  ({:,} rows)".format(FULL_PRIC,  len(full_price)))
    if fail_log:
        print("Saved: {}  ({} failure records)".format(FAILED_PATH, len(fail_log)))

    print_summary(full_earn, full_price, fail_log, len(remaining))


if __name__ == "__main__":
    main()
