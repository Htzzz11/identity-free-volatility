"""Cache adjusted daily closes and earnings timestamps from yfinance.

Closes:   data/prices/close.parquet        (wide: date x ticker, auto-adjusted)
Earnings: data/prices/earnings_dates.parquet (ticker, ts in America/New_York)
"""
from __future__ import annotations

import time

import pandas as pd
import yfinance as yf

from .paths import PRICES

CLOSE_FILE = PRICES / "close.parquet"
EARN_FILE = PRICES / "earnings_dates.parquet"

# Motley Fool / STRUX tickers that Yahoo lists under another symbol.
# Renamed since the transcripts were scraped: Marsh McLennan, Fiserv, BNY.
YF_ALIASES = {"BRK.B": "BRK-B", "BF.B": "BF-B", "FB": "META",
              "MMC": "MRSH", "FI": "FISV", "BK": "BNY"}


def yf_symbol(ticker: str) -> str:
    return YF_ALIASES.get(ticker, ticker.replace(".", "-"))


def _download(symbols, start, end) -> pd.DataFrame:
    df = yf.download(list(symbols), start=start, end=end, auto_adjust=True,
                     progress=False, threads=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(list(symbols)[0])
    return df.dropna(axis=1, how="all")


def cache_closes(tickers, start="2016-06-01", end="2024-12-31", chunk=40, retries=3) -> pd.DataFrame:
    """Download missing tickers into the wide close cache. Tickers Yahoo cannot
    serve (delisted, renamed) are simply absent from the result."""
    PRICES.mkdir(parents=True, exist_ok=True)
    have = pd.read_parquet(CLOSE_FILE) if CLOSE_FILE.exists() else pd.DataFrame()
    todo = sorted({t for t in tickers if t not in have.columns})
    frames = [have] if len(have) else []
    for attempt in range(retries):
        if not todo:
            break
        got = set()
        size = chunk if attempt == 0 else 10  # yahoo throttles big batches; retry small
        for k in range(0, len(todo), size):
            syms = {yf_symbol(t): t for t in todo[k:k + size]}
            df = _download(syms, start, end).rename(columns=syms)
            frames.append(df)
            got |= set(df.columns)
            time.sleep(1.0 if attempt else 0.2)
        todo = [t for t in todo if t not in got]
    if todo:
        print(f"[prices] no data for {len(todo)} tickers: {todo}")
    if not frames:
        return have
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()].sort_index()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    out.to_parquet(CLOSE_FILE)
    return out


def cache_earnings_dates(tickers, limit=100, pause=0.3) -> pd.DataFrame:
    PRICES.mkdir(parents=True, exist_ok=True)
    have = pd.read_parquet(EARN_FILE) if EARN_FILE.exists() else pd.DataFrame(columns=["ticker", "ts"])
    rows = [have]
    for t in sorted(set(tickers) - set(have["ticker"])):
        try:
            e = yf.Ticker(yf_symbol(t)).get_earnings_dates(limit=limit)
        except Exception as exc:  # noqa: BLE001 - yahoo is flaky; log and move on
            print(f"[earnings] {t}: {exc}")
            continue
        if e is None or len(e) == 0:
            continue
        ts = pd.DatetimeIndex(e.index).tz_convert("America/New_York").tz_localize(None)
        rows.append(pd.DataFrame({"ticker": t, "ts": ts}))
        time.sleep(pause)
    out = pd.concat(rows, ignore_index=True).drop_duplicates()
    out.to_parquet(EARN_FILE)
    return out
