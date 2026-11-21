"""STRUX transcripts (Motley Fool, 2017-2024) and release-time recovery.

STRUX gives only the calendar date D of the call. The label needs the first
trading session that prices in the release, which is fully determined by a
timing flag *relative to D*:
  BeforeMarket -> first session on or after D
  AfterMarket  -> first session strictly after D
We recover that flag from, in priority order:
  1. dec       DEC's own label window for the 90 DEC tickers (ground truth,
               see _dec_window_flags: the raw beforeAfterMarket column is not
               relative to the call date)
  2. yahoo     Yahoo's earnings timestamp (before 09:30 / 16:00 or later)
  3. greeting  the transcript header ("good morning" vs "good afternoon/evening";
               "good day" is ambiguous and skipped)
  4. ticker_mode  the ticker's own majority flag from sources 1-3
A release after the close on D-1 followed by a call on the morning of D is
BeforeMarket relative to D (first priced session is D either way).
scripts/validate_labels.py measures sources 2-4 against DEC.
"""
from __future__ import annotations

import glob
import re

import numpy as np
import pandas as pd

from .labels import AMC, BMO
from .paths import DEC_CSV, STRUX_DIR

MIN_CALLS = 20


def _join_turns(turns, max_turns=None) -> str:
    turns = list(turns)[:max_turns] if max_turns else list(turns)
    return "\n".join(f"{t['name']}: {' '.join(t['speech'])}" for t in turns)


def load_strux() -> pd.DataFrame:
    files = sorted(glob.glob(str(STRUX_DIR / "full-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no STRUX parquet under {STRUX_DIR}; run scripts/download_data.sh")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"]).reset_index(drop=True)
    df["call_id"] = df["ticker"] + "_" + df["date"].dt.strftime("%Y%m%d")
    df["header"] = df["prepared_remarks"].map(lambda pr: _join_turns(pr, 3))
    df["prepared_text"] = df["prepared_remarks"].map(_join_turns)
    df["qa_text"] = df["questions_and_answers"].map(_join_turns)
    n = df.groupby("ticker")["date"].transform("size")
    df["n_calls"] = n
    df["dense"] = n >= MIN_CALLS
    return df


_GREET = re.compile(r"good\s+(morning|afternoon|evening|day)", re.I)


def greeting_timing(header: str) -> str | None:
    m = _GREET.search(header or "")
    if not m:
        return None
    g = m.group(1).lower()
    if g == "morning":
        return BMO
    if g in ("afternoon", "evening"):
        return AMC
    return None  # "good day"


def _yf_timing(ts: pd.Timestamp) -> str | None:
    minutes = ts.hour * 60 + ts.minute
    if minutes < 9 * 60 + 30:
        return BMO
    if minutes >= 16 * 60:
        return AMC
    return None  # in-session or placeholder noon time


def _match_release(out: pd.DataFrame, rel: pd.DataFrame) -> pd.Series:
    """rel has ticker, day, flag. Flag relative to the call date, or None."""
    rel = rel.dropna(subset=["flag"]).drop_duplicates(["ticker", "day"])
    same = out[["ticker", "date"]].merge(rel, left_on=["ticker", "date"],
                                         right_on=["ticker", "day"], how="left")["flag"]
    prev = out[["ticker"]].assign(day=out["date"] - pd.Timedelta(days=1)).merge(
        rel, on=["ticker", "day"], how="left")["flag"]
    eff = same.to_numpy(dtype=object, copy=True)
    use_prev = pd.isna(eff) & (prev == AMC).to_numpy()
    eff[use_prev] = BMO  # released after the close on D-1, priced on D
    return pd.Series(eff, index=out.index).where(lambda x: x.notna(), None)


def _dec_window_flags(out: pd.DataFrame, close: pd.DataFrame, tol: float = 4e-3) -> pd.Series:
    """Flag relative to the call date, read off DEC's own label window.

    DEC's beforeAfterMarket describes its vendor's report date, which is not
    always the day_earnings column (the call date): CCI, EOG, O, PSA, WELL, PPG
    and others report after the close and hold the call the next morning, so
    "AfterMarket" applied to the call date starts the window one day late.
    Instead we check on which session DEC's first five returns begin: the first
    session on or after the call date (BeforeMarket) or the one after (AfterMarket).
    """
    cols = ["ticker", "day_earnings"] + [f"p_future_{k}" for k in range(1, 6)]
    dec = pd.read_csv(DEC_CSV, usecols=cols)
    dec["day"] = pd.to_datetime(dec["day_earnings"])
    window = {(r.ticker, r.day): np.array(r[3:8], dtype=float)
              for r in dec[["ticker", "day", "day_earnings", *cols[2:]]].itertuples(index=False)}
    returns = {}
    flags = np.full(len(out), None, dtype=object)
    for pos, (ticker, date) in enumerate(zip(out["ticker"], out["date"])):
        theirs = window.get((ticker, date))
        if theirs is None:
            theirs = window.get((ticker, date - pd.Timedelta(days=1)))
        if theirs is None or ticker not in close.columns:
            continue
        if ticker not in returns:
            series = close[ticker].dropna()
            returns[ticker] = (series.index, series.pct_change().to_numpy())
        sessions, ret = returns[ticker]
        for flag, side in ((BMO, "left"), (AMC, "right")):
            i = sessions.searchsorted(date, side=side)
            ours = ret[i:i + 5]
            if len(ours) == 5 and np.abs(ours - theirs).max() < tol:
                flags[pos] = flag
                break
    return pd.Series(flags, index=out.index)


def timing_candidates(calls: pd.DataFrame, earnings: pd.DataFrame | None,
                      close: pd.DataFrame | None = None) -> pd.DataFrame:
    """One column of flags per source: dec, yahoo, greeting (None where silent).
    The dec source needs the price cache; without `close` it stays empty."""
    out = calls[["call_id", "ticker", "date"]].reset_index(drop=True)
    out["greeting"] = calls["header"].map(greeting_timing).to_numpy()
    out["yahoo"] = None
    if earnings is not None and len(earnings):
        ts = pd.to_datetime(earnings["ts"])
        rel = pd.DataFrame({"ticker": earnings["ticker"].to_numpy(), "day": ts.dt.normalize().to_numpy(),
                            "flag": ts.map(_yf_timing).to_numpy()})
        out["yahoo"] = _match_release(out, rel)
    out["dec"] = None
    if DEC_CSV.exists() and close is not None:
        out["dec"] = _dec_window_flags(out, close)
    return out


def _ticker_mode(out: pd.DataFrame, known: pd.Series, min_share=0.75) -> pd.Series:
    share = (known == BMO).groupby(out["ticker"]).sum() / known.notna().groupby(out["ticker"]).sum()
    mode = pd.Series(None, index=share.index, dtype=object)
    mode[share >= min_share] = BMO
    mode[share <= 1 - min_share] = AMC
    return out["ticker"].map(mode)


def recover_release(calls: pd.DataFrame, earnings: pd.DataFrame | None,
                    close: pd.DataFrame | None = None,
                    sources=("dec", "yahoo", "greeting", "ticker_mode")) -> pd.DataFrame:
    """Add timing (relative to the call date) and timing_source."""
    out = timing_candidates(calls, earnings, close)
    timing = pd.Series(None, index=out.index, dtype=object)
    source = pd.Series(None, index=out.index, dtype=object)
    for name in sources:
        cand = _ticker_mode(out, timing) if name == "ticker_mode" else out[name]
        fill = timing.isna() & cand.notna()
        timing[fill] = cand[fill]
        source[fill] = name
    out["timing"], out["timing_source"] = timing, source
    return out


def duplicate_suspects(calls: pd.DataFrame, window_days: int = 10) -> pd.Series:
    """STRUX lists a few calls twice under dates a few days apart (AMZN 2018-01-31
    and 2018-02-01, for example). Of two calls of one ticker within `window_days`,
    flag the one whose date no earnings calendar (DEC or Yahoo) confirms while the
    other one's is confirmed. Needs ticker, date, timing_source."""
    c = calls.sort_values(["ticker", "date"])
    confirmed = c["timing_source"].isin(["dec", "yahoo"])
    by = c.groupby("ticker")
    gap_prev = by["date"].diff().dt.days
    gap_next = -by["date"].diff(-1).dt.days
    conf_prev = confirmed.groupby(c["ticker"]).shift(1, fill_value=False)
    conf_next = confirmed.groupby(c["ticker"]).shift(-1, fill_value=False)
    suspect = ~confirmed & (((gap_prev <= window_days) & conf_prev) | ((gap_next <= window_days) & conf_next))
    return suspect.reindex(calls.index)
