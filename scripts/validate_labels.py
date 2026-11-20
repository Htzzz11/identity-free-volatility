"""Check our STRUX labels and timing recovery against DEC on the overlap.

1. Label agreement: for STRUX calls that also appear in DEC, compare our
   yfinance-based lv{tau} with DEC's lv{tau}_future_{tau}.
2. Timing recovery: how often each non-DEC source (yahoo, greeting,
   ticker_mode) picks the same first priced session as DEC. This is the
   accuracy to expect on the STRUX tickers that DEC does not cover.

Run:  .venv/bin/python scripts/validate_labels.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from identity_free_volatility import labels, strux
from identity_free_volatility.paths import DEC_CSV, PROCESSED


def main() -> None:
    calls = pd.read_parquet(PROCESSED / "calls.parquet").drop(columns=["prepared_text", "qa_text"])
    dec = pd.read_csv(DEC_CSV)
    dec["day"] = pd.to_datetime(dec["day_earnings"])
    keep = ["ticker", "day", "beforeAfterMarket"] + [f"lv{h}_future_{h}" for h in labels.HORIZONS]

    # A STRUX call on D matches the DEC row released on D, or after the close on D-1.
    same = calls.merge(dec[keep], left_on=["ticker", "date"], right_on=["ticker", "day"])
    prev = calls.assign(day=calls["date"] - pd.Timedelta(days=1)).merge(
        dec.loc[dec.beforeAfterMarket == labels.AMC, keep], on=["ticker", "day"])
    both = pd.concat([same, prev], ignore_index=True)
    print(f"overlap: {len(both)} of {len(dec)} DEC calls are in STRUX "
          f"({both.ticker.nunique()} tickers)")

    print("\n== label agreement (ours vs DEC), calls we could label ==")
    rows = []
    for h in labels.HORIZONS:
        ok = both[f"lv{h}"].notna()
        diff = (both.loc[ok, f"lv{h}"] - both.loc[ok, f"lv{h}_future_{h}"]).abs()
        rows.append({"tau": h, "n": int(ok.sum()), "median|diff|": diff.median(),
                     "p95|diff|": diff.quantile(0.95), "share<0.05": (diff < 0.05).mean(),
                     "pearson": np.corrcoef(both.loc[ok, f"lv{h}"], both.loc[ok, f"lv{h}_future_{h}"])[0, 1]})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n== timing recovery without DEC, scored against DEC ==")
    truth = np.where(both["day"] < both["date"], labels.BMO, both["beforeAfterMarket"])
    rows = []
    for src in ("yahoo", "greeting"):
        has = both[src].notna()
        rows.append({"source": src, "coverage": has.mean(),
                     "accuracy": (both.loc[has, src] == truth[has]).mean()})
    # the full fallback chain as it runs on non-DEC tickers
    known = both["yahoo"].where(both["yahoo"].notna(), both["greeting"])
    mode = strux._ticker_mode(both, known)
    chain = known.where(known.notna(), mode)
    has = chain.notna()
    rows.append({"source": "yahoo > greeting > ticker_mode", "coverage": has.mean(),
                 "accuracy": (chain[has] == truth[has]).mean()})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    dense = calls[calls.dense]
    print(f"\n== dense set (>= {strux.MIN_CALLS} calls per ticker) ==")
    print(f"{dense.ticker.nunique()} tickers, {len(dense)} calls, "
          f"{dense.lv30.notna().sum()} with all four labels")
    print("timing source:", dense.timing_source.value_counts(dropna=False).to_dict())
    lost = dense[dense.lv30.isna()].groupby("ticker").size()
    print("tickers with no usable prices:", sorted(lost[lost >= 10].index))


if __name__ == "__main__":
    main()
