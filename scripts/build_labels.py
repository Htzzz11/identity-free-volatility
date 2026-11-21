"""Build the call table with volatility labels.

Output: data/processed/calls.parquet with one row per STRUX call:
  call_id, ticker, date, timing (relative to the call date), timing_source,
  dec/yahoo/greeting (per-source flags), t (first priced session),
  lv3/lv7/lv15/lv30, dense (ticker has >= 20 calls), dup_suspect (same call
  listed under a second date), year, quarter, prepared_text, qa_text.

Run:  .venv/bin/python scripts/build_labels.py
"""
from __future__ import annotations

import pandas as pd

from identity_free_volatility import labels, prices, strux
from identity_free_volatility.paths import PROCESSED


def main() -> None:
    calls = strux.load_strux()
    print(f"STRUX: {len(calls)} calls, {calls.ticker.nunique()} tickers, "
          f"{calls.dense.sum()} calls on {calls[calls.dense].ticker.nunique()} dense tickers")

    close = prices.cache_closes(calls.ticker.unique())
    earnings = prices.cache_earnings_dates(calls.loc[calls.dense, "ticker"].unique())

    rel = strux.recover_release(calls, earnings, close)
    print("timing source:\n", rel.timing_source.value_counts(dropna=False).to_string())

    rows = []
    for r in rel.itertuples(index=False):
        if pd.isna(r.timing) or r.ticker not in close.columns:
            rows.append({"call_id": r.call_id, "t": None,
                         **{f"lv{h}": float("nan") for h in labels.HORIZONS}})
            continue
        rows.append({"call_id": r.call_id,
                     **labels.labels_for_call(close[r.ticker], r.date, r.timing)})
    lab = pd.DataFrame(rows)

    out = (calls[["call_id", "ticker", "date", "n_calls", "dense", "prepared_text", "qa_text"]]
           .merge(rel.drop(columns=["ticker", "date"]), on="call_id")
           .merge(lab, on="call_id"))
    out["dup_suspect"] = strux.duplicate_suspects(out)
    out["year"] = out["date"].dt.year        # calendar quarter of the call, as in DEC
    out["quarter"] = out["date"].dt.quarter
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED / "calls.parquet")
    out.drop(columns=["prepared_text", "qa_text"]).to_csv(PROCESSED / "labels.csv", index=False)
    ok = out.lv3.notna()
    print(f"labelled {ok.sum()}/{len(out)} calls ({(ok & out.dense).sum()} dense), "
          f"{out.dup_suspect.sum()} duplicate suspects ({(out.dup_suspect & out.dense).sum()} dense)")
    print(out.loc[ok, [f"lv{h}" for h in labels.HORIZONS]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
