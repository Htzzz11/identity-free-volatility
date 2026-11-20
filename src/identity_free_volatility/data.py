"""Corpora and rolling-by-quarter splits shared by every model.

A Corpus is a frame of calls (call_id, ticker, date, year, quarter, lv3..lv30)
plus, for each test quarter such as "2021Q1", an array assigning every call to
train / val / test / none.

DEC ships its own split columns and we use them untouched so numbers are
comparable with Yu et al. (2025), Table 4. For STRUX we rebuild the same
protocol: test = one calendar quarter, every earlier call is train or val
(random 2:1 within the history, redrawn per test quarter).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .labels import HORIZONS
from .paths import DEC_CSV, DEC_DRIVE, PROCESSED

LABELS = [f"lv{h}" for h in HORIZONS]
_QNAME = {"first": 1, "second": 2, "third": 3, "fourth": 4}


@dataclass
class Corpus:
    name: str
    frame: pd.DataFrame
    splits: dict[str, np.ndarray] = field(default_factory=dict)

    def masks(self, test: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = self.splits[test]
        return s == "train", s == "val", s == "test"

    @property
    def y(self) -> np.ndarray:
        return self.frame[LABELS].to_numpy(dtype=float)


def load_dec() -> Corpus:
    raw = pd.read_csv(DEC_CSV)
    frame = pd.DataFrame({
        "call_id": raw["id"],
        "ticker": raw["ticker"],
        "sector": raw["sector"],
        "date": pd.to_datetime(raw["day_earnings"]),
        "year": raw["year"].astype(int),
        "quarter": raw["quarter"].map(_QNAME).astype(int),
        "timing": raw["beforeAfterMarket"],
        **{f"lv{h}": raw[f"lv{h}_future_{h}"] for h in HORIZONS},
    })
    splits = {}
    for col in raw.columns:
        if col.startswith("rolling_test_on_"):
            _, _, _, year, q, _ = col.split("_")
            splits[f"{year}Q{_QNAME[q]}"] = raw[col].to_numpy(dtype=str)
    return Corpus("dec", frame, splits)


def rolling_splits(frame: pd.DataFrame, first_test: str = "2019Q1",
                   val_frac: float = 1 / 3, seed: int = 0) -> dict[str, np.ndarray]:
    period = frame["year"] * 4 + (frame["quarter"] - 1)
    y0, q0 = first_test.split("Q")
    start = int(y0) * 4 + int(q0) - 1
    splits = {}
    for p in sorted(set(period[period >= start])):
        s = np.full(len(frame), "none", dtype=object)
        hist = (period < p).to_numpy()
        rng = np.random.default_rng([seed, p])
        is_val = rng.random(len(frame)) < val_frac
        s[hist & ~is_val] = "train"
        s[hist & is_val] = "val"
        s[(period == p).to_numpy()] = "test"
        splits[f"{p // 4}Q{p % 4 + 1}"] = s.astype(str)
    return splits


def load_strux(first_test: str = "2019Q1", seed: int = 0) -> Corpus:
    """Dense STRUX subset: tickers with >= 20 calls, calls with all four labels."""
    calls = pd.read_parquet(PROCESSED / "calls.parquet",
                            columns=["call_id", "ticker", "date", "year", "quarter",
                                     "dense", "timing", "timing_source", *LABELS])
    frame = calls[calls["dense"] & calls[LABELS].notna().all(axis=1)]
    frame = frame.drop(columns="dense").sort_values(["ticker", "date"]).reset_index(drop=True)
    return Corpus("strux", frame, rolling_splits(frame, first_test, seed=seed))


def load_corpus(name: str) -> Corpus:
    return {"dec": load_dec, "strux": load_strux}[name]()


def load_text(corpus: Corpus) -> list[str]:
    """Full transcript text aligned with corpus.frame. Local only, never commit."""
    if corpus.name == "dec":
        with open(DEC_DRIVE / "DEC.json") as f:
            raw = json.load(f)
        return [raw[f"{cid}.txt"] for cid in corpus.frame["call_id"]]
    text = pd.read_parquet(PROCESSED / "calls.parquet",
                           columns=["call_id", "prepared_text", "qa_text"]).set_index("call_id")
    text = text.loc[corpus.frame["call_id"]]
    return (text["prepared_text"] + "\n" + text["qa_text"]).tolist()


def load_dec_embeddings(corpus: Corpus) -> np.ndarray:
    """Yu et al.'s frozen OpenAI text-embedding-3-large vectors (1800 x 3072)."""
    with np.load(DEC_DRIVE / "Embeddings" / "openai" / "DEC.npz", allow_pickle=False) as z:
        emb, ids = z["embeddings"], z["ids"]
    assert (ids == corpus.frame["call_id"].to_numpy()).all(), "embedding ids do not match DEC.csv"
    return emb.astype(np.float32)
