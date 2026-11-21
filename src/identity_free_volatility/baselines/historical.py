"""Training-free baselines from Yu et al. (2025).

PEV(Mean):   mean label over every call visible before the test quarter.
STPEV(Mean): same, restricted to the test call's own ticker (falls back to
             PEV for a ticker with no history yet).
"Visible" = train + val of the rolling split, i.e. all earlier quarters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import LABELS, Corpus


def _predict(corpus: Corpus, same_ticker: bool) -> pd.DataFrame:
    frame, out = corpus.frame, []
    for test in corpus.splits:
        tr, va, te = corpus.masks(test)
        hist = frame[tr | va]
        pev = hist[LABELS].mean()
        pred = pd.DataFrame(np.tile(pev.to_numpy(), (te.sum(), 1)), columns=LABELS,
                            index=frame.index[te])
        if same_ticker:
            by_ticker = hist.groupby("ticker")[LABELS].mean()
            own = by_ticker.reindex(frame.loc[te, "ticker"]).set_index(pred.index)
            pred = own.fillna(pred)
        out.append(pred.assign(call_id=frame.loc[te, "call_id"], test=test))
    return pd.concat(out, ignore_index=True)


def pev(corpus: Corpus) -> pd.DataFrame:
    return _predict(corpus, same_ticker=False)


def stpev(corpus: Corpus) -> pd.DataFrame:
    return _predict(corpus, same_ticker=True)
