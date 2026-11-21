"""TF-IDF + ridge on the full transcript.

Uni+bigram counts are computed once for the corpus; for every test quarter the
vocabulary (document frequency >= MIN_DF, top MAX_FEATURES by term frequency)
and the IDF weights are re-derived from the fitting rows only, so nothing from
the test quarter leaks in. Alpha is chosen on val, then the model is refit on
train + val. One ridge per horizon, sharing alpha.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import Ridge

from ..data import LABELS, Corpus

MIN_DF = 5
MAX_FEATURES = 50_000
ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0)
# words only: digits would hand the model the fiscal year and raw dollar figures
TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z'-]+\b"


def count_matrix(texts: list[str]):
    cv = CountVectorizer(lowercase=True, ngram_range=(1, 2), min_df=MIN_DF,
                         token_pattern=TOKEN, dtype=np.float32)
    return cv.fit_transform(texts).tocsr()


def _features(counts, fit_rows: np.ndarray, rows: list[np.ndarray]):
    sub = counts[fit_rows]
    df = np.asarray((sub > 0).sum(axis=0)).ravel()
    tf = np.asarray(sub.sum(axis=0)).ravel()
    keep = np.flatnonzero(df >= MIN_DF)
    if len(keep) > MAX_FEATURES:
        keep = np.sort(keep[np.argsort(-tf[keep], kind="stable")[:MAX_FEATURES]])
    tfidf = TfidfTransformer(sublinear_tf=True).fit(sub[:, keep])
    return [tfidf.transform(counts[r][:, keep]) for r in rows]


def run(corpus: Corpus, counts, log=print) -> pd.DataFrame:
    y = corpus.y
    out = []
    for test in corpus.splits:
        tr, va, te = corpus.masks(test)
        x_tr, x_va = _features(counts, tr, [tr, va])
        val_mse = {}
        for alpha in ALPHAS:
            model = Ridge(alpha=alpha).fit(x_tr, y[tr])
            val_mse[alpha] = float(((model.predict(x_va) - y[va]) ** 2).mean())
        alpha = min(val_mse, key=val_mse.get)
        fit = tr | va
        x_fit, x_te = _features(counts, fit, [fit, te])
        pred = Ridge(alpha=alpha).fit(x_fit, y[fit]).predict(x_te)
        mse = ((pred - y[te]) ** 2).mean(axis=0)
        log(f"  {test} n_fit={int(fit.sum())} alpha={alpha} mse={np.round(mse, 3)}")
        out.append(pd.DataFrame(pred, columns=LABELS).assign(
            call_id=corpus.frame.loc[te, "call_id"].to_numpy(), test=test))
    return pd.concat(out, ignore_index=True)
