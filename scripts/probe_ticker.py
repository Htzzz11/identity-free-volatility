"""Ticker probe on the hierarchical model's call vectors, same protocol as
scripts/analysis/probe.py so the numbers are directly comparable.

Train on each ticker's first 15 calls, test on its last 5. StandardScaler and
PCA-256 fit on train, then logistic regression (C=1) and a 1-layer MLP (512).
Also prints the correlation of each model's predictions with STPEV.

  uv run python scripts/probe_ticker.py
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from identity_free_volatility import data
from identity_free_volatility.baselines import historical
from identity_free_volatility.models import chunks
from identity_free_volatility.paths import RESULTS

warnings.filterwarnings("ignore")
NAME = "hier_frozen_deberta"


def topk(p, y, k):
    return float(np.mean([yy in np.argsort(-row)[:k] for row, yy in zip(p, y)]))


def probe(x, y, test):
    sc = StandardScaler().fit(x[~test])
    xtr, xte = sc.transform(x[~test]), sc.transform(x[test])
    pca = PCA(min(256, x.shape[1]), random_state=0).fit(xtr)
    ptr, pte = pca.transform(xtr), pca.transform(xte)
    out = {}
    for name, clf in (("lin", LogisticRegression(max_iter=3000, C=1.0)),
                      ("mlp", MLPClassifier((512,), max_iter=500, alpha=1e-3, random_state=0))):
        p = clf.fit(ptr, y[~test]).predict_proba(pte)
        out[f"{name}_top1"], out[f"{name}_top5"] = float((p.argmax(1) == y[test]).mean()), topk(p, y[test], 5)
    centered = x - x.mean(0)
    means = pd.DataFrame(x).groupby(y).transform("mean").to_numpy() - x.mean(0)
    out["between_var_share"] = float((means ** 2).sum() / (centered ** 2).sum())
    return out


def main() -> None:
    c = data.load_corpus("dec")
    f = c.frame
    y = pd.factorize(f["ticker"])[0]
    rank = f.groupby("ticker")["date"].rank(method="first")
    test = (rank > f.groupby("ticker")["call_id"].transform("count") - 5).to_numpy()
    print(f"train {int((~test).sum())} test {int(test.sum())} chance top1 {1 / (y.max() + 1):.3f}")
    rows = {}
    for seed_file in sorted((RESULTS / "vectors" / "dec" / NAME).glob("seed*.npy")):
        rows[f"{NAME} {seed_file.stem}"] = probe(np.load(seed_file), y, test)
    rows["mean-pooled frozen DeBERTa chunks"] = probe(np.stack([v.mean(0) for v in chunks.load_chunks(c)]), y, test)
    rows["openai (check against analysis/probe.py)"] = probe(data.load_dec_embeddings(c), y, test)
    print(pd.DataFrame(rows).T.round(3).to_string())

    ref = historical.stpev(c).set_index("call_id")
    for m in (NAME, "openai_mlp", "random_ticker", "tfidf_ridge"):
        rs = []
        for path in sorted((RESULTS / "preds" / "dec" / m).glob("seed*.parquet")):
            p = pd.read_parquet(path)
            p = p[p["test"].str[:4].astype(int) >= 2021].set_index("call_id")
            rs.append(np.mean([pearsonr(p[l], ref.loc[p.index, l]).statistic for l in data.LABELS]))
        print(f"corr with STPEV, {m}: {np.mean(rs):.3f}")


if __name__ == "__main__":
    main()
