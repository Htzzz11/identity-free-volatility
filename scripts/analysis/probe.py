"""Ticker probe on frozen DEC call embeddings (milestone, Table/Results).

Train on each ticker's first 15 calls, test on its last 5. Standardize + PCA-256
(fit on train), then multinomial logistic regression (C=1) and a 1-layer MLP (512).
Reports top-1/top-5 and the between-ticker share of embedding variance.
Usage: python scripts/analysis/probe.py [repo_root]
"""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
R = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2])
d = pd.read_csv(f'{R}/data/raw/dec_repo/SameCompanySameSignal/dataset/DEC.csv')
z = np.load(f'{R}/data/raw/dec_drive/Embeddings/openai/DEC.npz', allow_pickle=True)
assert (z['ids'] == d['id'].to_numpy()).all()
embs = {'openai': z['embeddings'].astype(np.float32),
        'sonnet_summary_bge': np.load(f'{R}/data/processed/summaries/dec/claude-sonnet-5-5/emb_bge-m3.npy'),
        'random_all': np.random.default_rng(0).random((1800, 3072), dtype=np.float32)}
d['date'] = pd.to_datetime(d['day_earnings'])
rank = d.groupby('ticker')['date'].rank(method='first')
n_per = d.groupby('ticker')['id'].transform('count')
test = (rank > n_per - 5).to_numpy()   # last 5 calls of each ticker
codes, uniq = pd.factorize(d['ticker']); y = codes
print('tickers', len(uniq), 'train', (~test).sum(), 'test', test.sum())
def topk(P, y, k):
    return np.mean([yy in np.argsort(-p)[:k] for p, yy in zip(P, y)])
for name, X in embs.items():
    # between-ticker share of embedding variance (ANOVA, all calls)
    Xc = X - X.mean(0)
    cent = pd.DataFrame(X).groupby(codes).transform('mean').to_numpy() - X.mean(0)
    between = (cent**2).sum() / (Xc**2).sum()
    sc = StandardScaler().fit(X[~test]); Xtr, Xte = sc.transform(X[~test]), sc.transform(X[test])
    pca = PCA(256, random_state=0).fit(Xtr); Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
    res = {'between_var_share': between}
    lin = LogisticRegression(max_iter=3000, C=1.0).fit(Ptr, y[~test])
    P = lin.predict_proba(Pte); res['lin_top1'] = (P.argmax(1) == y[test]).mean(); res['lin_top5'] = topk(P, y[test], 5)
    mlp = MLPClassifier((512,), max_iter=500, early_stopping=False, alpha=1e-3, random_state=0).fit(Ptr, y[~test])
    P = mlp.predict_proba(Pte); res['mlp_top1'] = (P.argmax(1) == y[test]).mean(); res['mlp_top5'] = topk(P, y[test], 5)
    # control task (Hewitt & Liang): random but consistent per-call label -> memorization only
    print(name, {k: round(float(v), 3) for k, v in res.items()})
print('chance top1', 1/len(uniq), 'top5', 5/len(uniq))
