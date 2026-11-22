"""Frozen-embedding M1 (residual target) + M2 matched-information controls on DEC.

Ridge on standardized features predicting v - STPEV (alpha on val), DEC rolling splits,
test quarters 2021-2023. Inputs: A z_t, B z_{t-1}, C [z_t; z_{t-1}], D z_t - z_{t-1},
F z_t - mean of earlier z, G z_t - z of a random other ticker.
Usage: python scripts/analysis/m2.py [openai|summary] [repo_root]
"""
import sys, numpy as np, pandas as pd, warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
from pathlib import Path
which = sys.argv[1] if len(sys.argv) > 1 else 'openai'   # 'openai' or 'summary'
R = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parents[2])
H = (3, 7, 15, 30); L = [f'lv{h}' for h in H]
d = pd.read_csv(f'{R}/data/raw/dec_repo/SameCompanySameSignal/dataset/DEC.csv')
d = d.rename(columns={f'lv{h}_future_{h}': f'lv{h}' for h in H})
z = np.load(f'{R}/data/raw/dec_drive/Embeddings/openai/DEC.npz', allow_pickle=True)
X = z['embeddings'].astype(np.float64) if which == 'openai' else np.load(f'{R}/data/processed/summaries/dec/claude-sonnet-5-5/emb_bge-m3.npy').astype(np.float64)
d['date'] = pd.to_datetime(d['day_earnings'])
order = d.sort_values(['ticker', 'date']).index.to_numpy()
prev = np.full(len(d), -1); cent = np.full_like(X, np.nan)
for t, g in d.loc[order].groupby('ticker', sort=False):
    idx = g.index.to_numpy()
    for k in range(1, len(idx)):
        prev[idx[k]] = idx[k-1]; cent[idx[k]] = X[idx[:k]].mean(0)
rng = np.random.default_rng(0)
tick = d['ticker'].to_numpy()
other = np.array([rng.choice(np.flatnonzero((tick != tick[i]) & (prev >= 0))) if prev[i] >= 0 else -1 for i in range(len(d))])
has = prev >= 0
P = np.where(has, prev, 0); O = np.where(has, other, 0)
feats = {'A: z_t': X, 'B: z_{t-1}': X[P], 'C: [z_t; z_{t-1}]': np.hstack([X, X[P]]),
         'D: z_t - z_{t-1}': X - X[P], 'F: z_t - centroid': np.nan_to_num(X - cent),
         'G: z_t - z_other': X - X[O]}
Y = d[L].to_numpy()
splits = {}
for col in d.columns:
    if col.startswith('rolling_test_on_'):
        _, _, _, yr, q, _ = col.split('_'); splits[(int(yr), q)] = d[col].astype(str).to_numpy()
tests = sorted(k for k in splits if 2021 <= k[0] <= 2023)
def run(F):
    rows = []
    for k in tests:
        s = splits[k]; tr = (s == 'train') & has; va = (s == 'val') & has; te = (s == 'test') & has
        fit = tr | va
        hist = pd.DataFrame(Y[(s == 'train') | (s == 'val')], columns=L).groupby(tick[(s == 'train') | (s == 'val')]).mean()
        mu = hist.reindex(tick).to_numpy()            # ticker mean over history (STPEV), per call
        res = Y - mu
        sc = StandardScaler().fit(F[tr]); best = None
        for a in (1e2, 1e3, 1e4, 1e5):
            m = Ridge(alpha=a).fit(sc.transform(F[tr]), res[tr])
            e = ((m.predict(sc.transform(F[va])) - res[va])**2).mean()
            if best is None or e < best[0]: best = (e, a)
        sc = StandardScaler().fit(F[fit])
        m = Ridge(alpha=best[1]).fit(sc.transform(F[fit]), res[fit])
        pred = mu[te] + m.predict(sc.transform(F[te]))
        rows.append(pd.DataFrame({'q': str(k), 'tick': tick[te], **{f'p{c}': pred[:, j] for j, c in enumerate(L)},
                                  **{f'y{c}': Y[te, j] for j, c in enumerate(L)}, **{f's{c}': mu[te, j] for j, c in enumerate(L)}}))
    return pd.concat(rows)
def score(df, pre='p'):
    mse = np.mean([((df[f'{pre}{c}'] - df[f'y{c}'])**2).groupby(df['q']).mean().mean() for c in L])
    rw = []
    for c in L:
        r = [spearmanr(g[f'{pre}{c}'], g[f'y{c}']).statistic for _, g in df.groupby('tick') if len(g) >= 5 and g[f'{pre}{c}'].std() > 0]
        rw.append(np.mean(r) if r else np.nan)
    return mse, np.nanmean(rw)
base = None
for name, F in feats.items():
    df = run(F)
    if base is None:
        base = score(df, 's'); print(f'STPEV (same calls) mse={base[0]:.4f} n={len(df)}')
    mse, rw = score(df)
    print(f'{name:22s} mse={mse:.4f} r2oos={1-mse/base[0]:+.4f} rho_w={rw:+.3f}')
