"""Linear ticker erasure (LEACE, INLP) on frozen DEC call embeddings.

Asks whether identity in the frozen embeddings is removed or only hidden, and
what erasing it does to volatility prediction.

1. Probe: erasers are fit on each ticker's first 15 calls (with ticker labels) and
   applied to all calls; then the probe.py protocol (standardize, PCA-256,
   logistic C=1 and MLP-512) is trained on the erased first 15 and tested on the
   erased last 5. A linear probe at chance with an MLP probe far above it means
   identity is hidden from linear readers, not removed.
2. Prediction: for each DEC test quarter 2021-2023, the eraser is fit on that
   quarter's train+val calls only, then ridge (alpha on val) predicts either the
   raw label v or the residual v - STPEV (M1, as in m2.py). Reports MSE, R2_oos
   against STPEV on the same calls, ticker R^2 of the predictions and the
   within-ticker Spearman of predictions with labels.

LEACE: Belrose et al. 2023, closed form, computed in the span of the fit data.
Matches EleutherAI's concept-erasure 0.2.4, LeaceFitter(shrinkage=False,
constrain_cov_trace=False), to 1e-13 on the DEC OpenAI and summary embeddings.
The library's defaults (covariance shrinkage, trace constraint) give a slightly
different eraser; with them the MLP probe after erasure is higher (OpenAI 0.144,
Sonnet summaries 0.296), so the plain version here is the conservative one.
INLP: Ravfogel et al. 2020, with a multi-class ridge classifier per iteration
(each iteration removes up to 89 directions). INLP-10 is used for the probe only:
in the prediction step it would remove more directions than early quarters have calls.

Usage: python scripts/analysis/erasure.py [openai|summary|both] [repo_root]
"""
import sys, time, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

which = sys.argv[1] if len(sys.argv) > 1 else 'both'
R = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parents[2])
H = (3, 7, 15, 30); L = [f'lv{h}' for h in H]
INLP_ITERS = (1, 3, 10)


# ---------------------------------------------------------------- erasers
class Leace:
    """x' = x - W^+ P W (x - mu), W the (pseudo-)whitening of the fit data and P the
    orthogonal projector onto the column space of W Sigma_xz. After erasure the
    class means of the fit data coincide, so no linear classifier beats a constant
    on it."""
    def fit(self, X, y, tol=1e-8):
        self.mu = X.mean(0); Xc = X - self.mu; n = len(X)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        keep = S > tol * S[0]
        self.V = Vt[keep].T                       # d x r basis of the data span
        s = S[keep] / np.sqrt(n)                  # sqrt of covariance eigenvalues
        Z = np.array(pd.get_dummies(y), dtype=float); Z = Z - Z.mean(0)
        Wsxz = (Xc @ self.V / s).T @ Z / n        # W Sigma_xz in span coordinates, r x k
        Q, sv, _ = np.linalg.svd(Wsxz, full_matrices=False)
        Q = Q[:, sv > tol * sv[0]]
        self.s, self.Q = s, Q
        return self
    def transform(self, X):
        c = (X - self.mu) @ self.V                # span coordinates
        w = c / self.s                            # whiten
        e = (w @ self.Q) @ self.Q.T * self.s      # project, un-whiten
        return X - e @ self.V.T


class Inlp:
    """Repeatedly fit a linear ticker classifier and project its row space out."""
    def fit(self, X, y, n_iter):
        self.mu = X.mean(0); Xc = X - self.mu
        d = X.shape[1]; B = np.zeros((d, 0))
        for _ in range(n_iter):
            Xp = Xc - (Xc @ B) @ B.T
            W = RidgeClassifier(alpha=1.0).fit(Xp, y).coef_      # k x d
            W = W - (W @ B) @ B.T                                # stay orthogonal to removed dirs
            U, sv, _ = np.linalg.svd(W.T, full_matrices=False)
            U = U[:, sv > 1e-8 * sv[0]]
            B = np.linalg.qr(np.hstack([B, U]))[0]
        self.B = B
        return self
    def transform(self, X):
        return X - (X @ self.B) @ self.B.T


def erasers(Xfit, yfit, iters=INLP_ITERS):
    out = {'none': None, 'leace': Leace().fit(Xfit, yfit)}
    for k in iters:
        out[f'inlp{k}'] = Inlp().fit(Xfit, yfit, k)
    return out


def apply(e, X):
    return X if e is None else e.transform(X)


# ---------------------------------------------------------------- data
d = pd.read_csv(f'{R}/data/raw/dec_repo/SameCompanySameSignal/dataset/DEC.csv')
d = d.rename(columns={f'lv{h}_future_{h}': f'lv{h}' for h in H})
z = np.load(f'{R}/data/raw/dec_drive/Embeddings/openai/DEC.npz', allow_pickle=True)
assert (z['ids'] == d['id'].to_numpy()).all()
embs = {}
if which in ('openai', 'both'):
    embs['openai'] = z['embeddings'].astype(np.float64)
if which in ('summary', 'both'):
    embs['sonnet_summary_bge'] = np.load(f'{R}/data/processed/summaries/dec/claude-sonnet-5-5/emb_bge-m3.npy').astype(np.float64)
tick = d['ticker'].to_numpy()
codes, uniq = pd.factorize(tick)
d['date'] = pd.to_datetime(d['day_earnings'])
rank = d.groupby('ticker')['date'].rank(method='first')
probe_test = (rank > d.groupby('ticker')['id'].transform('count') - 5).to_numpy()
Y = d[L].to_numpy()
splits = {}
for col in d.columns:
    if col.startswith('rolling_test_on_'):
        _, _, _, yr, q, _ = col.split('_'); splits[(int(yr), q)] = d[col].astype(str).to_numpy()
tests = sorted(k for k in splits if 2021 <= k[0] <= 2023)


def ticker_share(v, t):
    m = v.groupby(t).transform('mean'); return ((m - v.mean())**2).sum() / ((v - v.mean())**2).sum()


# ---------------------------------------------------------------- 1. probe
def probe(X):
    tr, te = ~probe_test, probe_test
    sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    pca = PCA(256, random_state=0).fit(Xtr); Ptr, Pte = pca.transform(Xtr), pca.transform(Xte)
    lin = LogisticRegression(max_iter=3000, C=1.0).fit(Ptr, codes[tr])
    mlp = MLPClassifier((512,), max_iter=500, alpha=1e-3, random_state=0).fit(Ptr, codes[tr])
    # how much of the fit data's class structure is left in-sample (LEACE: should be ~0)
    cen = pd.DataFrame(X[tr]).groupby(codes[tr]).mean().to_numpy()
    spread = np.linalg.norm(cen - cen.mean(0), axis=1).mean()
    return {'lin_top1': (lin.predict(Pte) == codes[te]).mean(),
            'mlp_top1': (mlp.predict(Pte) == codes[te]).mean(),
            'train_centroid_spread': spread}


# ---------------------------------------------------------------- 2. prediction
def predict(X, target):
    rows = []
    for k in tests:
        s = splits[k]; tr = s == 'train'; va = s == 'val'; te = s == 'test'; fit = tr | va
        mu = pd.DataFrame(Y[fit], columns=L).groupby(tick[fit]).mean().reindex(tick).to_numpy()   # STPEV
        T = Y - mu if target == 'resid' else Y
        # INLP-10 removes ~890 directions, more than the rank of the early quarters'
        # history (720 calls for 2021Q1), which leaves nothing of the fit data; skip it here.
        ers = erasers(X[fit], tick[fit], iters=(1, 3))
        for name, e in ers.items():
            F = apply(e, X)
            sc = StandardScaler().fit(F[tr]); best = None
            for a in (1e2, 1e3, 1e4, 1e5):
                m = Ridge(alpha=a).fit(sc.transform(F[tr]), T[tr])
                err = ((m.predict(sc.transform(F[va])) - T[va])**2).mean()
                if best is None or err < best[0]: best = (err, a)
            sc = StandardScaler().fit(F[fit])
            p = Ridge(alpha=best[1]).fit(sc.transform(F[fit]), T[fit]).predict(sc.transform(F[te]))
            if target == 'resid': p = p + mu[te]
            rows.append(pd.DataFrame({'eraser': name, 'q': str(k), 'tick': tick[te],
                                      **{f'p{c}': p[:, j] for j, c in enumerate(L)},
                                      **{f'y{c}': Y[te, j] for j, c in enumerate(L)},
                                      **{f's{c}': mu[te, j] for j, c in enumerate(L)}}))
    return pd.concat(rows, ignore_index=True)


def score(df, pre):
    mse = np.mean([((df[f'{pre}{c}'] - df[f'y{c}'])**2).groupby(df['q']).mean().mean() for c in L])
    r2t = np.mean([ticker_share(df[f'{pre}{c}'], df['tick']) for c in L])
    rw = []
    for c in L:
        r = [spearmanr(g[f'{pre}{c}'], g[f'y{c}']).statistic for _, g in df.groupby('tick')
             if len(g) >= 5 and g[f'{pre}{c}'].std() > 0]
        rw.append(np.mean(r))
    return mse, r2t, np.mean(rw)


for ename, X in embs.items():
    t0 = time.time()
    print(f'\n=== {ename}  ({X.shape[1]} dims)')
    print('-- probe: erasers fit on first 15 calls per ticker, probe tested on last 5 (chance 0.011)')
    ers = erasers(X[~probe_test], tick[~probe_test])
    for name, e in ers.items():
        r = probe(apply(e, X))
        extra = f'  removed dims {e.B.shape[1]}' if isinstance(e, Inlp) else (f'  removed dims {e.Q.shape[1]}' if isinstance(e, Leace) else '')
        print(f'  {name:7s} linear {r["lin_top1"]:.3f}  mlp {r["mlp_top1"]:.3f}  '
              f'train centroid spread {r["train_centroid_spread"]:.4f}{extra}')
    for target in ('raw', 'resid'):
        df = predict(X, target)
        base = score(df[df['eraser'] == 'none'], 's')
        print(f'-- ridge on {target} target, DEC 2021-2023 (n={int((df["eraser"] == "none").sum())}); '
              f'STPEV mse {base[0]:.4f}, ticker R2 {base[1]:.2f}, rho_w {base[2]:+.3f}')
        for name, g in df.groupby('eraser', sort=False):
            mse, r2t, rw = score(g, 'p')
            print(f'  {name:7s} mse {mse:.4f}  r2oos {1 - mse / base[0]:+.4f}  ticker R2 {r2t:.2f}  rho_w {rw:+.4f}')
    print(f'({time.time() - t0:.0f}s)')
