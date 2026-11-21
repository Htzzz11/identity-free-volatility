"""Shared prediction format and metrics.

Every model writes  results/preds/<corpus>/<model>/seed<k>.parquet  with columns
  call_id, test ("2021Q1"), lv3, lv7, lv15, lv30   (predicted log volatility)
and summarize() scores whatever is there, so fine-tuned models and baselines
land in the same table.

Metrics (per seed, then mean and std over seeds):
  mse{h}       MSE per horizon, computed per test quarter then averaged over
               quarters (the convention of Yu et al., Table 4)
  mse          mean over the four horizons
  r2_oos       1 - mse / mse(STPEV): above 0 means signal beyond the ticker mean
  rho_resid    Spearman between (pred - STPEV) and (y - STPEV), pooled
  rho_within   Spearman between pred and y inside each ticker, averaged over
               tickers with >= 5 test calls (company level differences cannot
               contribute)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .baselines import historical
from .data import LABELS, Corpus
from .paths import RESULTS


def save_preds(corpus: Corpus, model: str, seed: int, preds: pd.DataFrame) -> None:
    path = RESULTS / "preds" / corpus.name / model
    path.mkdir(parents=True, exist_ok=True)
    preds[["call_id", "test", *LABELS]].to_parquet(path / f"seed{seed}.parquet", index=False)


def _score(corpus: Corpus, preds: pd.DataFrame, ref: pd.DataFrame, tests: list[str]) -> dict:
    truth = corpus.frame.set_index("call_id")
    p = preds[preds["test"].isin(tests)].set_index("call_id")
    r = ref.set_index("call_id").loc[p.index]
    y = truth.loc[p.index, LABELS]
    tick = truth.loc[p.index, "ticker"]
    se = (p[LABELS] - y) ** 2
    se_ref = (r[LABELS] - y) ** 2
    by_q = se.groupby(p["test"]).mean().mean()          # per quarter, then over quarters
    by_q_ref = se_ref.groupby(p["test"]).mean().mean()
    out = {f"mse{c[2:]}": by_q[c] for c in LABELS}
    out["mse"] = by_q.mean()
    out["r2_oos"] = 1 - by_q.mean() / by_q_ref.mean()
    resid, within = [], []
    for c in LABELS:
        dp, dy = p[c] - r[c], y[c] - r[c]
        resid.append(spearmanr(dp, dy).statistic if dp.std() > 0 else np.nan)
        per_ticker = [spearmanr(p.loc[i, c], y.loc[i, c]).statistic
                      for _, i in tick.groupby(tick).groups.items()
                      if len(i) >= 5 and p.loc[i, c].std() > 0]
        within.append(np.mean(per_ticker) if per_ticker else np.nan)
    out["rho_resid"] = np.nanmean(resid) if not np.all(np.isnan(resid)) else np.nan
    out["rho_within"] = np.nanmean(within) if not np.all(np.isnan(within)) else np.nan
    out["n"] = len(p)
    return out


def summarize(corpus: Corpus, years: tuple[int, ...] | None = None) -> pd.DataFrame:
    """One row per model: mean over seeds, plus the seed std of the average MSE."""
    tests = [t for t in corpus.splits if years is None or int(t[:4]) in years]
    ref = historical.stpev(corpus)
    rows = []
    root = RESULTS / "preds" / corpus.name
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        scores = [_score(corpus, pd.read_parquet(f), ref, tests)
                  for f in sorted(model_dir.glob("seed*.parquet"))]
        if not scores:
            continue
        s = pd.DataFrame(scores)
        rows.append({"model": model_dir.name, "seeds": len(s), **s.mean().to_dict(),
                     "mse_std": s["mse"].std(ddof=0)})
    cols = ["model", "seeds", "mse", "mse_std", "mse3", "mse7", "mse15", "mse30",
            "r2_oos", "rho_resid", "rho_within", "n"]
    return pd.DataFrame(rows)[cols].sort_values("mse").reset_index(drop=True)
