"""Between-ticker R^2 of predictions: share of prediction variance explained by ticker,
DEC test quarters 2021-2023, reading results/preds/dec/<model>/<seed>.parquet.
Usage: python scripts/analysis/r2.py [repo_root]
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
R = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[2])
d = pd.read_csv(f'{R}/data/raw/dec_repo/SameCompanySameSignal/dataset/DEC.csv', usecols=['id','ticker']+[f'lv{h}_future_{h}' for h in (3,7,15,30)])
d = d.rename(columns={f'lv{h}_future_{h}': f'lv{h}' for h in (3,7,15,30)}).set_index('id')
L = ['lv3','lv7','lv15','lv30']
def share(v, t):
    m = v.groupby(t).transform('mean'); return ((m - v.mean())**2).sum() / ((v - v.mean())**2).sum()
models = {'stpev':'stpev/seed0','pev':'pev/seed0','tfidf_ridge':'tfidf_ridge/seed0','openai_mlp':'openai_mlp/seed2021',
          'random_ticker':'random_ticker/seed2021','summary_sonnet':'summary_sonnet-5-5_mlp/seed2021'}
first = None
for name, f in models.items():
    p = pd.read_parquet(f'{R}/results/preds/dec/{f}.parquet')
    p = p[p['test'].str[:4].astype(int).between(2021, 2023)].set_index('call_id')
    t = d.loc[p.index, 'ticker']
    if first is None:
        first = 1
        print('truth y', round(np.mean([share(d.loc[p.index, c], t) for c in L]), 3), 'n', len(p))
    print(name, round(np.mean([share(p[c], t) for c in L]), 3))
