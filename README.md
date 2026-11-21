# Same Company, Different Quarter

Isolating call-specific volatility signal in earnings call transcripts.
COMS 4705 (Natural Language Processing), Columbia, Fall 2026.

Yanshengtian Dong · Songhan Wu · Tianzhi Hou

## Question

Yu, Liu and He (ACL Findings 2025) show that transcript models for
post-earnings volatility mostly learn company identity: a per-ticker historical
mean (STPEV) beats them, and random per-ticker vectors match real embeddings.
We ask how much volatility signal a call carries **after** identity is removed
from its representation, and whether identity-removal training recovers signal
that standard training hides.

- **H1** A model trained on the residual target `v - STPEV_c` beats STPEV.
- **H2** Domain-adversarial training (ticker as domain) lowers post-hoc ticker
  probe accuracy without raising residual error.
- **H3** Representations of the change between consecutive calls carry more
  call-specific signal than single-call representations.

Targets: log volatility over the next 3, 7, 15, 30 trading days.
Metrics: MSE per horizon (DEC rolling-by-quarter protocol), Spearman on
residuals, post-hoc ticker probe accuracy. Three seeds per configuration.

## Setup

```bash
uv sync                                   # or: python -m venv .venv && .venv/bin/pip install -e . pytest
uv run python scripts/download_data.py    # STRUX, DEC labels, DEC transcripts + embeddings -> data/raw/
uv run python scripts/build_labels.py     # prices + labels -> data/processed/calls.parquet
uv run python scripts/validate_labels.py  # agreement with DEC, timing-recovery accuracy
uv run pytest
```

Everything under `data/` and `results/` stays local. Transcript text must never
be committed or re-hosted (STRUX is CC BY-NC-ND, DEC text is third-party).

## Labels

`v_tau = log sqrt( (1/tau) * sum_{i=0}^{tau-1} (r_{t+i} - rbar)^2 )` for tau in
{3, 7, 15, 30}, with **simple** daily returns of adjusted closes and `t` the first
session that prices in the release (release day if before the open, next session
if after the close). This is the DEC / Qin and Yang definition: it reproduces
DEC's label columns exactly and the Target Q3 2017 check (tau = 3: -2.726).

STRUX has only the call date, so the before/after-market flag (relative to the
call date) is recovered from DEC's own label window where the corpora overlap,
then Yahoo's earnings timestamp, then the greeting in the transcript header,
then the ticker's own majority. DEC's raw `beforeAfterMarket` column describes
the vendor's report date, not the call date, so it is not used directly. See
`src/identity_free_volatility/strux.py` and `docs/mason_plan.md` for the checks.

## Baselines

```bash
uv run python scripts/run_baselines.py --corpus dec      # DEC, the authors' rolling splits
uv run python scripts/run_baselines.py --corpus strux    # STRUX tickers with >= 20 calls
```

PEV(Mean), STPEV(Mean), Random(Ticker), Random(All), TF-IDF + ridge, and an MLP
on DEC's frozen OpenAI embeddings. Each model writes
`results/preds/<corpus>/<model>/seed<k>.parquet` (`call_id, test, lv3, lv7, lv15,
lv30`); `evaluate.summarize` scores every model found there with the same
metrics, so new models only need to write that file. Splits come from
`data.load_corpus(name)`: `corpus.masks("2021Q1")` gives train / val / test masks.
