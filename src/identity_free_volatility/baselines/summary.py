"""LLM-summary baseline, step 2: embed the summaries and train the same MLP.

Mirrors Yu et al.'s "Summarization" rows (LLM summary -> text embedding -> MLP).
They embed with OpenAI text-embedding-3-large; we use an open encoder that runs
locally (BAAI/bge-m3, 1024-d, 8k-token context) so this step costs nothing.
Summaries come from scripts/llm_summaries.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..data import Corpus
from ..paths import PROCESSED
from .mlp import pick_device

ENCODER = "BAAI/bge-m3"


def load_summaries(corpus: Corpus, model: str) -> list[str]:
    path = PROCESSED / "summaries" / corpus.name / model / "summaries.parquet"
    got = pd.read_parquet(path)
    got = got[got["stop_reason"] == "end_turn"].set_index("call_id")["summary"]
    missing = corpus.frame.loc[~corpus.frame["call_id"].isin(got.index), "call_id"]
    if len(missing):
        raise ValueError(f"{len(missing)} calls have no summary yet, e.g. {list(missing[:3])}")
    return got.loc[corpus.frame["call_id"]].tolist()


@torch.no_grad()
def embed(texts: list[str], encoder: str = ENCODER, batch: int = 8, max_length: int = 2048) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder).to(device).eval()
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(device)
        cls = model(**enc).last_hidden_state[:, 0]          # bge: normalized [CLS]
        out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def summary_embeddings(corpus: Corpus, model: str) -> np.ndarray:
    cache = PROCESSED / "summaries" / corpus.name / model / f"emb_{ENCODER.split('/')[-1]}.npy"
    if cache.exists():
        return np.load(cache)
    emb = embed(load_summaries(corpus, model))
    np.save(cache, emb)
    return emb
