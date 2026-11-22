"""Chunk vectors from a frozen pretrained encoder.

Each transcript is tokenized once, cut into windows of 510 tokens, wrapped in
[CLS] ... [SEP], and encoded by the frozen encoder. The chunk vector is the
mean of the last hidden states over real tokens. One .npy per call
(n_chunks x hidden, float16) under data/processed/chunks/<corpus>/<encoder>/,
so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from ..baselines.mlp import pick_device
from ..data import Corpus, load_text
from ..paths import PROCESSED

ENCODER = "microsoft/deberta-v3-base"
WINDOW = 510


def cache_dir(corpus: Corpus, encoder: str = ENCODER) -> Path:
    return PROCESSED / "chunks" / corpus.name / encoder.split("/")[-1]


def encode_corpus(corpus: Corpus, encoder: str = ENCODER, batch: int = 32,
                  limit: int | None = None, log=print) -> None:
    from transformers import AutoModel, AutoTokenizer

    out = cache_dir(corpus, encoder)
    out.mkdir(parents=True, exist_ok=True)
    ids = corpus.frame["call_id"].tolist()
    texts = load_text(corpus)
    todo = [(c, t) for c, t in zip(ids, texts) if not (out / f"{c}.npy").exists()]
    if limit is not None:
        todo = todo[:limit]
    log(f"{len(ids) - len(todo)} calls cached, {len(todo)} to encode")
    if not todo:
        return
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(encoder)
    model = AutoModel.from_pretrained(encoder).eval().to(device)
    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
    t0, n_chunks = time.time(), 0
    for k, (cid, text) in enumerate(todo):
        toks = tok(text, add_special_tokens=False, verbose=False)["input_ids"]
        chunks = [[cls, *toks[i:i + WINDOW], sep] for i in range(0, max(len(toks), 1), WINDOW)]
        vecs = []
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            width = max(len(c) for c in part)
            x = torch.full((len(part), width), pad, dtype=torch.long)
            for r, c in enumerate(part):
                x[r, :len(c)] = torch.tensor(c)
            mask = (x != pad).to(device)
            with torch.no_grad():
                h = model(input_ids=x.to(device), attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).to(h.dtype)
            vecs.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
        v = np.concatenate(vecs)
        assert np.isfinite(v).all(), cid
        tmp = out / f"{cid}.tmp.npy"
        np.save(tmp, v.astype(np.float16))
        tmp.rename(out / f"{cid}.npy")
        n_chunks += len(chunks)
        if (k + 1) % 25 == 0 or k + 1 == len(todo):
            el = time.time() - t0
            log(f"  {k + 1}/{len(todo)} calls, {n_chunks / el:.1f} chunks/s, "
                f"eta {el / (k + 1) * (len(todo) - k - 1) / 60:.0f} min")


def load_chunks(corpus: Corpus, encoder: str = ENCODER) -> list[np.ndarray]:
    """Chunk matrices aligned with corpus.frame. Fails if any call is missing."""
    d = cache_dir(corpus, encoder)
    return [np.load(d / f"{c}.npy").astype(np.float32) for c in corpus.frame["call_id"]]
