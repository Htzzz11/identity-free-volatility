"""End-to-end fine-tuning of the hierarchical model (standard training).

Same architecture as models/hier.py, but the DeBERTa chunk encoder is updated
too. For each test quarter:
  1. train the top (document transformer + head) on the cached frozen chunk
     vectors, exactly as hier.py does;
  2. start from the pretrained encoder and that top, and fine-tune everything
     on full transcripts, one call per forward pass, gradients accumulated
     over ACCUM calls;
  3. keep the weights with the best val MSE (the frozen model of step 1 is
     the epoch-0 candidate), predict the test quarter, write it to disk.

Resumable: a finished quarter is one parquet file and is skipped next time; an
unfinished quarter restarts from its last checkpoint (written every
CKPT_EVERY calls and at each epoch end). Kill the process whenever you like.

Only calls longer than MAX_TRAIN_CHUNKS chunks are subsampled, and only in
training (1.4% of DEC calls). Validation and test always read the full call.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data import LABELS, Corpus, load_text
from ..paths import PROCESSED, REPO, RESULTS
from . import chunks as chunk_cache
from . import hier

NAME = "hier_e2e_deberta"
EPOCHS, ACCUM, CKPT_EVERY = 3, 4, 100
LR_ENCODER, LR_TOP, WEIGHT_DECAY, WARMUP = 1e-5, 1e-4, 0.01, 0.1
MAX_TRAIN_CHUNKS = 40


def token_chunks(corpus: Corpus, encoder: str = chunk_cache.ENCODER) -> list[list[list[int]]]:
    """Token ids of every chunk of every call, same windows as the frozen cache."""
    path = PROCESSED / "chunks" / corpus.name / f"{encoder.split('/')[-1]}_tokens.pt"
    if path.exists():
        return torch.load(path)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(encoder)
    out = [chunk_cache.chunk_ids(tok, t) for t in load_text(corpus)]
    torch.save(out, path)
    return out


class E2E(nn.Module):
    def __init__(self, encoder: nn.Module, top: hier.HierModel, pad_id: int):
        super().__init__()
        self.encoder, self.top, self.pad_id = encoder, top, pad_id

    def chunk_vectors(self, chunks: list[list[int]], batch: int | None = None) -> torch.Tensor:
        device = next(self.parameters()).device
        out = []
        step = batch or len(chunks)
        for i in range(0, len(chunks), step):
            part = chunks[i:i + step]
            x = torch.full((len(part), max(len(c) for c in part)), self.pad_id, dtype=torch.long)
            for r, c in enumerate(part):
                x[r, :len(c)] = torch.tensor(c)
            x = x.to(device)
            mask = x != self.pad_id
            h = self.encoder(input_ids=x, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).to(h.dtype)
            out.append((h * m).sum(1) / m.sum(1))
        return torch.cat(out)

    def forward(self, chunks: list[list[int]], vectors: bool = False, batch: int | None = None) -> torch.Tensor:
        v = self.chunk_vectors(chunks, batch).unsqueeze(0)
        pad = torch.zeros(1, v.shape[1], dtype=torch.bool, device=v.device)
        return self.top.call_vector(v, pad)[0] if vectors else self.top(v, pad)[0]


@torch.no_grad()
def predict(model: E2E, toks, idx, vectors: bool = False) -> torch.Tensor:
    model.eval()
    return torch.stack([model(toks[i], vectors=vectors, batch=32) for i in idx])


def _quarter_dir(corpus: Corpus, seed: int) -> Path:
    return RESULTS / "preds_partial" / corpus.name / NAME / f"seed{seed}"


def finished(corpus: Corpus, seed: int) -> list[str]:
    return sorted(p.stem for p in _quarter_dir(corpus, seed).glob("*.parquet"))


def run_quarter(corpus: Corpus, toks, frozen, test: str, seed: int, device, log=print,
                max_train: int | None = None, max_eval: int | None = None) -> None:
    """max_train / max_eval shrink the quarter for smoke tests only."""
    from transformers import AutoModel, AutoTokenizer

    out_dir = _quarter_dir(corpus, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = REPO / "checkpoints" / corpus.name / NAME / f"seed{seed}" / test
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last, best_file, init_file = ckpt_dir / "last.pt", ckpt_dir / "best.pt", ckpt_dir / "init.pt"

    tr, va, te = corpus.masks(test)
    tr_idx, va_idx, te_idx = (np.flatnonzero(m) for m in (tr, va, te))
    if max_train:
        tr_idx = tr_idx[:max_train]
    if max_eval:
        va_idx, te_idx = va_idx[:max_eval], te_idx[:max_eval]
    y = torch.as_tensor(corpus.y, dtype=torch.float32, device=device)

    # step 1: top trained on frozen chunk vectors (saved so a resume reuses the same start)
    if init_file.exists():
        init = torch.load(init_file, map_location=device)
    else:
        x, pad = (t.to(device) for t in hier.pad_chunks(frozen))
        trm, vam = torch.zeros(len(y), dtype=torch.bool, device=device), torch.zeros(len(y), dtype=torch.bool, device=device)
        trm[torch.as_tensor(tr_idx, device=device)] = True
        vam[torch.as_tensor(va_idx, device=device)] = True
        top0, mu, sd = hier.fit(x, pad, y, trm, vam, seed)
        z = (y - mu) / sd
        val0 = hier._predict(top0, x, pad, torch.as_tensor(va_idx, device=device), loss_target=z)
        init = {"top": top0.state_dict(), "mu": mu, "sd": sd, "val": val0}
        torch.save(init, init_file)
        del x, pad
    mu, sd = init["mu"].to(device), init["sd"].to(device)
    z = (y - mu) / sd

    tok = AutoTokenizer.from_pretrained(chunk_cache.ENCODER)
    top = hier.HierModel(frozen[0].shape[1])
    top.load_state_dict(init["top"])
    model = E2E(AutoModel.from_pretrained(chunk_cache.ENCODER), top, tok.pad_token_id).to(device)
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": LR_ENCODER},
                             {"params": model.top.parameters(), "lr": LR_TOP}], weight_decay=WEIGHT_DECAY)
    total = EPOCHS * int(np.ceil(len(tr_idx) / ACCUM))
    warm = max(1, int(WARMUP * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    epoch0, pos0, best_val, best_epoch = 0, 0, init["val"], 0
    if last.exists():
        state = torch.load(last, map_location=device)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"]); sched.load_state_dict(state["sched"])
        epoch0, pos0, best_val, best_epoch = state["epoch"], state["pos"], state["best_val"], state["best_epoch"]
        log(f"  {test}: resumed at epoch {epoch0 + 1}, call {pos0}/{len(tr_idx)}")
    log(f"  {test}: train {len(tr_idx)} val {len(va_idx)} test {len(te_idx)}, frozen val mse {init['val']:.4f}")

    def save_last(epoch, pos):
        tmp = ckpt_dir / "last.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "pos": pos, "best_val": best_val, "best_epoch": best_epoch}, tmp)
        tmp.rename(last)

    qcode = int(test[:4]) * 4 + int(test[-1])
    for epoch in range(epoch0, EPOCHS):
        perm = np.random.default_rng([seed, qcode, epoch]).permutation(tr_idx)
        pick = np.random.default_rng([seed, qcode, epoch, 1])
        model.train()
        opt.zero_grad()
        t0, seen, running = time.time(), 0, 0.0
        for pos in range(pos0 if epoch == epoch0 else 0, len(perm)):
            i = int(perm[pos])
            c = toks[i]
            if len(c) > MAX_TRAIN_CHUNKS:   # keep order, drop random chunks (deterministic in pos)
                keep = np.sort(np.random.default_rng([seed, qcode, epoch, pos]).choice(len(c), MAX_TRAIN_CHUNKS, replace=False))
                c = [c[k] for k in keep]
            loss = nn.functional.mse_loss(model(c), z[i])
            (loss / ACCUM).backward()
            running += loss.item(); seen += len(c)
            if (pos + 1) % ACCUM == 0 or pos + 1 == len(perm):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if (pos + 1) % 20 == 0:
                done = pos + 1 - (pos0 if epoch == epoch0 else 0)
                el = time.time() - t0
                log(f"    epoch {epoch + 1}/{EPOCHS} call {pos + 1}/{len(perm)} loss {running / done:.3f} "
                    f"{seen / el:.1f} chunks/s, epoch eta {el / done * (len(perm) - pos - 1) / 60:.0f} min")
            if (pos + 1) % CKPT_EVERY == 0 and pos + 1 < len(perm):
                save_last(epoch, pos + 1)
        val = nn.functional.mse_loss(predict(model, toks, va_idx), z[torch.as_tensor(va_idx, device=device)]).item()
        log(f"  {test}: epoch {epoch + 1} val mse {val:.4f} (best so far {best_val:.4f} from epoch {best_epoch})")
        if val < best_val:
            best_val, best_epoch = val, epoch + 1
            torch.save(model.state_dict(), best_file)
        save_last(epoch + 1, 0)

    if best_epoch > 0:
        model.load_state_dict(torch.load(best_file, map_location=device))
    else:   # no epoch beat the frozen model: fall back to pretrained encoder + step-1 top
        top.load_state_dict(init["top"])
        model = E2E(AutoModel.from_pretrained(chunk_cache.ENCODER), top, tok.pad_token_id).to(device)
    pred = (predict(model, toks, te_idx) * sd + mu).cpu().numpy()
    vec = predict(model, toks, te_idx, vectors=True).cpu().numpy()
    mse = ((pred - corpus.y[te_idx]) ** 2).mean(axis=0)
    log(f"  {test}: DONE best epoch {best_epoch} val {best_val:.4f} test mse {np.round(mse, 3)} mean {mse.mean():.3f}")
    np.save(out_dir / f"{test}.vectors.npy", vec)
    frame = pd.DataFrame(pred, columns=LABELS).assign(
        call_id=corpus.frame["call_id"].to_numpy()[te_idx], test=test, best_epoch=best_epoch)
    tmp = out_dir / f"{test}.tmp"
    frame.to_parquet(tmp, index=False)
    tmp.rename(out_dir / f"{test}.parquet")   # written last: its presence marks the quarter finished
    for f in (last, best_file, init_file):
        f.unlink(missing_ok=True)


def report(corpus: Corpus, seed: int) -> pd.DataFrame | None:
    """Every model scored on the quarters this run has finished (mean over seeds)."""
    files = sorted(_quarter_dir(corpus, seed).glob("*.parquet"))
    if not files:
        return None
    ours = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tests = sorted(ours["test"].unique())
    truth = corpus.frame.set_index("call_id")[LABELS]

    def score(p):
        p = p[p["test"].isin(tests)].set_index("call_id")
        se = (p[LABELS] - truth.loc[p.index]) ** 2
        by_q = se.groupby(p["test"]).mean()
        return by_q.mean().mean(), len(p)

    rows = [{"model": f"{NAME} (1 seed)", "mse": score(ours)[0], "n": len(ours)}]
    for d in sorted((RESULTS / "preds" / corpus.name).iterdir()):
        s = [score(pd.read_parquet(f)) for f in sorted(d.glob("seed*.parquet"))]
        if s:
            rows.append({"model": d.name, "mse": float(np.mean([a for a, _ in s])), "n": s[0][1]})
    out = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
    out.attrs["tests"] = tests
    return out
