"""Standard training of the hierarchical model on frozen chunk vectors.

  uv run python scripts/encode_chunks.py --corpus dec     # once, about 45 min on an M4 Max
  uv run python scripts/run_hier.py --corpus dec --seeds 2021

Writes results/preds/<corpus>/hier_frozen_deberta/seed<k>.parquet, the call
vectors under results/vectors/, and reprints the summary table.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from identity_free_volatility import data, evaluate
from identity_free_volatility.baselines.mlp import pick_device
from identity_free_volatility.models import chunks, hier
from identity_free_volatility.paths import RESULTS

NAME = "hier_frozen_deberta"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["dec", "strux"], required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022, 2023])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    corpus = data.load_corpus(args.corpus)
    device = torch.device(args.device) if args.device else pick_device()
    vecs = chunks.load_chunks(corpus)
    n = [len(v) for v in vecs]
    print(f"{corpus.name}: {len(vecs)} calls, chunks per call mean {np.mean(n):.1f} max {max(n)}, device {device}")
    for seed in args.seeds:
        t0 = time.time()
        print(f"== {NAME} seed {seed}")
        preds, call_vectors = hier.run(corpus, vecs, seed, device)
        evaluate.save_preds(corpus, NAME, seed, preds)
        out = RESULTS / "vectors" / corpus.name / NAME
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"seed{seed}.npy", call_vectors)
        print(f"  done in {time.time() - t0:.0f}s")

    years = (2021, 2022, 2023) if corpus.name == "dec" else (2021, 2022, 2023, 2024)
    for title, ys in (("all test quarters", None), (f"{years[0]}-{years[-1]}", years)):
        print(f"\n### {corpus.name}: {title}\n")
        print(evaluate.summarize(corpus, ys).round(3).to_markdown(index=False))


if __name__ == "__main__":
    main()
