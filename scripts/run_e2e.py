"""End-to-end fine-tuning, one test quarter at a time. Safe to stop and rerun.

  uv run python scripts/run_e2e.py --corpus dec            # start or resume
  uv run python scripts/run_e2e.py --corpus dec --report   # table on finished quarters

Needs the frozen chunk cache (scripts/encode_chunks.py). Quarters run from
--first (default 2021Q1) onward; earlier quarters come last.
"""
from __future__ import annotations

import argparse
import time

import torch

from identity_free_volatility import data
from identity_free_volatility.baselines.mlp import pick_device
from identity_free_volatility.models import chunks, e2e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["dec", "strux"], required=True)
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--first", default="2021Q1")
    ap.add_argument("--only", nargs="+", help="run just these test quarters")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--max-train-chunks", type=int, default=e2e.MAX_TRAIN_CHUNKS,
                    help="chunks per call in one training step; lower it if the GPU runs out of memory")
    ap.add_argument("--max-train", type=int, help="smoke test only")
    ap.add_argument("--max-eval", type=int, help="smoke test only")
    args = ap.parse_args()

    corpus = data.load_corpus(args.corpus)
    e2e.MAX_TRAIN_CHUNKS = args.max_train_chunks
    if not args.report:
        device = pick_device()
        toks, frozen = e2e.token_chunks(corpus), chunks.load_chunks(corpus)
        order = args.only or ([t for t in corpus.splits if t >= args.first] + [t for t in corpus.splits if t < args.first])
        done = set(e2e.finished(corpus, args.seed))
        print(f"{corpus.name}: {len(order)} quarters planned, {len(done & set(order))} already finished, device {device}", flush=True)
        for test in order:
            if test in done:
                continue
            t0 = time.time()
            e2e.run_quarter(corpus, toks, frozen, test, args.seed, device, log=lambda s: print(s, flush=True),
                            max_train=args.max_train, max_eval=args.max_eval)
            print(f"  {test}: quarter took {(time.time() - t0) / 60:.0f} min", flush=True)
            if device.type == "mps":
                torch.mps.empty_cache()
    table = e2e.report(corpus, args.seed)
    if table is None:
        print("no finished quarter yet")
    else:
        print(f"\nfinished quarters: {', '.join(table.attrs['tests'])}\n")
        print(table.round(4).to_markdown(index=False))


if __name__ == "__main__":
    main()
