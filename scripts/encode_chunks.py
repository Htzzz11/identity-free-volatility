"""Encode every transcript into frozen chunk vectors (resumable).

  uv run python scripts/encode_chunks.py --corpus dec
"""
from __future__ import annotations

import argparse

from identity_free_volatility import data
from identity_free_volatility.models import chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["dec", "strux"], required=True)
    ap.add_argument("--encoder", default=chunks.ENCODER)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    corpus = data.load_corpus(args.corpus)
    chunks.encode_corpus(corpus, args.encoder, args.batch, args.limit)


if __name__ == "__main__":
    main()
