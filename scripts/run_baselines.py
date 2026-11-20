"""Run baselines under the rolling-by-quarter protocol and print the summary.

  .venv/bin/python scripts/run_baselines.py --corpus dec
  .venv/bin/python scripts/run_baselines.py --corpus strux --models pev stpev tfidf_ridge

Models: pev, stpev, random_ticker, random_all, tfidf_ridge, openai_mlp (DEC only,
uses the frozen embeddings released with DEC).
"""
from __future__ import annotations

import argparse
import time

from identity_free_volatility import data, evaluate
from identity_free_volatility.baselines import historical, mlp, tfidf
from identity_free_volatility.paths import RESULTS

ALL = ["pev", "stpev", "random_ticker", "random_all", "tfidf_ridge", "openai_mlp"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["dec", "strux"], required=True)
    ap.add_argument("--models", nargs="+", default=ALL, choices=ALL)
    ap.add_argument("--seeds", nargs="+", type=int, default=[2021, 2022, 2023])
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    corpus = data.load_corpus(args.corpus)
    print(f"{corpus.name}: {len(corpus.frame)} calls, {corpus.frame.ticker.nunique()} tickers, "
          f"{len(corpus.splits)} test quarters ({min(corpus.splits)} to {max(corpus.splits)})")

    for model in [] if args.summary_only else args.models:
        t0 = time.time()
        if model == "openai_mlp" and corpus.name != "dec":
            print("openai_mlp: skipped, embeddings exist for DEC only")
            continue
        print(f"== {model}")
        if model == "pev":
            evaluate.save_preds(corpus, model, 0, historical.pev(corpus))
        elif model == "stpev":
            evaluate.save_preds(corpus, model, 0, historical.stpev(corpus))
        elif model == "tfidf_ridge":
            counts = tfidf.count_matrix(data.load_text(corpus))
            print(f"  count matrix {counts.shape}, {time.time() - t0:.0f}s")
            evaluate.save_preds(corpus, model, 0, tfidf.run(corpus, counts))
        else:
            for seed in args.seeds:
                print(f" seed {seed}")
                if model == "openai_mlp":
                    emb = data.load_dec_embeddings(corpus)
                elif model == "random_ticker":
                    emb = mlp.random_ticker(corpus, seed)
                else:
                    emb = mlp.random_all(corpus, seed)
                evaluate.save_preds(corpus, model, seed, mlp.run(corpus, emb, seed))
        print(f"  done in {time.time() - t0:.0f}s")

    blocks = [("all test quarters", None)]
    if corpus.name == "dec":
        blocks.append(("2021-2023, as in Yu et al. Table 4", (2021, 2022, 2023)))
    else:
        blocks.append(("2021-2024", (2021, 2022, 2023, 2024)))
    lines = []
    for title, years in blocks:
        table = evaluate.summarize(corpus, years)
        lines += [f"### {corpus.name}: {title}", "", table.round(3).to_markdown(index=False), ""]
    text = "\n".join(lines)
    print("\n" + text)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"summary_{corpus.name}.md").write_text(text)


if __name__ == "__main__":
    main()
