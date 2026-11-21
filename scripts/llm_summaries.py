"""LLM-summary baseline, step 1: summarize every DEC transcript with Claude.

This costs real money. Nothing billable is sent without --yes-spend.

  python scripts/llm_summaries.py estimate --model claude-haiku-5-5   # free, counts tokens on a sample
  python scripts/llm_summaries.py submit   --model claude-haiku-5-5 --limit 20 --yes-spend   # pilot
  python scripts/llm_summaries.py submit   --model claude-haiku-5-5 --yes-spend
  python scripts/llm_summaries.py collect  --model claude-haiku-5-5 --wait

Requests go through the Message Batches API (half price, results within 24 h).
Credentials come from ANTHROPIC_API_KEY, read from the environment or from a
.env file in the repo root (gitignored). Summaries are derived from the
transcripts, so they stay under data/ like the transcripts themselves.

Step 2 is  scripts/run_baselines.py --corpus dec --models summary_mlp --summary-model <model>.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import anthropic
import numpy as np
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from identity_free_volatility import data
from identity_free_volatility.paths import PROCESSED, REPO

# USD per million tokens (input, output): list prices on 2026-11-03. Batch API halves both.
PRICES = {
    "claude-fable-5-2": (8.00, 40.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
BATCH_DISCOUNT = 0.5
MAX_TOKENS = 4000          # summary plus the model's own thinking
CHUNK = 500                # requests per batch (about 30 MB of transcript text)

# Same three perspectives as the "Summarization" strategy of Yu et al. (2025), Fig. 12.
SYSTEM = """You analyze earnings call transcripts for a financial research dataset.

Summarize the call below in at most 400 words, as three short sections:

1. Financial performance: revenue, earnings per share, margins and guidance, \
and how each compares with expectations and with the prior year where the call says so.
2. Management commentary: tone (confident, cautious, defensive), recurring themes, \
and whether analysts' concerns were answered directly or vaguely.
3. Operational updates: segments or regions driving results, new initiatives, \
and challenges or risks that were raised.

Report only what is said in this call. Write plain prose under each numbered \
heading and output nothing before or after the three sections."""


def load_env() -> None:
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def out_dir(model: str):
    path = PROCESSED / "summaries" / "dec" / model
    path.mkdir(parents=True, exist_ok=True)
    return path


def params(model: str, effort: str, transcript: str) -> MessageCreateParamsNonStreaming:
    return MessageCreateParamsNonStreaming(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": f"<transcript>\n{transcript}\n</transcript>"}],
    )


def load_calls() -> pd.DataFrame:
    corpus = data.load_dec()
    return pd.DataFrame({"call_id": corpus.frame["call_id"], "text": data.load_text(corpus)})


def done_ids(model: str) -> set[str]:
    path = out_dir(model) / "summaries.parquet"
    if not path.exists():
        return set()
    got = pd.read_parquet(path)
    return set(got.loc[got["stop_reason"] == "end_turn", "call_id"])


def cost(model: str, input_tokens: float, output_tokens: float) -> float:
    price_in, price_out = PRICES[model]
    return BATCH_DISCOUNT * (input_tokens * price_in + output_tokens * price_out) / 1e6


def estimate(args, client) -> None:
    calls = load_calls()
    sample = calls.sample(min(args.sample, len(calls)), random_state=0)
    tokens = [client.messages.count_tokens(
        model=args.model, system=SYSTEM,
        messages=[{"role": "user", "content": f"<transcript>\n{t}\n</transcript>"}]).input_tokens
        for t in sample["text"]]
    per_char = sum(tokens) / sample["text"].str.len().sum()
    total_in = per_char * calls["text"].str.len().sum()
    total_out = args.assumed_output * len(calls)
    print(f"{args.model}: {len(calls)} calls, about {total_in / 1e6:.1f}M input tokens "
          f"(from {len(sample)} counted calls), assuming {args.assumed_output} output tokens per call")
    print(f"estimated batch cost: ${cost(args.model, total_in, total_out):.2f} "
          f"(input ${cost(args.model, total_in, 0):.2f}, output ${cost(args.model, 0, total_out):.2f})")


def submit(args, client) -> None:
    if not args.yes_spend:
        raise SystemExit("refusing to submit without --yes-spend (run `estimate` first)")
    calls = load_calls()
    todo = calls[~calls["call_id"].isin(done_ids(args.model))]
    if args.limit:
        todo = todo.head(args.limit)
    if todo.empty:
        raise SystemExit("nothing to do: every call already has a summary")
    state_file = out_dir(args.model) / "batches.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else []
    for start in range(0, len(todo), CHUNK):
        part = todo.iloc[start:start + CHUNK]
        batch = client.messages.batches.create(requests=[
            Request(custom_id=row.call_id, params=params(args.model, args.effort, row.text))
            for row in part.itertuples(index=False)])
        state.append({"id": batch.id, "n": len(part), "collected": False})
        state_file.write_text(json.dumps(state, indent=1))  # save after every batch
        print(f"submitted {batch.id}: {len(part)} calls")


def collect(args, client) -> None:
    state_file = out_dir(args.model) / "batches.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else []
    path = out_dir(args.model) / "summaries.parquet"
    rows = pd.read_parquet(path).to_dict("records") if path.exists() else []
    for entry in state:
        if entry["collected"]:
            continue
        batch = client.messages.batches.retrieve(entry["id"])
        while args.wait and batch.processing_status != "ended":
            print(f"{entry['id']}: {batch.processing_status}, "
                  f"{batch.request_counts.processing} still processing")
            time.sleep(60)
            batch = client.messages.batches.retrieve(entry["id"])
        if batch.processing_status != "ended":
            print(f"{entry['id']}: {batch.processing_status}, try again later or pass --wait")
            continue
        for result in client.messages.batches.results(entry["id"]):
            if result.result.type != "succeeded":
                print(f"  {result.custom_id}: {result.result.type}")
                continue
            msg = result.result.message
            # stop_reason can be "refusal" or "max_tokens": keep it so those calls are retried
            text = "".join(b.text for b in msg.content if b.type == "text")
            rows.append({"call_id": result.custom_id, "summary": text,
                         "stop_reason": msg.stop_reason, "model": msg.model,
                         "input_tokens": msg.usage.input_tokens,
                         "output_tokens": msg.usage.output_tokens})
        entry["collected"] = True
        pd.DataFrame(rows).drop_duplicates("call_id", keep="last").to_parquet(path, index=False)
        state_file.write_text(json.dumps(state, indent=1))
    if not rows:
        return
    got = pd.DataFrame(rows).drop_duplicates("call_id", keep="last")
    print(f"{len(got)} summaries on disk, stop reasons: {got.stop_reason.value_counts().to_dict()}")
    print(f"summary length: median {int(np.median(got.summary.str.split().str.len()))} words")
    print(f"billed so far: {got.input_tokens.sum() / 1e6:.2f}M input, "
          f"{got.output_tokens.sum() / 1e6:.2f}M output tokens, "
          f"about ${cost(args.model, got.input_tokens.sum(), got.output_tokens.sum()):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["estimate", "submit", "collect"])
    ap.add_argument("--model", required=True, choices=sorted(PRICES))
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--sample", type=int, default=40, help="estimate: calls to count tokens on")
    ap.add_argument("--assumed-output", type=int, default=1500,
                    help="estimate: output tokens per call, thinking included")
    ap.add_argument("--limit", type=int, default=0, help="submit: only the first N remaining calls")
    ap.add_argument("--yes-spend", action="store_true", help="submit: really send billable requests")
    ap.add_argument("--wait", action="store_true", help="collect: poll until the batches end")
    args = ap.parse_args()

    load_env()
    client = anthropic.Anthropic()
    try:
        {"estimate": estimate, "submit": submit, "collect": collect}[args.command](args, client)
    except anthropic.AuthenticationError:
        raise SystemExit("the API key was rejected: check ANTHROPIC_API_KEY in .env at the repo root")
    except TypeError as exc:  # the SDK raises this when it finds no credentials at all
        if "authentication" not in str(exc):
            raise
        raise SystemExit("no API key found: put ANTHROPIC_API_KEY=... in .env at the repo root")
    except anthropic.RateLimitError as exc:
        raise SystemExit(f"rate limited, retry later: {exc}")
    except anthropic.APIStatusError as exc:
        raise SystemExit(f"API error {exc.status_code}: {exc.message}")
    except anthropic.APIConnectionError as exc:
        raise SystemExit(f"could not reach the API: {exc}")


if __name__ == "__main__":
    main()
