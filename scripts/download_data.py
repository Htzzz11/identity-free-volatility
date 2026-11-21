"""Fetch the raw data into data/raw/ (gitignored; do not commit or re-host it).

  STRUX transcripts   huggingface.co/datasets/BUILDERlym/STRUX-Transcripts
  DEC labels + code   github.com/piqueyd/Same-Company-Same-Signal
  DEC transcripts and frozen OpenAI embeddings: the authors' Google Drive

Run:  .venv/bin/python scripts/download_data.py
"""
from __future__ import annotations

import subprocess

import gdown
from huggingface_hub import snapshot_download

from identity_free_volatility.paths import DEC_DRIVE, RAW, STRUX_DIR

# File ids inside the Drive folder linked from the DEC README. We skip the
# 1,800 per-call .txt files there because DEC.json holds the same text.
DEC_DRIVE_FILES = {
    "DEC.json": "1FI-Yboi0xiWcV_uumv8SLrn4ew35nZb6",
    "Embeddings/openai/DEC.npz": "15OAJ-Lp14DvvcCOX6ygWXkqQUVupyJXo",
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    if not any(STRUX_DIR.glob("full-*.parquet")):
        snapshot_download("BUILDERlym/STRUX-Transcripts", repo_type="dataset",
                          local_dir=STRUX_DIR.parent)
    print("STRUX ok")

    repo = RAW / "dec_repo"
    if not repo.exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/piqueyd/Same-Company-Same-Signal.git", str(repo)],
                       check=True)
    print("DEC repo ok")

    for rel, file_id in DEC_DRIVE_FILES.items():
        dest = DEC_DRIVE / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(id=file_id, output=str(dest), quiet=False)
    print("DEC drive files ok")


if __name__ == "__main__":
    main()
