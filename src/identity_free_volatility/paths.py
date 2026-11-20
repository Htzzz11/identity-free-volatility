"""Filesystem layout. Everything under data/ is local-only (gitignored)."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
RAW = DATA / "raw"
PRICES = DATA / "prices"
PROCESSED = DATA / "processed"
RESULTS = REPO / "results"

DEC_CSV = RAW / "dec_repo" / "SameCompanySameSignal" / "dataset" / "DEC.csv"
DEC_DRIVE = RAW / "dec_drive"
STRUX_DIR = RAW / "strux" / "data"
