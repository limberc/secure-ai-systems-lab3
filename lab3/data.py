"""Loading the evaluation sets written by fetch_data.py."""

import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

FILES = {"harmful": "harmful.jsonl", "twins": "benign_twins.jsonl", "xstest": "xstest_safe.jsonl"}


def load(name, split=None, limit=None):
    """Records from one set. `split` is "dev", "test" or None for both."""
    path = DATA / FILES[name]
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run `python3 fetch_data.py` first.")
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    if split is not None:
        recs = [r for r in recs if r["split"] == split]
    return recs[:limit] if limit else recs


def subset_sha256():
    """SHA-256 of the three data files as they are on disk now -- recorded in every run
    header, so an edited file shows up as a different hash."""
    paths = [DATA / FILES[n] for n in ("harmful", "twins", "xstest")]
    if not all(p.exists() for p in paths):
        return None
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()
