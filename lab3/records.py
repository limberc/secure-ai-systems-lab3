"""Run directories: every query and every response, with the settings that produced them.

A run is one command against one target. It writes

    out/runs/<run_id>/run.json             header: command, models, versions, settings, data hash
    out/runs/<run_id>/attempts.jsonl       one record per attempt, prompt and response in full
    out/runs/<run_id>/judgements.<j>.jsonl one verdict per attempt, per judge
    out/runs/<run_id>/labels.jsonl         your hand labels (label.py)

and appends a line to out/runlog.jsonl when it starts and when it finishes. score.py reads
the run log to count how many times you ran the test split; the brief asks you to report
that number, so do not delete the log.
"""

import json
import platform
import sys
import time
from pathlib import Path

from . import data

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
RUNS = OUT / "runs"
RUNLOG = OUT / "runlog.jsonl"


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _append(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class Run:
    def __init__(self, kind, name, target, split, header):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.id = f"{stamp}-{kind}-{name}-{target}-{split}"
        self.dir = RUNS / self.id
        n = 1
        while self.dir.exists():
            n += 1
            self.dir = RUNS / f"{self.id}-{n}"
        self.id = self.dir.name
        self.dir.mkdir(parents=True)
        self.header = {"run_id": self.id, "kind": kind, "name": name, "target": target,
                       "split": split, "argv": sys.argv, "started": _now(),
                       "python": platform.python_version(), "platform": platform.platform(),
                       "data_subset_sha256": data.subset_sha256(), **header}
        self._write_header()
        self.n = 0
        _append(RUNLOG, {"event": "start", "run_id": self.id, "kind": kind, "name": name,
                         "target": target, "split": split, "time": self.header["started"]})

    def _write_header(self):
        (self.dir / "run.json").write_text(json.dumps(self.header, indent=2, ensure_ascii=False) + "\n")

    def add(self, record):
        _append(self.dir / "attempts.jsonl", record)
        self.n += 1

    def close(self):
        self.header.update(finished=_now(), n_records=self.n)
        self._write_header()
        _append(RUNLOG, {"event": "finish", "run_id": self.id, "n_records": self.n,
                         "time": self.header["finished"]})
        return self.dir


def judged_text(record):
    """What judges and label.py read: the decoded reply if the attack decoded it."""
    return record["decoded"] if record.get("decoded") is not None else record["response"]


def resolve(run):
    """Accept a run directory path or a bare run id."""
    p = Path(run)
    if not p.exists():
        p = RUNS / run
    if not (p / "run.json").exists():
        raise SystemExit(f"{run}: not a run directory (no run.json)")
    return p


def read(run_dir):
    run_dir = resolve(run_dir)
    header = json.loads((run_dir / "run.json").read_text())
    recs = []
    if (run_dir / "attempts.jsonl").exists():
        with open(run_dir / "attempts.jsonl", encoding="utf-8") as f:
            recs = [json.loads(line) for line in f]
    return header, recs


def write_judgements(run_dir, judge_name, verdicts, judge_info):
    path = resolve(run_dir) / f"judgements.{judge_name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"judge": judge_name, "info": judge_info, "time": _now()}) + "\n")
        for v in verdicts:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    return path


def read_judgements(run_dir):
    """{judge_name: {record key: verdict}}"""
    out = {}
    for path in sorted(resolve(run_dir).glob("judgements.*.jsonl")):
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        out[lines[0]["judge"]] = {v["key"]: v for v in lines[1:]}
    return out


def read_labels(run_dir):
    path = resolve(run_dir) / "labels.jsonl"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["key"]: r for r in map(json.loads, f)}


def all_runs():
    return sorted(p for p in RUNS.glob("*") if (p / "run.json").exists())


def runlog():
    if not RUNLOG.exists():
        return []
    with open(RUNLOG, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
