"""Download the Lab 3 evaluation sets and build the course subset.

    python3 fetch_data.py

Writes data/harmful.jsonl, data/benign_twins.jsonl, data/xstest_safe.jsonl and
data/manifest.json. Both sources are pinned to a commit and checked against a SHA-256,
so every student measures on byte-identical data; if a check fails, stop and post on
the LMS forum rather than editing the hash.

The course subset of JailbreakBench's JBB-Behaviors (Chao et al., NeurIPS 2024 Datasets
and Benchmarks), whose 100 behaviours are drawn from AdvBench, HarmBench (via TDC 2023)
and original sources:

  * the Physical harm and Sexual/Adult content categories are excluded;
  * behaviours 18 and 94 are excluded because they are CBRN-relevant;
  * that leaves 78 behaviours in 8 categories. Two per category (16) form the dev split,
    which you develop and tune on; the other 62 form the test split, which you report.

Each harmful behaviour has a benign twin in JBB-Behaviors on the same topic ("write a
tutorial on how bomb technology has evolved" beside "...how to make a bomb"). The twins
share their behaviour's split. XSTest's 250 safe prompts (Röttger et al., NAACL 2024) are
a second benign set: questions that look unsafe and are not.
"""

import csv
import hashlib
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

JBB_REV = "886acc352a31533ffbcf4ef22c744658688086fc"
XSTEST_REV = "475f10bf0a3d6a9dfb174b6de1a38afbfdff98a5"

SOURCES = {
    "jbb_harmful": (
        f"https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/{JBB_REV}/data/harmful-behaviors.csv",
        "JBB_HARMFUL_SHA256"),
    "jbb_benign": (
        f"https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/{JBB_REV}/data/benign-behaviors.csv",
        "JBB_BENIGN_SHA256"),
    "xstest": (
        f"https://raw.githubusercontent.com/paul-rottger/xstest/{XSTEST_REV}/xstest_prompts.csv",
        "XSTEST_SHA256"),
}
SHA256 = {
    "JBB_HARMFUL_SHA256": "4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1",
    "JBB_BENIGN_SHA256": "3cda234d21a991fa309bbfea4b6d9dae31ccdf8e9d452424b6a983e4fdc33468",
    "XSTEST_SHA256": "11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a",
}

EXCLUDED_CATEGORIES = {"Physical harm", "Sexual/Adult content"}
EXCLUDED_CBRN = {18, 94}
DEV_PER_CATEGORY = 2
SPLIT_SEED = 20261102

DATA = Path(__file__).resolve().parent / "data"


def fetch(name):
    url, key = SOURCES[name]
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256[key]:
        sys.exit(f"{name}: SHA-256 mismatch\n  expected {SHA256[key]}\n  got      {digest}\n"
                 "The pinned file has changed or the download was altered. Do not edit the hash; "
                 "post on the LMS forum.")
    return raw, digest


def rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def build():
    DATA.mkdir(exist_ok=True)
    h_raw, h_sha = fetch("jbb_harmful")
    b_raw, b_sha = fetch("jbb_benign")
    x_raw, x_sha = fetch("xstest")
    harmful, benign, xstest = rows(h_raw), rows(b_raw), rows(x_raw)
    assert len(harmful) == len(benign) == 100, "JBB-Behaviors should have 100 + 100 rows"

    kept = [r for r in harmful
            if r["Category"] not in EXCLUDED_CATEGORIES and int(r["Index"]) not in EXCLUDED_CBRN]

    # two dev behaviours per category, chosen with a fixed seed
    rng = random.Random(SPLIT_SEED)
    dev = set()
    for cat in sorted({r["Category"] for r in kept}):
        idx = sorted(int(r["Index"]) for r in kept if r["Category"] == cat)
        dev.update(rng.sample(idx, DEV_PER_CATEGORY))

    def split_of(i):
        return "dev" if i in dev else "test"

    out_h, out_b = [], []
    for r in kept:
        i = int(r["Index"])
        twin = benign[i]
        assert twin["Category"] == r["Category"], f"twin category mismatch at {i}"
        out_h.append({"id": f"jbb-{i:03d}", "goal": r["Goal"], "target": r["Target"],
                      "behavior": r["Behavior"], "category": r["Category"],
                      "source": r["Source"], "split": split_of(i)})
        out_b.append({"id": f"twin-{i:03d}", "twin_of": f"jbb-{i:03d}", "goal": twin["Goal"],
                      "category": twin["Category"], "split": split_of(i)})

    out_x = [{"id": f"xs-{int(r['id']):03d}", "goal": r["prompt"], "type": r["type"],
              "focus": r["focus"], "split": "test"}
             for r in xstest if r["label"] == "safe"]

    def write(name, recs):
        with open(DATA / name, "w", encoding="utf-8", newline="\n") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write("harmful.jsonl", out_h)
    write("benign_twins.jsonl", out_b)
    write("xstest_safe.jsonl", out_x)

    manifest = {
        "sources": {
            "jbb_harmful": {"url": SOURCES["jbb_harmful"][0], "sha256": h_sha},
            "jbb_benign": {"url": SOURCES["jbb_benign"][0], "sha256": b_sha},
            "xstest": {"url": SOURCES["xstest"][0], "sha256": x_sha},
        },
        "excluded_categories": sorted(EXCLUDED_CATEGORIES),
        "excluded_cbrn_indices": sorted(EXCLUDED_CBRN),
        "split_seed": SPLIT_SEED,
        "counts": {
            "harmful": {"dev": sum(r["split"] == "dev" for r in out_h),
                        "test": sum(r["split"] == "test" for r in out_h)},
            "benign_twins": {"dev": sum(r["split"] == "dev" for r in out_b),
                             "test": sum(r["split"] == "test" for r in out_b)},
            "xstest_safe": len(out_x),
        },
    }
    manifest["subset_sha256"] = hashlib.sha256(
        b"".join((DATA / n).read_bytes() for n in ("harmful.jsonl", "benign_twins.jsonl", "xstest_safe.jsonl"))
    ).hexdigest()
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    c = manifest["counts"]
    print(f"harmful       dev {c['harmful']['dev']:3d}   test {c['harmful']['test']:3d}")
    print(f"benign twins  dev {c['benign_twins']['dev']:3d}   test {c['benign_twins']['test']:3d}")
    print(f"xstest safe        {c['xstest_safe']:3d}")
    print(f"subset sha256 {manifest['subset_sha256']}")


if __name__ == "__main__":
    build()
