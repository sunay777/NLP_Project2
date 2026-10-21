"""B0 gold set - sample word types for hand-checking and score the normaliser.

Two modes:

  --make   write data/gold_candidates.tsv: ~200 historical types stratified by
           frequency band (and by whether the normaliser changed them), with the
           system output pre-filled. A checker (ideally an isiXhosa speaker) fills
           `modern` with the correct modern spelling (or copies `historical` if no
           change is needed) and puts their initials in `checked_by`. Then append
           the checked rows to data/gold_normalisation.tsv.
  default  evaluate the map against every row of data/gold_normalisation.tsv whose
           `checked_by` is filled in (rows marked TODO / UNVERIFIED are ignored).

Reported (results/B0/gold_eval.json):
  accuracy   system output == gold, over all checked types
  precision  of the rewrites the system made, share that are correct
  recall     of the types that need a change, share the system fixed correctly
  (each with a 95% Wilson interval; also reported without words marked "not sure")
Returned volunteer forms are merged in first with scripts/collect_gold_forms.py.

Run:  python experiments/b0_gold.py --make
      python experiments/b0_gold.py
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
from _common import config, outdir, save_json

from src import corpus
from src.text import words

GOLD = config.ROOT / "data" / "gold_normalisation.tsv"
CANDIDATES = config.ROOT / "data" / "gold_candidates.tsv"
UNCHECKED = {"", "todo", "unverified", "nan"}
BANDS = [(1000, None), (100, 1000), (10, 100), (3, 10)]


def make(mapping: dict, n_per_cell: int, seed: int) -> None:
    hist = Counter(w for t in corpus.load_pages().xh for w in words(t))
    rng = np.random.default_rng(seed)
    rows = []
    for lo, hi in BANDS:
        band = [w for w, c in hist.items() if c >= lo and (hi is None or c < hi) and len(w) > 1]
        for changed in (True, False):
            pool = sorted(w for w in band if (w in mapping) == changed)
            for w in rng.choice(pool, size=min(n_per_cell, len(pool)), replace=False):
                rows.append({"historical": w, "modern": "", "checked_by": "TODO",
                             "system": mapping.get(w, w), "hist_count": hist[w],
                             "band": f"{lo}+" if hi is None else f"{lo}-{hi - 1}",
                             "system_changed": changed})
    pd.DataFrame(rows).to_csv(CANDIDATES, sep="\t", index=False)
    print(f"wrote {len(rows)} rows to {CANDIDATES}")


def evaluate(mapping: dict) -> dict:
    g = pd.read_csv(GOLD, sep="\t", dtype=str).fillna("")
    g = g[~g.checked_by.str.strip().str.lower().isin(UNCHECKED)]
    if g.empty:
        print(f"no checked rows in {GOLD}; fill `checked_by` after verifying entries")
        return {}
    res = score(g, mapping)
    # Robustness: the same scores without the words the checker marked as uncertain.
    sure = g[~g.get("note", pd.Series("", index=g.index)).fillna("").str.lower().str.contains("not sure|unsure|\\?")]
    if 0 < len(sure) < len(g):
        res["excluding_unsure"] = {k: v for k, v in score(sure, mapping).items() if k != "errors"}
    res["checkers"] = g.checked_by.value_counts().to_dict()
    return res


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for a proportion k/n (sound for small n)."""
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [round(mid - half, 4), round(mid + half, 4)]


def score(g: pd.DataFrame, mapping: dict) -> dict:
    hist, gold = g.historical.str.lower(), g.modern.str.lower()
    system = hist.map(lambda w: mapping.get(w, w))
    changed, needs, right = system != hist, gold != hist, system == gold
    return {
        "n_checked": int(len(g)),
        "accuracy": float(right.mean()), "accuracy_ci": wilson(int(right.sum()), len(g)),
        "precision": float(right[changed].mean()) if changed.any() else None,
        "precision_ci": wilson(int(right[changed].sum()), int(changed.sum())),
        "recall": float(right[needs].mean()) if needs.any() else None,
        "recall_ci": wilson(int(right[needs].sum()), int(needs.sum())),
        # over-normalisation: words that should stay the same but the system changed
        "over_normalised": int((changed & ~needs).sum()),
        "n_system_changed": int(changed.sum()),
        "n_gold_needs_change": int(needs.sum()),
        "errors": [{"historical": h, "gold": o, "system": s}
                   for h, o, s in zip(hist, gold, system) if s != o],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--n-per-cell", type=int, default=25, help="types per (frequency band x changed?) cell")
    args = ap.parse_args()
    mapping = json.loads((config.RESULTS_DIR / "B0" / "normalisation_map.json").read_text())
    if args.make:
        make(mapping, args.n_per_cell, config.SEED)
        return
    res = evaluate(mapping)
    if res:
        save_json(res, outdir("B0") / "gold_eval.json")
        print({k: v for k, v in res.items() if k != "errors"})


if __name__ == "__main__":
    main()
