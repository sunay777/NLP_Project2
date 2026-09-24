"""B0 - build the historical -> modern spelling map and a normalised corpus.

Needs modern isiXhosa text in resources/modern/ (scripts/prepare_modern.py).

Outputs (results/B0/):
  normalisation_map.json     historical type -> modern type
  top_changes.csv            most frequent rewrites (audit these by hand!)
  coverage.json              share of historical tokens attested in the modern
                             lexicon before and after normalisation
  data/cache/pages_normalised.pkl   pages with an extra `xh_norm` column

Run:  python experiments/b0_normalise.py [--min-count 3] [--no-join]
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter

import pandas as pd
from _common import config, outdir, save_json

from src import corpus, modern, orthography
from src.text import words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--no-join", action="store_true", help="skip prefix joining (ablation)")
    args = ap.parse_args()
    out = outdir("B0")

    lex = modern.lexicon()
    if not lex:
        raise SystemExit("No modern text in resources/modern/ - run scripts/prepare_modern.py first.")
    df = corpus.load_pages()
    hist = Counter(w for t in df.xh for w in words(t))

    mapping = orthography.build_map(hist, lex, args.min_count, args.ratio)
    save_json(mapping, out / "normalisation_map.json")
    top = sorted(((hist[w], w, m) for w, m in mapping.items()), reverse=True)[:500]
    pd.DataFrame(top, columns=["hist_count", "historical", "modern"]).to_csv(out / "top_changes.csv", index=False)

    df["xh_norm"] = [orthography.normalise_text(t, mapping, join=not args.no_join) for t in df.xh]
    norm = Counter(w for t in df.xh_norm for w in words(t))
    cov = {
        "modern_lexicon_types": len(lex),
        "map_size": len(mapping),
        "coverage_raw": orthography.coverage(hist, lex),
        "coverage_norm": orthography.coverage(norm, lex),
        "tokens_changed_frac": sum(hist[w] for w in mapping) / sum(hist.values()),
        "by_paper": {},
    }
    for p, g in df.groupby("paper"):
        h = Counter(w for t in g.xh for w in words(t))
        n = Counter(w for t in g.xh_norm for w in words(t))
        cov["by_paper"][p] = {"raw": orthography.coverage(h, lex), "norm": orthography.coverage(n, lex)}
    save_json(cov, out / "coverage.json")
    print({k: v for k, v in cov.items() if k != "by_paper"}, cov["by_paper"])

    with open(config.CACHE_DIR / "pages_normalised.pkl", "wb") as f:
        pickle.dump(df, f)


if __name__ == "__main__":
    main()
