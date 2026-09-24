"""B1 - the tokeniser tax: fertility of modern tokenisers on
modern vs historical-raw vs historical-normalised isiXhosa.

Needs: transformers + sentencepiece, B0 run first, modern text present.
Tokenizers download from HuggingFace on first use (do this on your own machine).

Run:  python experiments/b1_tokenizer_fertility.py --units 2000
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
from _common import config, outdir, plt

from src import corpus, modern, stats
from src.text import words
from src.tokenizers_eval import DEFAULT_TOKENIZERS, fertility_stats, load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=2000, help="paragraphs per condition")
    ap.add_argument("--tokenizers", nargs="*", default=list(DEFAULT_TOKENIZERS))
    args = ap.parse_args()
    out = outdir("B1")

    with open(config.CACHE_DIR / "pages_normalised.pkl", "rb") as f:
        pages = pickle.load(f)
    raw = corpus.to_units(pages, "xh", "paragraph")
    # min_words=1 so every raw unit has a normalised twin (prefix joining shortens word counts)
    norm = corpus.to_units(pages.assign(xh=pages.xh_norm), "xh", "paragraph", min_words=1)
    idx = np.random.default_rng(config.SEED).choice(len(raw), size=min(args.units, len(raw)), replace=False)
    conds = {"hist_raw": raw.iloc[idx], "hist_norm": norm.set_index("unit_id").loc[raw.iloc[idx].unit_id].reset_index()}
    mod = [d for _, d in modern.iter_docs()]
    if mod:
        m_idx = np.random.default_rng(config.SEED).choice(len(mod), size=min(args.units, len(mod)), replace=False)
        conds["modern"] = pd.DataFrame({"text": [mod[i] for i in m_idx], "edition": [f"m{i}" for i in m_idx]})

    rows = []
    for tname in args.tokenizers:
        tok = load_tokenizer(DEFAULT_TOKENIZERS.get(tname, tname))
        for cname, units in conds.items():
            per_unit = units.assign(fert=[fertility_stats(tok, words(t, lower=False))["fertility"]
                                          if words(t) else np.nan for t in units.text]).dropna(subset=["fert"])
            est, lo, hi = stats.cluster_bootstrap(per_unit, lambda d: d.fert.mean(), n_boot=500)
            rows.append({"tokenizer": tname, "condition": cname, "fertility": est, "ci_lo": lo, "ci_hi": hi})
            print(rows[-1])
    res = pd.DataFrame(rows)
    res.to_csv(out / "fertility.csv", index=False)

    p = plt()
    fig, ax = p.subplots(figsize=(5, 2.6))
    conds_order = [c for c in ("modern", "hist_norm", "hist_raw") if c in set(res.condition)]
    w = 0.8 / len(conds_order)
    for i, c in enumerate(conds_order):
        r = res[res.condition == c].set_index("tokenizer").loc[args.tokenizers]
        x = np.arange(len(r)) + i * w
        ax.bar(x, r.fertility, w, yerr=[r.fertility - r.ci_lo, r.ci_hi - r.fertility], label=c, capsize=2)
    ax.set_xticks(np.arange(len(args.tokenizers)) + w, args.tokenizers, rotation=15)
    ax.set_ylabel("subword tokens / word")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(config.FIG_DIR / "B1_fertility.png")


if __name__ == "__main__":
    main()
