"""B2 - Controlled 2 x k experiment: does normalisation and/or modern data help
model historical isiXhosa?

Factors
  spelling    : raw | normalised (historical train AND test text)
  modern_frac : share of a FIXED training-character budget taken from modern
                isiXhosa (0, .25, .5, .75). Fixing the budget separates
                "modern data helps" from "more data helps".
Model: character KN n-gram (src/lm.py). Hyper-parameters (order, discount) are
tuned per condition on the validation split; the test split is touched once.
Metric: bits per ORIGINAL word on the temporal test split (later editions of
both papers), with edition-cluster bootstrap CIs.

Run:  python experiments/b2_lm_2x2.py --budget 3000000
"""
from __future__ import annotations

import argparse
import itertools
import pickle

import numpy as np
import pandas as pd
from _common import config, outdir, plt

from src import modern, splits, stats
from src.lm import CharKN
from src.text import words


def take_chars(texts, budget: int, rng) -> list[str]:
    texts = list(texts)
    order = rng.permutation(len(texts))
    out, n = [], 0
    for i in order:
        if n >= budget:
            break
        out.append(texts[i])
        n += len(texts[i])
    return out


def per_word_bits(lm, texts, raw_texts) -> float:
    bits = -sum(lm.logprob2(t) for t in texts)
    return bits / max(1, sum(len(words(t)) for t in raw_texts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=3_000_000, help="training characters per condition")
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75])
    ap.add_argument("--orders", type=int, nargs="*", default=[3, 4, 5, 6])
    ap.add_argument("--discounts", type=float, nargs="*", default=[0.6, 0.75, 0.9])
    ap.add_argument("--eval-units", type=int, default=400, help="pages sampled from val/test")
    args = ap.parse_args()
    out = outdir("B2")
    rng = np.random.default_rng(config.SEED)

    with open(config.CACHE_DIR / "pages_normalised.pkl", "rb") as f:
        pages = pickle.load(f)
    pages["split"] = splits.assign(pages, "temporal")
    pages = pages[pages.n_xh_words >= 50]
    mod = [d for _, d in modern.iter_docs()]
    if not mod:
        raise SystemExit("needs modern text in resources/modern/")

    val = pages[pages.split == "val"].sample(min(args.eval_units, (pages.split == "val").sum()), random_state=0)
    test = pages[pages.split == "test"].sample(min(args.eval_units, (pages.split == "test").sum()), random_state=0)
    train = pages[pages.split == "train"]

    rows = []
    for spelling, frac in itertools.product(("raw", "norm"), args.fracs):
        col = "xh" if spelling == "raw" else "xh_norm"
        tr = take_chars(train[col], int(args.budget * (1 - frac)), rng) + take_chars(mod, int(args.budget * frac), rng)
        best = None
        for order, d in itertools.product(args.orders, args.discounts):
            lm = CharKN(order, d).fit(tr)
            v = per_word_bits(lm, val[col], val.xh)
            if best is None or v < best[0]:
                best = (v, order, d, lm)
        v, order, d, lm = best
        t = test.assign(bits=[-lm.logprob2(x) for x in test[col]], nw=[len(words(x)) for x in test.xh])
        est, lo, hi = stats.cluster_bootstrap(t, lambda g: g.bits.sum() / g.nw.sum(), n_boot=1000)
        by_paper = {p: g.bits.sum() / g.nw.sum() for p, g in t.groupby("paper")}
        rows.append({"spelling": spelling, "modern_frac": frac, "order": order, "discount": d,
                     "val_bpw": v, "test_bpw": est, "ci_lo": lo, "ci_hi": hi,
                     **{f"test_bpw_{p}": b for p, b in by_paper.items()}})
        print(rows[-1])
    res = pd.DataFrame(rows)
    res.to_csv(out / "lm_2x2.csv", index=False)

    p = plt()
    fig, ax = p.subplots(figsize=(4, 2.6))
    for spelling, g in res.groupby("spelling"):
        ax.errorbar(g.modern_frac, g.test_bpw, yerr=[g.test_bpw - g.ci_lo, g.ci_hi - g.test_bpw],
                    marker="o", capsize=2, label=spelling)
    ax.set_xlabel("fraction of training budget from modern isiXhosa")
    ax.set_ylabel("test bits / original word")
    ax.legend(frameon=False)
    fig.savefig(config.FIG_DIR / "B2_lm_2x2.png")


if __name__ == "__main__":
    main()
