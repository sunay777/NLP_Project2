"""B2 - Controlled 2 x k experiment: does normalisation and/or modern data help
model historical isiXhosa?

Factors
  spelling    : raw | normalised (historical train AND test text)
  modern_frac : share of a FIXED training-character budget taken from modern
                isiXhosa (0, .25, .5, .75, 1). Fixing the budget separates
                "modern data helps" from "more data helps". frac = 1 is a purely
                modern model reading 1880s text: the direct penalty.
Model: character KN n-gram (src/lm.py). Order and discount are tuned per
condition on the validation split; the test split is touched once.

Metrics on the temporal test split (later editions of both papers), with
edition-cluster bootstrap CIs:
  bits per ORIGINAL word  (total bits / words in the raw text)
  bits per ORIGINAL letter (total bits / letters in the raw text)
Both hold the content fixed, so raw and normalised are comparable (normalising
adds 'h's and joins prefixes, changing character and word counts). Each model is
also scored on held-out modern documents (never used for training), so that the
gap   hist_raw - modern   splits into   orthography (raw - norm)  +  residual.

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


def letters(t: str) -> int:
    return sum(len(w) for w in words(t))


def run_condition(cond, ctx) -> dict:
    """Tune (order, discount) on val, then score the test sets once. Independent per condition."""
    spelling, frac = cond
    args, train, val, test = ctx["args"], ctx["train"], ctx["val"], ctx["test"]
    col = "xh" if spelling == "raw" else "xh_norm"
    rng = np.random.default_rng(config.SEED)  # same draw for every condition
    tr = (take_chars(train[col], int(args.budget * (1 - frac)), rng)
          + take_chars(ctx["mod_train"], int(args.budget * frac), rng))
    best = None
    for order in args.orders:
        lm = CharKN(order).fit(tr)
        for d in args.discounts:  # discount only enters prob(), so no refit needed
            lm.d = d
            v = -sum(lm.logprob2(x) for x in val[col]) / ctx["val_nl"]  # val bits / raw letter
            if best is None or v < best[0]:
                best = (v, order, d, lm)
    v, order, d, lm = best
    lm.d = d
    row = {"spelling": spelling, "modern_frac": frac, "order": order, "discount": d, "val_bpl": v}
    for name, df, c in (("hist", test, col), ("modern", ctx["mod_test"], "text")):
        t = df.assign(bits=[-lm.logprob2(x) for x in df[c]])
        for metric, den in (("bpw", "nw"), ("bpl", "nl")):
            est, lo, hi = stats.cluster_bootstrap(t, lambda g, den=den: g.bits.sum() / g[den].sum(),
                                                  n_boot=args.n_boot)
            row.update({f"{name}_{metric}": est, f"{name}_{metric}_lo": lo, f"{name}_{metric}_hi": hi})
        if name == "hist":
            row.update({f"hist_bpw_{p}": g.bits.sum() / g.nw.sum() for p, g in t.groupby("paper")})
    print({k: round(float(x), 3) if isinstance(x, (float, np.floating)) else x
           for k, x in row.items() if not k.endswith(("_lo", "_hi"))}, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=3_000_000, help="training characters per condition")
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--orders", type=int, nargs="*", default=[3, 4, 5, 6])
    ap.add_argument("--discounts", type=float, nargs="*", default=[0.6, 0.75, 0.9])
    ap.add_argument("--eval-units", type=int, default=300, help="pages sampled from val/test")
    ap.add_argument("--modern-test-frac", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--val-units", type=int, default=100, help="val pages used for tuning")
    ap.add_argument("--jobs", type=int, default=5, help="conditions run in parallel (~0.5 GB RAM each)")
    args = ap.parse_args()
    out = outdir("B2")

    with open(config.CACHE_DIR / "pages_normalised.pkl", "rb") as f:
        pages = pickle.load(f)
    pages["split"] = splits.assign(pages, "temporal")
    pages = pages[pages.n_xh_words >= 50]
    mod = [d for _, d in modern.iter_docs()]
    if not mod:
        raise SystemExit("needs modern text in resources/modern/")
    perm = np.random.default_rng(config.SEED).permutation(len(mod))
    n_mt = int(len(mod) * args.modern_test_frac)
    mod_test = pd.DataFrame({"text": [mod[i] for i in perm[:n_mt]], "edition": [f"m{i}" for i in perm[:n_mt]]})
    mod_train = [mod[i] for i in perm[n_mt:]]

    val = pages[pages.split == "val"].sample(min(args.val_units, (pages.split == "val").sum()), random_state=0)
    test = pages[pages.split == "test"].sample(min(args.eval_units, (pages.split == "test").sum()), random_state=0)
    train = pages[pages.split == "train"]
    test = test.assign(nw=test.xh.map(lambda t: len(words(t))), nl=test.xh.map(letters))
    mod_test = mod_test.assign(nw=mod_test.text.map(lambda t: len(words(t))), nl=mod_test.text.map(letters))
    val_nl = val.xh.map(letters).sum()

    ctx = dict(train=train, val=val, test=test, mod_train=mod_train, mod_test=mod_test, val_nl=val_nl, args=args)
    conds = list(itertools.product(("raw", "norm"), args.fracs))
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(args.jobs) as ex:
            rows = list(ex.map(run_condition, conds, [ctx] * len(conds)))
    else:
        rows = [run_condition(c, ctx) for c in conds]
    res = pd.DataFrame(rows)
    res.to_csv(out / "lm_2x2.csv", index=False)

    p = plt()
    fig, ax = p.subplots(figsize=(4, 2.6))
    for spelling, g in res.groupby("spelling"):
        ax.errorbar(g.modern_frac, g.hist_bpw, yerr=[g.hist_bpw - g.hist_bpw_lo, g.hist_bpw_hi - g.hist_bpw],
                    marker="o", capsize=2, label=f"historical test, {spelling}")
    ax.set_xlabel("fraction of training budget from modern isiXhosa")
    ax.set_ylabel("test bits / original word")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(config.FIG_DIR / "B2_lm_2x2.png")
    print("saved to", out)


if __name__ == "__main__":
    main()
