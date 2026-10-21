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

The historical budget is counted in RAW characters for both spellings, so the
raw and normalised models see exactly the same pages (normalising adds letters,
so counting normalised characters would give that model less content).
H4 is tested directly: for each modern share f, the interaction
  [bpl_norm(0) - bpl_norm(f)] - [bpl_raw(0) - bpl_raw(f)]
(how much more modern data helps after normalisation) gets a paired
edition-cluster bootstrap CI over the same test pages (results/B2/h4_interaction.csv).

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


def take_chars(texts, budget: int, rng, lengths=None) -> list[str]:
    """Random documents until `budget` characters; `lengths` (default: len of each
    text) lets the budget be counted on the raw text when training on normalised text."""
    texts = list(texts)
    lengths = [len(t) for t in texts] if lengths is None else list(lengths)
    order = rng.permutation(len(texts))
    out, n = [], 0
    for i in order:
        if n >= budget:
            break
        out.append(texts[i])
        n += lengths[i]
    return out


def letters(t: str) -> int:
    return sum(len(w) for w in words(t))


def run_condition(cond, ctx) -> dict:
    """Tune (order, discount) on val, then score the test sets once. Independent per condition."""
    spelling, frac = cond
    args, train, val, test = ctx["args"], ctx["train"], ctx["val"], ctx["test"]
    col = "xh" if spelling == "raw" else "xh_norm"
    rng = np.random.default_rng(config.SEED)  # same draw for every condition
    tr = (take_chars(train[col], int(args.budget * (1 - frac)), rng, lengths=train.xh.str.len())
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
            row["_hist_bits"] = t.bits.to_numpy()  # per test page, for the paired H4 bootstrap
    print({k: round(float(x), 3) if isinstance(x, (float, np.floating)) else x
           for k, x in row.items() if not k.endswith(("_lo", "_hi")) and not k.startswith("_")}, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=3_000_000, help="training characters per condition")
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    # Validation keeps improving slowly up to order 12 (2.77 -> 2.73 bits/letter from 8 to 12);
    # test conclusions are the same at orders 8 and 12.
    ap.add_argument("--orders", type=int, nargs="*", default=list(range(3, 13)))
    ap.add_argument("--discounts", type=float, nargs="*", default=[0.6, 0.75, 0.9, 0.95])
    ap.add_argument("--eval-units", type=int, default=300, help="pages sampled from val/test")
    ap.add_argument("--modern-test-docs", type=int, default=400, help="held-out modern docs scored")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--val-units", type=int, default=100, help="val pages used for tuning")
    ap.add_argument("--jobs", type=int, default=3, help="conditions run in parallel (~1.5 GB RAM each at order 12)")
    args = ap.parse_args()
    out = outdir("B2")

    with open(config.CACHE_DIR / "pages_normalised.pkl", "rb") as f:
        pages = pickle.load(f)
    pages["split"] = splits.assign(pages, "temporal")
    pages = pages[pages.n_xh_words >= 50]
    # Modern training text (Wikipedia + lexicon half) and the held-out modern test set
    # (control half) are disjoint, and the test set was never seen by the normaliser.
    mod_train = [d for _, d in modern.iter_role("lm_train")]
    ctrl = [d for _, d in modern.iter_role("control")]
    if not mod_train or not ctrl:
        raise SystemExit("needs modern text in resources/modern/")
    pick = np.random.default_rng(config.SEED).permutation(len(ctrl))[:args.modern_test_docs]
    mod_test = pd.DataFrame({"text": [ctrl[i] for i in pick], "edition": [f"m{i}" for i in pick]})

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
    bits = {(r["spelling"], r["modern_frac"]): r.pop("_hist_bits") for r in rows}
    res = pd.DataFrame(rows)
    res.to_csv(out / "lm_2x2.csv", index=False)

    # H4: does modern data help more after normalisation? Paired over the same test pages.
    h4 = []
    for f in [f for f in args.fracs if f > 0]:
        d = test[["edition", "nl"]].assign(r0=bits[("raw", 0.0)], rf=bits[("raw", f)],
                                           n0=bits[("norm", 0.0)], nf=bits[("norm", f)])

        def inter(g):
            bpl = {c: g[c].sum() / g.nl.sum() for c in ("r0", "rf", "n0", "nf")}
            return (bpl["n0"] - bpl["nf"]) - (bpl["r0"] - bpl["rf"])

        est, lo, hi = stats.cluster_bootstrap(d, inter, n_boot=args.n_boot)
        h4.append({"modern_frac": f, "interaction_bpl": est, "ci_lo": lo, "ci_hi": hi})
    if h4:
        pd.DataFrame(h4).to_csv(out / "h4_interaction.csv", index=False)
        print(pd.DataFrame(h4).round(3))

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
