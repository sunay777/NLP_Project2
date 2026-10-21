"""A1 - Would this archive survive an LLM pretraining pipeline?

Applies LID -> Gopher -> bad-words -> CCNet perplexity to
  hist_xh   historical isiXhosa (raw, or normalised with --normalised)
  hist_en   historical English sections (target language 'eng')
  modern_xh modern isiXhosa control (same filters, same thresholds)
and records every decision. Missing resources are skipped with a warning so
the script runs end-to-end from day one.

Units: by default fixed 150-word windows for every corpus (--unit window), so
length-dependent rules cannot drive the comparison. Modern data is split into
disjoint roles (src/modern.py): Wikipedia trains the CCNet-style reference LM (as
in CCNet); the held-out 'control' half of news + government text is the control,
so the control is never scored in-sample and never seen by the normaliser.

Outputs (results/A1[_norm][_xhstop][_<unit>][_t<thr>]/):
  units.csv.gz                one row per unit with all decisions
  survival_by_corpus.csv      unit- and word-weighted survival per gate
  survival_ci.csv             95% edition-cluster bootstrap CIs
  survival_by_paper_year.csv  historical isiXhosa by paper x year
  survival_by_modern_source.csv
  lid_labels_<model>.csv      what the LID models think each corpus *is*
  removed_examples.csv        units removed only by each gate (qualitative audit)
  badword_hits_hist_xh.csv    which listed words fired on historical isiXhosa
  figures/<name>_cascade.png

Run:  python experiments/a1_pipeline_survival.py                 # raw, windows
      python experiments/a1_pipeline_survival.py --normalised    # link to B
      python experiments/a1_pipeline_survival.py --xh-stopwords  # Gopher ablation
      python experiments/a1_pipeline_survival.py --unit paragraph
"""
from __future__ import annotations

import argparse
import pickle
import warnings

import pandas as pd
from _common import PAPER_COLORS, config, outdir, plt

from src import corpus, modern, stats
from src.filters import badwords, gopher, lid, pipeline
from src.filters.perplexity import PerplexityFilter


def build_units(args) -> pd.DataFrame:
    if args.normalised:
        with open(config.CACHE_DIR / "pages_normalised.pkl", "rb") as f:
            pages = pickle.load(f)
        pages = pages.assign(xh=pages.xh_norm)
    else:
        pages = corpus.load_pages()
    kw = dict(unit=args.unit, min_words=args.min_words, window=args.window)
    xh = corpus.to_units(pages, "xh", **kw).assign(corpus="hist_xh", target="xho")
    en = corpus.to_units(pages, "en", **kw).assign(corpus="hist_en", target="eng")
    parts = [xh, en]
    mod = []
    for i, (src, d) in enumerate(modern.iter_role("control", min_words=args.min_words)):
        chunks = corpus.windows(d, args.window) if args.unit == "window" else [d]
        # every chunk of one modern document shares an `edition` (bootstrap cluster)
        mod += [(src, c, f"modern_{i}") for c in chunks if len(c.split()) >= args.min_words]
    if mod:
        parts.append(pd.DataFrame({"text": [d for _, d, _ in mod], "source": [s for s, _, _ in mod],
                                   "corpus": "modern_xh", "target": "xho", "paper": "modern",
                                   "edition": [e for _, _, e in mod]}))
    else:
        warnings.warn("no modern control corpus found in resources/modern/")
    units = pd.concat(parts, ignore_index=True)
    if args.max_units:  # stratified subsample, same n per corpus
        units = pd.concat([g.sample(min(len(g), args.max_units), random_state=config.SEED)
                           for _, g in units.groupby("corpus")])
    return units.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", choices=["window", "page", "paragraph"], default="window")
    ap.add_argument("--window", type=int, default=150, help="words per unit when --unit window")
    ap.add_argument("--min-words", type=int, default=50)
    ap.add_argument("--max-units", type=int, default=5000, help="per corpus; 0 = all")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--normalised", action="store_true")
    ap.add_argument("--xh-stopwords", action="store_true",
                    help="ablation: Gopher stop-word rule with isiXhosa stop words")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    name = "A1" + "_norm" * args.normalised + "_xhstop" * args.xh_stopwords
    if args.unit != "window":
        name += f"_{args.unit}"
    if args.threshold != 0.5:
        name += f"_t{args.threshold:g}"
    out = outdir(name)

    units = build_units(args)
    print(units.groupby("corpus").size())

    lid_models = []
    for path, name in ((config.LID_NLLB, "nllb"), (config.LID_GLOT, "glotlid")):
        try:
            lid_models.append(lid.FastTextLID(path, name))
        except (FileNotFoundError, ImportError) as e:
            warnings.warn(f"LID {name} skipped: {e}")

    bw_words = badwords.load_lists(config.BADWORDS_DIR, ("en",))
    bw = badwords.BadWordFilter(bw_words) if bw_words else None
    if bw is None:
        warnings.warn("bad-word list missing - filter skipped")

    def as_units(docs):  # reference LM is trained and calibrated on the same unit type it scores
        if args.unit != "window":
            return list(docs)
        return [c for d in docs for c in corpus.windows(d, args.window) if len(c.split()) >= args.min_words]

    ppl = {}
    ref_xh = as_units(d for _, d in modern.iter_role("ppl_ref"))
    if ref_xh:
        ppl["xho"] = PerplexityFilter(ref_xh)
    if config.REF_EN.exists():
        ref_en = as_units(config.REF_EN.read_text(encoding="utf-8").splitlines())
        if ref_xh and len(ref_en) > len(ref_xh):  # same-sized reference LMs for both languages
            ref_en = pd.Series(ref_en).sample(len(ref_xh), random_state=config.SEED).tolist()
        ppl["eng"] = PerplexityFilter(ref_en)
    else:
        warnings.warn(f"{config.REF_EN} missing - no perplexity gate for hist_en (scripts/prepare_reference_en.py)")

    # Each language gets its own stop words: the ablation swaps the English list for
    # isiXhosa on isiXhosa text only (English text keeps Gopher's English list).
    stop = {"eng": gopher.EN_STOPWORDS, "xho": gopher.EN_STOPWORDS}
    if args.xh_stopwords:
        lex = modern.lexicon(role="lexicon")  # never the control or the perplexity reference
        stop["xho"] = frozenset(w for w, _ in lex.most_common(8))
        print("isiXhosa stop words:", sorted(stop["xho"]))

    # Each corpus is judged against its own target language.
    results = []
    for tgt, g in units.groupby("target"):
        results.append(pipeline.run(g, target_lang=tgt, lid_models=lid_models, lid_threshold=args.threshold,
                                    badword_filter=bw, ppl_filter=ppl.get(tgt),
                                    gopher_stopwords=stop[tgt]))
    res = pd.concat(results, ignore_index=True)
    res.drop(columns=["text"]).to_csv(out / "units.csv.gz", index=False)

    surv = pipeline.survival_table(res, by=("corpus",))
    # 95% CIs on end-to-end and per-gate survival, resampling editions (or modern documents).
    ci_rows = []
    for corp, g in res.groupby("corpus"):
        for col in [c for c in g if c.startswith("keep_") or c == "survives"]:
            est, lo, hi = stats.cluster_bootstrap(g, lambda d, c=col: d[c].mean(), n_boot=args.n_boot)
            ci_rows.append({"corpus": corp, "decision": col, "rate": est, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(ci_rows).to_csv(out / "survival_ci.csv", index=False)
    surv.to_csv(out / "survival_by_corpus.csv")
    print(surv.filter(like="_units").round(3).T)
    hx = res[res.corpus == "hist_xh"]
    pipeline.survival_table(hx, by=("paper", "year")).to_csv(out / "survival_by_paper_year.csv")
    if "source" in res:
        pipeline.survival_table(res[res.corpus == "modern_xh"], by=("source",)).to_csv(out / "survival_by_modern_source.csv")
    for m in lid_models:
        (res.groupby("corpus")[f"lid_{m.name}_label"].value_counts(normalize=True)
         .groupby(level=0).head(5).to_csv(out / f"lid_labels_{m.name}.csv"))

    # Qualitative audit: up to 10 historical isiXhosa units removed *only* by each gate.
    keep_cols = [c for c in hx if c.startswith("keep_")]
    ex = []
    for c in keep_cols:
        only = hx[~hx[c] & hx[[k for k in keep_cols if k != c]].all(axis=1)]
        pool = only if len(only) else hx[~hx[c]]
        ex.append(pool.sample(min(10, len(pool)), random_state=config.SEED).assign(
            gate=c, removed_only_by_this_gate=bool(len(only))))
    if ex:
        cols = ["gate", "removed_only_by_this_gate", "unit_id", "paper", "year", "gopher_failed",
                *[c for c in ("badword_hits", "ppl_bpc") if c in hx], "text"]
        pd.concat(ex)[cols].to_csv(out / "removed_examples.csv", index=False)
    if "badword_hits" in hx:
        hits = hx.badword_hits.fillna("").str.split(";").explode()
        hits[hits != ""].value_counts().to_csv(out / "badword_hits_hist_xh.csv")

    # Cascade figure: share of units alive after each stage, per corpus.
    stage_cols = [c for c in res if c.startswith("survive_after_")]
    p = plt()
    fig, ax = p.subplots(figsize=(4.5, 2.6))
    for corp, g in res.groupby("corpus"):
        y = [1.0] + [g[c].mean() for c in stage_cols]
        ax.plot(range(len(y)), y, marker="o", label=corp,
                color={"hist_xh": PAPER_COLORS["IMVO"], "hist_en": PAPER_COLORS["EN"]}.get(corp, PAPER_COLORS["modern"]))
    ax.set_xticks(range(len(stage_cols) + 1), ["all"] + [c.replace("survive_after_", "") for c in stage_cols],
                  rotation=20)
    ax.set_ylabel("fraction surviving")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(config.FIG_DIR / f"{out.name}_cascade.png")
    print("saved to", out)


if __name__ == "__main__":
    main()
