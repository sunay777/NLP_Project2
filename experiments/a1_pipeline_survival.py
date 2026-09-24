"""A1 - Would this archive survive an LLM pretraining pipeline?

Applies LID -> Gopher -> bad-words -> CCNet perplexity to
  hist_xh   historical isiXhosa (raw, or normalised with --normalised)
  hist_en   historical English sections (target language 'eng')
  modern_xh modern isiXhosa control (same filters, same thresholds)
and records every decision. Missing resources are skipped with a warning so
the script runs end-to-end from day one.

Outputs (results/A1[_norm]/):
  units.csv.gz            one row per unit with all decisions
  survival_by_corpus.csv
  survival_by_paper_year.csv
  lid_labels.csv          what the LID models think historical isiXhosa *is*
  figures/A1_cascade.png

Run:  python experiments/a1_pipeline_survival.py --unit paragraph --max-units 5000
      python experiments/a1_pipeline_survival.py --normalised      # link to B
      python experiments/a1_pipeline_survival.py --threshold 0.65  # sweep this
"""
from __future__ import annotations

import argparse
import pickle
import warnings

import pandas as pd
from _common import PAPER_COLORS, config, outdir, plt

from src import corpus, modern
from src.filters import badwords, gopher, lid, pipeline
from src.filters.perplexity import PerplexityFilter


def build_units(args) -> pd.DataFrame:
    if args.normalised:
        with open(config.CACHE_DIR / "pages_normalised.pkl", "rb") as f:
            pages = pickle.load(f)
        pages = pages.assign(xh=pages.xh_norm)
    else:
        pages = corpus.load_pages()
    xh = corpus.to_units(pages, "xh", args.unit, args.min_words).assign(corpus="hist_xh", target="xho")
    en = corpus.to_units(pages, "en", args.unit, args.min_words).assign(corpus="hist_en", target="eng")
    parts = [xh, en]
    mod = [(src, d) for src, d in modern.iter_docs(min_words=args.min_words)]
    if mod:
        parts.append(pd.DataFrame({"text": [d for _, d in mod], "source": [s for s, _ in mod],
                                   "corpus": "modern_xh", "target": "xho", "paper": "modern",
                                   "edition": [f"modern_{i}" for i in range(len(mod))]}))
    else:
        warnings.warn("no modern control corpus found in resources/modern/")
    units = pd.concat(parts, ignore_index=True)
    if args.max_units:  # stratified subsample, same n per corpus
        units = pd.concat([g.sample(min(len(g), args.max_units), random_state=config.SEED)
                           for _, g in units.groupby("corpus")])
    return units.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", choices=["page", "paragraph"], default="paragraph")
    ap.add_argument("--min-words", type=int, default=20)
    ap.add_argument("--max-units", type=int, default=5000, help="per corpus; 0 = all")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--normalised", action="store_true")
    ap.add_argument("--xh-stopwords", action="store_true",
                    help="ablation: Gopher stop-word rule with isiXhosa stop words")
    args = ap.parse_args()
    out = outdir("A1_norm" if args.normalised else "A1")

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

    ref = [d for _, d in modern.iter_docs()]
    ppl = PerplexityFilter(ref) if ref else None

    stop = gopher.EN_STOPWORDS
    if args.xh_stopwords:
        lex = modern.lexicon()
        stop = frozenset(w for w, _ in lex.most_common(8))
        print("isiXhosa stop words:", sorted(stop))

    # Each corpus is judged against its own target language.
    results = []
    for tgt, g in units.groupby("target"):
        results.append(pipeline.run(g, target_lang=tgt, lid_models=lid_models, lid_threshold=args.threshold,
                                    badword_filter=bw, ppl_filter=ppl if tgt == "xho" else None,
                                    gopher_stopwords=stop))
    res = pd.concat(results, ignore_index=True)
    res.drop(columns=["text"]).to_csv(out / "units.csv.gz", index=False)

    surv = pipeline.survival_table(res, by=("corpus",))
    surv.to_csv(out / "survival_by_corpus.csv")
    print(surv.filter(like="_units").round(3).T)
    hx = res[res.corpus == "hist_xh"]
    pipeline.survival_table(hx, by=("paper", "year")).to_csv(out / "survival_by_paper_year.csv")
    for m in lid_models:
        (res.groupby("corpus")[f"lid_{m.name}_label"].value_counts(normalize=True)
         .groupby(level=0).head(5).to_csv(out / f"lid_labels_{m.name}.csv"))

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
