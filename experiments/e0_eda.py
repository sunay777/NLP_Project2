"""Exploratory data analysis (brief step 2).

Outputs (results/eda/):
  corpus_by_paper_year.csv     pages / words / vocab per paper-year
  orthography_evidence.csv     historical vs modern spellings of frequent words
  lexicon_rates_overlap.csv    political / cultural vocabulary per 100k words, 1884-88
  figures/eda_words_per_year.png

Run:  python experiments/e0_eda.py
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
from _common import PAPER_COLORS, config, outdir, plt

from src import corpus, stats
from src.text import ocr_noise, words

# Historical spelling -> modern spelling (aspiration marked). Extend this list;
# it doubles as a seed for the gold normalisation set.
ORTHO_PAIRS = [("kuti", "kuthi"), ("ukuti", "ukuthi"), ("pakati", "phakathi"),
               ("umteto", "umthetho"), ("tixo", "thixo"), ("rulumente", "rhulumente"),
               ("into", "into"), ("kwa", "kwa")]
# Vocabulary for the Isigidimi vs Imvo comparison (C). Glosses should be checked.
LEXICON = {
    "vote": ["ivoti", "voti", "amavoti"],
    "government": ["rulumente", "urulumente"],
    "law": ["umteto", "imiteto"],
    "land": ["umhlaba", "ilizwe"],
    "rights": ["ilungelo", "amalungelo"],
    "freedom": ["inkululeko"],
    "God": ["tixo", "utixo"],
    "white people": ["abelungu", "umlungu"],
}


def main():
    out = outdir("eda")
    df = corpus.load_pages()
    df["ocr_noise"] = df.xh.map(ocr_noise)

    # 1. Size table.
    vocab = {}
    counts = {"ISIG": Counter(), "IMVO": Counter()}
    for r in df.itertuples():
        w = words(r.xh)
        counts[r.paper].update(w)
        vocab.setdefault((r.paper, r.year), set()).update(w)
    tab = df.groupby(["paper", "year"]).agg(pages=("page_id", "size"), editions=("edition", "nunique"),
                                             xh_words=("n_xh_words", "sum"), en_words=("n_en_words", "sum"),
                                             ocr_noise=("ocr_noise", "mean"))
    tab["xh_vocab"] = [len(vocab[k]) for k in tab.index]
    tab.to_csv(out / "corpus_by_paper_year.csv")
    print(tab.groupby("paper")[["pages", "editions", "xh_words", "en_words"]].sum())

    # 2. Orthography evidence.
    rows = []
    for old, new in ORTHO_PAIRS:
        rows.append({"historical": old, "modern": new,
                     **{f"{p}_{old}": counts[p][old] for p in counts},
                     **{f"{p}_{new}": counts[p][new] for p in counts}})
    pd.DataFrame(rows).to_csv(out / "orthography_evidence.csv", index=False)

    # 3. Lexicon rates in the overlap window (both papers in print).
    lo, hi = config.OVERLAP_YEARS
    ov = df[df.year.between(lo, hi)]
    rows = []
    for paper, g in ov.groupby("paper"):
        c = Counter(w for t in g.xh for w in words(t))
        tot = sum(c.values())
        for concept, forms in LEXICON.items():
            rows.append({"paper": paper, "concept": concept,
                         "per_100k": stats.rate_per_100k(sum(c[f] for f in forms), tot)})
    rates = pd.DataFrame(rows).pivot(index="concept", columns="paper", values="per_100k")
    rates["IMVO/ISIG"] = rates["IMVO"] / rates["ISIG"]
    rates.to_csv(out / "lexicon_rates_overlap.csv")
    print(rates.round(2))

    # 4. Figure: words per year by paper and language.
    p = plt()
    fig, ax = p.subplots(figsize=(6, 2.6))
    for paper in config.PAPERS:
        t = tab.loc[paper]
        ax.bar(t.index + (-0.2 if paper == "ISIG" else 0.2), t.xh_words / 1e3, width=0.4,
               color=PAPER_COLORS[paper], label=f"{config.PAPER_NAMES[paper]} (isiXhosa)")
        ax.bar(t.index + (-0.2 if paper == "ISIG" else 0.2), t.en_words / 1e3, width=0.4,
               bottom=t.xh_words / 1e3, color=PAPER_COLORS[paper], alpha=0.35, label=f"{paper} (English)")
    ax.axvspan(lo - 0.5, hi + 0.5, color="k", alpha=0.06, label="overlap window")
    ax.set_ylabel("words (thousands)")
    ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.savefig(config.FIG_DIR / "eda_words_per_year.png")
    print("saved to", out)


if __name__ == "__main__":
    main()
