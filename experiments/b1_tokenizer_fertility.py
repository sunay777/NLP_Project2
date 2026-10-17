"""B1 - the tokeniser tax: how many subword tokens modern tokenisers spend on
modern vs historical-raw vs historical-normalised isiXhosa.

Metrics (ratio of sums over units, 95% edition-cluster bootstrap CIs):
  fertility        tokens / word of the text as written (Rust et al., 2021)
  continued_words  share of words split into >= 2 pieces
  tokens_per_char  tokens / letters; robust to word-division conventions, so it
                   is the fair metric for historical vs modern
  tokens_per_raw_word
                   tokens / words of the RAW historical text; holds content fixed,
                   so it is the fair metric for raw vs normalised (prefix joining
                   changes the word count, which distorts plain fertility)

Needs: transformers + sentencepiece, B0 run first, modern text present.
Tokenizers download from HuggingFace on first use.

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
from src.tokenizers_eval import DEFAULT_TOKENIZERS, load_tokenizer, word_pieces

METRICS = ("fertility", "continued_words", "tokens_per_char", "tokens_per_raw_word")


def unit_counts(tok, units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, raw_t in zip(units.text, units.raw_text):
        ws = words(t, lower=False)
        pieces = word_pieces(tok, ws) if ws else np.array([], int)
        rows.append({"tokens": int(pieces.sum()), "n_words": len(ws), "n_cont": int((pieces > 1).sum()),
                     "n_chars": sum(len(w) for w in ws), "n_raw_words": len(words(raw_t))})
    return pd.concat([units.reset_index(drop=True), pd.DataFrame(rows)], axis=1).query("n_words > 0")


def metric_fns():
    return {
        "fertility": lambda d: d.tokens.sum() / d.n_words.sum(),
        "continued_words": lambda d: d.n_cont.sum() / d.n_words.sum(),
        "tokens_per_char": lambda d: d.tokens.sum() / d.n_chars.sum(),
        "tokens_per_raw_word": lambda d: d.tokens.sum() / d.n_raw_words.sum(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=2000, help="paragraphs per condition")
    ap.add_argument("--tokenizers", nargs="*", default=list(DEFAULT_TOKENIZERS))
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    out = outdir("B1")

    with open(config.CACHE_DIR / "pages_normalised.pkl", "rb") as f:
        pages = pickle.load(f)
    raw = corpus.to_units(pages, "xh", "paragraph")
    # min_words=1 so every raw unit has a normalised twin (prefix joining shortens word counts)
    norm = corpus.to_units(pages.assign(xh=pages.xh_norm), "xh", "paragraph", min_words=1).set_index("unit_id")
    idx = np.random.default_rng(config.SEED).choice(len(raw), size=min(args.units, len(raw)), replace=False)
    hraw = raw.iloc[idx].assign(raw_text=lambda d: d.text)
    hnorm = norm.loc[hraw.unit_id].reset_index().assign(raw_text=hraw.text.values)
    en = corpus.to_units(pages, "en", "paragraph")
    en = en.sample(min(args.units, len(en)), random_state=config.SEED).assign(raw_text=lambda d: d.text)
    # hist_en: same pages, English sections -- separates the "isiXhosa tax" from any "1880s tax"
    conds = {"hist_raw": hraw, "hist_norm": hnorm, "hist_en": en}
    mod = [d for _, d in modern.iter_docs()]
    if mod:
        m_idx = np.random.default_rng(config.SEED).choice(len(mod), size=min(args.units, len(mod)), replace=False)
        texts = [mod[i] for i in m_idx]
        conds["modern"] = pd.DataFrame({"text": texts, "raw_text": texts, "edition": [f"m{i}" for i in m_idx]})

    rows = []
    for tname in args.tokenizers:
        try:
            tok = load_tokenizer(DEFAULT_TOKENIZERS.get(tname, tname))
        except Exception as e:  # noqa: BLE001 - gated/broken repos should not stop the sweep
            print(f"[skip] {tname}: {type(e).__name__}: {e}")
            continue
        for cname, units in conds.items():
            per_unit = unit_counts(tok, units)
            for metric, fn in metric_fns().items():
                est, lo, hi = stats.cluster_bootstrap(per_unit, fn, n_boot=args.n_boot)
                rows.append({"tokenizer": tname, "condition": cname, "metric": metric,
                             "value": est, "ci_lo": lo, "ci_hi": hi})
        print(pd.DataFrame(rows).query("tokenizer == @tname").pivot(index="metric", columns="condition",
                                                                       values="value").round(3))
    res = pd.DataFrame(rows)
    res.to_csv(out / "fertility.csv", index=False)

    # Figure: tokens per character (fair across corpora) per tokenizer and condition.
    p = plt()
    fig, ax = p.subplots(figsize=(5, 2.6))
    r = res[res.metric == "tokens_per_char"]
    names = list(dict.fromkeys(r.tokenizer))
    conds_order = [c for c in ("hist_en", "modern", "hist_norm", "hist_raw") if c in set(r.condition)]
    w = 0.8 / len(conds_order)
    for i, c in enumerate(conds_order):
        rc = r[r.condition == c].set_index("tokenizer").loc[names]
        ax.bar(np.arange(len(rc)) + i * w, rc.value, w, yerr=[rc.value - rc.ci_lo, rc.ci_hi - rc.value],
               label=c, capsize=2)
    ax.set_xticks(np.arange(len(names)) + w * (len(conds_order) - 1) / 2, names, rotation=15)
    ax.set_ylabel("subword tokens / letter")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(config.FIG_DIR / "B1_tokens_per_char.png")
    print("saved to", out)


if __name__ == "__main__":
    main()
