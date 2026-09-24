"""B3 (optional) - pseudo-log-likelihood of historical text under a modern
African-language masked LM, raw vs normalised. No training required.

Run:  python experiments/b3_mlm_pll.py --model Davlan/afro-xlmr-base --n 300
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
from _common import config, outdir

from src import corpus, stats
from src.mlm_score import pseudo_log_likelihood
from src.text import words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Davlan/afro-xlmr-base")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out = outdir("B3")

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(args.device).eval()

    with open(config.CACHE_DIR / "pages_normalised.pkl", "rb") as f:
        pages = pickle.load(f)
    raw = corpus.to_units(pages, "xh", "paragraph", min_words=10)
    norm = corpus.to_units(pages.assign(xh=pages.xh_norm), "xh", "paragraph", min_words=1).set_index("unit_id")
    sample = raw.sample(args.n, random_state=config.SEED)

    rows = []
    for r in sample.itertuples():
        nw = len(words(r.text))
        for cond, text in (("raw", r.text), ("norm", norm.loc[r.unit_id, "text"])):
            ll, _ = pseudo_log_likelihood(" ".join(text.split()[:120]), tok, model, args.device)
            rows.append({"unit_id": r.unit_id, "edition": r.edition, "paper": r.paper,
                         "cond": cond, "pll_per_word": ll / max(nw, 1)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "pll.csv", index=False)
    wide = res.pivot(index=["unit_id", "edition", "paper"], columns="cond", values="pll_per_word").reset_index()
    wide["gain"] = wide["norm"] - wide["raw"]
    print(stats.cluster_bootstrap(wide, lambda d: d.gain.mean()))
    print("share of paragraphs improved:", float(np.mean(wide.gain > 0)))


if __name__ == "__main__":
    main()
