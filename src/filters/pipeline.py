"""Run a pretraining-style filter cascade and record every decision.

Each filter is applied *independently* to every unit (so we can say which
filter is responsible for what) and the cascade survival is derived afterwards
in the canonical order LID -> quality (Gopher) -> bad words -> perplexity.
"""
from __future__ import annotations

import pandas as pd

from . import gopher, lid

ORDER = ("lid", "gopher", "badwords", "perplexity")


def run(units: pd.DataFrame, target_lang: str = "xho", lid_models=(), lid_threshold: float = 0.5,
        badword_filter=None, ppl_filter=None, gopher_stopwords=gopher.EN_STOPWORDS) -> pd.DataFrame:
    """`units` needs a `text` column. Returns units + one column per decision."""
    rows = []
    for t in units.text:
        r = {}
        for m in lid_models:
            preds = m.predict(t, k=3)
            r[f"lid_{m.name}_label"], r[f"lid_{m.name}_prob"] = preds[0]
            r[f"lid_{m.name}_top3"] = ";".join(f"{l}:{p:.3f}" for l, p in preds)
            r[f"keep_lid_{m.name}"] = lid.keep(preds, target_lang, lid_threshold)
        g = gopher.gopher_checks(t, gopher_stopwords)
        r["keep_gopher"] = all(g.values())
        r["gopher_failed"] = ";".join(k for k, v in g.items() if not v)
        if badword_filter is not None:
            hits = badword_filter.hits(t)
            r["keep_badwords"], r["badword_hits"] = not hits, ";".join(sorted(set(hits)))
        if ppl_filter is not None:
            r["ppl_bpc"] = ppl_filter.score(t)
            r["ppl_bucket"] = ppl_filter.bucket(t)
            r["keep_perplexity"] = r["ppl_bucket"] != "tail"
        rows.append(r)
    out = pd.concat([units.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return add_cascade(out, lid_col=f"keep_lid_{lid_models[0].name}" if lid_models else None)


def add_cascade(df: pd.DataFrame, lid_col: str | None) -> pd.DataFrame:
    stages = [c for c in (lid_col, "keep_gopher", "keep_badwords", "keep_perplexity") if c and c in df]
    alive = pd.Series(True, index=df.index)
    for c in stages:
        alive &= df[c]
        df[f"survive_after_{c.replace('keep_', '')}"] = alive
    df["survives"] = alive
    return df


def survival_table(df: pd.DataFrame, by=("corpus",)) -> pd.DataFrame:
    """Survival rate per group, weighted by units and by words."""
    df = df.assign(n_words=df.text.str.split().str.len())
    keep_cols = [c for c in df if c.startswith("keep_") or c.startswith("survive")]
    g = df.groupby(list(by))
    unit_rates = g[keep_cols].mean().add_suffix("_units")
    word_rates = pd.DataFrame({k: x[keep_cols].mul(x.n_words, axis=0).sum() / x.n_words.sum()
                               for k, x in g}).T.add_suffix("_words")
    word_rates.index = unit_rates.index
    return pd.concat([g.size().rename("n_units"), unit_rates, word_rates], axis=1)
