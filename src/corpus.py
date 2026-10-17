"""Load the isiXhosa Intellectual Traditions newspaper pages with metadata.

One row per page. The unit of *independence* is the edition (paper + date):
pages of one issue share continued articles, so splits and bootstrap resampling
must group by `edition`, never by page.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import pandas as pd

from . import config
from .text import clean, words

FNAME_RE = re.compile(r"^(IMVO|ISIG)_(\d{4})-(\d{2})-(\d{2})_p(\d{3})\.txt$")


def parse_name(fname: str) -> dict:
    m = FNAME_RE.match(fname)
    if not m:
        raise ValueError(f"unexpected file name: {fname}")
    paper, y, mo, d, p = m.groups()
    return {
        "page_id": fname[:-4],
        "paper": paper,
        "date": f"{y}-{mo}-{d}",
        "year": int(y),
        "page": int(p),
        "edition": f"{paper}_{y}-{mo}-{d}",
    }


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def load_pages(use_cache: bool = True, do_clean: bool = True) -> pd.DataFrame:
    """Return a DataFrame with columns:
    page_id, paper, date, year, page, edition, xh, en, n_xh_words, n_en_words
    plus the tracker columns (Xh%, En%, ...)."""
    config.ensure_dirs()
    cache = config.CACHE_DIR / f"pages_clean{int(do_clean)}.pkl"
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    rows = []
    for p in sorted(config.XH_DIR.glob("*.txt")):
        meta = parse_name(p.name)
        xh, en = _read(p), _read(config.EN_DIR / p.name)
        if do_clean:
            xh, en = clean(xh), clean(en)
        meta.update(xh=xh, en=en, n_xh_words=len(words(xh)), n_en_words=len(words(en)))
        rows.append(meta)
    df = pd.DataFrame(rows)

    if config.TRACKER_CSV.exists():
        tr = pd.read_csv(config.TRACKER_CSV)
        tr["page_id"] = tr["Edition"].str.replace(".txt", "", regex=False)
        df = df.merge(tr.drop(columns=["Edition"]), on="page_id", how="left")

    df = df.sort_values(["paper", "date", "page"]).reset_index(drop=True)
    with open(cache, "wb") as f:
        pickle.dump(df, f)
    return df


def select(df: pd.DataFrame, papers=None, years=None, overlap_only: bool = False) -> pd.DataFrame:
    if papers:
        df = df[df.paper.isin(papers)]
    if overlap_only:
        lo, hi = config.OVERLAP_YEARS
        df = df[df.year.between(lo, hi)]
    elif years:
        df = df[df.year.between(*years)]
    return df


def windows(text: str, size: int) -> list[str]:
    """Consecutive non-overlapping chunks of `size` whitespace words (single line).
    The trailing remainder is kept as its own chunk; callers filter by min_words."""
    toks = text.split()
    return [" ".join(toks[i:i + size]) for i in range(0, len(toks), size)]


def to_units(df: pd.DataFrame, lang: str = "xh", unit: str = "page", min_words: int = 20,
             window: int = 150) -> pd.DataFrame:
    """Explode pages into analysis units carrying metadata.
    'paragraph': what a web pipeline would see as documents after boilerplate removal;
    'page': how the archive is stored;
    'window': fixed `window`-word chunks of each page, so that historical and modern
    corpora are compared on equal-length units (length-dependent rules such as
    Gopher's 50-word minimum then cannot drive the difference)."""
    from .text import paragraphs

    meta_cols = ["page_id", "paper", "date", "year", "page", "edition"]
    out = []
    for r in df.itertuples(index=False):
        text = getattr(r, lang)
        if unit == "page":
            chunks = [text]
        elif unit == "window":
            chunks = windows(" ".join(paragraphs(text)), window)
        else:
            chunks = paragraphs(text)
        for i, c in enumerate(chunks):
            if len(c.split()) >= min_words:
                d = {k: getattr(r, k) for k in meta_cols}
                d.update(unit_id=f"{r.page_id}#{i}", lang=lang, text=c)
                out.append(d)
    return pd.DataFrame(out)
