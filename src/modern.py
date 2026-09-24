"""Load modern isiXhosa text used as (a) the lexicon for the normaliser,
(b) the CCNet reference corpus, (c) the augmentation data in B2 and
(d) the 'control' corpus in A.

Expects plain-text files in resources/modern/, one document per line
(scripts/prepare_modern.py writes them). Keep the source of each file in its
name (e.g. masakhanews_xho.txt, wiki_xh.txt) so results can be split by source.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import config
from .text import words


def iter_docs(folder: Path = config.MODERN_DIR, sources=None, min_words: int = 20):
    for p in sorted(Path(folder).glob("*.txt")):
        if sources and p.stem not in sources:
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if len(line.split()) >= min_words:
                    yield p.stem, line


def lexicon(folder: Path = config.MODERN_DIR, sources=None) -> Counter:
    c: Counter = Counter()
    for _, doc in iter_docs(folder, sources, min_words=1):
        c.update(words(doc))
    return c
