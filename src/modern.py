"""Load modern isiXhosa text used as (a) the lexicon for the normaliser,
(b) the CCNet reference corpus, (c) the augmentation data in B2 and
(d) the 'control' corpus in A.

Roles are disjoint, so nothing is evaluated on the text it was built from:
  ppl_ref  Wikipedia: trains the CCNet-style perplexity reference (as in CCNet)
  lexicon  half of the news + government documents: the normaliser's lexicon
           and the isiXhosa stop-word list
  control  the other half: the modern control in A1, the modern condition in
           B1 and the held-out modern test set in B2
  lm_train ppl_ref + lexicon: modern training data for the B2 language models
News/government documents are assigned to lexicon or control by a hash of the
document text, so the split is fixed and identical on every machine.

Expects plain-text files in resources/modern/, one document per line
(scripts/prepare_modern.py writes them). Keep the source of each file in its
name (e.g. masakhanews_xho.txt, wiki_xh.txt) so results can be split by source.
"""
from __future__ import annotations

import hashlib
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


PPL_REF_SOURCES = ("wiki_xh",)
SPLIT_SOURCES = ("masakhanews_xho", "vukuzenzele_xho")  # halved into lexicon / control
ROLES = ("ppl_ref", "lexicon", "control", "lm_train")


def _half(doc: str) -> str:
    return "lexicon" if int(hashlib.md5(doc.encode("utf-8")).hexdigest(), 16) % 2 == 0 else "control"


def iter_role(role: str, folder: Path = config.MODERN_DIR, min_words: int = 20):
    """Yield (source, doc) for the documents assigned to `role` (see module docstring)."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    for src, doc in iter_docs(folder, PPL_REF_SOURCES + SPLIT_SOURCES, min_words):
        if src in PPL_REF_SOURCES:
            if role in ("ppl_ref", "lm_train"):
                yield src, doc
        elif _half(doc) == role or (role == "lm_train" and _half(doc) == "lexicon"):
            yield src, doc


def lexicon(folder: Path = config.MODERN_DIR, sources=None, role: str | None = None) -> Counter:
    c: Counter = Counter()
    docs = iter_role(role, folder, min_words=1) if role else iter_docs(folder, sources, min_words=1)
    for _, doc in docs:
        c.update(words(doc))
    return c
