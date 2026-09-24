"""C4-style "bad words" filter (Raffel et al., 2020).

C4 dropped any page containing a word from the LDNOOBW list. That rule is
blunt: 19th-century anti-colonial writing that *quotes* a slur in order to
contest it gets removed just like abusive text. Recording *which* words fired
lets the discussion say what kind of content is lost, not just how much.
"""
from __future__ import annotations

import re
from pathlib import Path


def load_lists(folder: Path, langs=("en",)) -> set[str]:
    words: set[str] = set()
    for lang in langs:
        p = Path(folder) / lang
        if p.exists():
            words |= {w.strip().lower() for w in p.read_text(encoding="utf-8").splitlines() if w.strip()}
    return words


class BadWordFilter:
    def __init__(self, words: set[str]):
        single = sorted(w for w in words if " " not in w)
        multi = sorted(w for w in words if " " in w)
        self.single = set(single)
        self.multi = [re.compile(r"\b" + re.escape(m) + r"\b") for m in multi]

    def hits(self, text: str) -> list[str]:
        low = text.lower()
        found = [t for t in re.findall(r"[a-z']+", low) if t in self.single]
        found += [m.pattern for m in self.multi if m.search(low)]
        return found

    def keep(self, text: str) -> bool:
        return not self.hits(text)
