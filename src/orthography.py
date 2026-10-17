"""Historical -> modern isiXhosa orthography normaliser (Experiment B).

Two systematic differences are visible in the 1870-1894 corpus:

1. **Aspiration is not written.** kuti/kuthi, pakati/phakathi, umteto/umthetho,
   Tixo/Thixo, rulumente/rhulumente. In the corpus the modern forms occur ~0 times.
   Modern spelling marks aspiration with 'h' after p, t, k, the clicks c/q/x,
   the affricates ts/ty (tsh, tyh) and r (rh).
2. **Word division is disjunctive.** Prefixes are split off proper nouns
   (ku Magosa, i Wesile, u-Maseya); modern spelling writes them conjunctively
   (kuMagosa, iWesile, uMaseya).

The normaliser generates every "insert an h" candidate for a word and picks the
one that is most frequent in a *modern* isiXhosa lexicon. It never invents words:
if no candidate is attested in modern text often enough, the word is left alone.
This makes it transparent and easy to audit, which matters for the write-up.

Limitations to discuss (do not hide them):
  * relies on modern-lexicon coverage; rare/obsolete words stay unnormalised;
  * homographs: historical 'ukuti' could be modern 'ukuthi' (to say) or a genuine
    unaspirated word; frequency picks the majority reading, context is ignored;
  * other historical spellings (e.g. vowel-length marks, loanword spellings) are
    not modelled. A small hand-checked gold set (data/gold_normalisation.tsv),
    ideally checked by an isiXhosa speaker, is needed to report precision.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter

from .text import WORD_RE

MAX_SITES = 10  # 2**10 candidates at most per word type


def aspiration_sites(w: str) -> list[int]:
    """Positions at which an 'h' could be inserted to mark aspiration."""
    sites, i, n = [], 0, len(w)
    while i < n:
        ch = w[i]
        nxt = w[i + 1] if i + 1 < n else ""
        if ch == "t" and nxt in ("s", "y"):
            after = w[i + 2] if i + 2 < n else ""
            if after != "h":
                sites.append(i + 2)  # ts -> tsh, ty -> tyh
            i += 2
            continue
        if ch in "ptkcqxr" and nxt != "h":
            sites.append(i + 1)
        i += 1
    return sites


def candidates(w: str) -> list[str]:
    sites = aspiration_sites(w)
    if not sites or len(sites) > MAX_SITES:
        return [w]
    out = []
    for mask in itertools.product((0, 1), repeat=len(sites)):
        chars, prev = [], 0
        for s, on in zip(sites, mask):
            chars.append(w[prev:s])
            if on:
                chars.append("h")
            prev = s
        chars.append(w[prev:])
        out.append("".join(chars))
    return out


def _match_case(src: str, tgt: str) -> str:
    if src.isupper() and len(src) > 1:
        return tgt.upper()
    if src[:1].isupper():
        return tgt[:1].upper() + tgt[1:]
    return tgt


def build_map(hist_counts: Counter, modern_counts: Counter, min_count: int = 3,
              ratio: float = 1.0) -> dict[str, str]:
    """Map lower-cased historical word types to modern spellings.

    A candidate replaces the original only if it occurs >= min_count times in the
    modern lexicon AND more than `ratio` x as often as the original form does.
    """
    mapping = {}
    for w in hist_counts:
        base = modern_counts.get(w, 0)
        best, best_c = w, base
        for c in candidates(w):
            cc = modern_counts.get(c, 0)
            if cc > best_c:
                best, best_c = c, cc
        if best != w and best_c >= min_count and best_c > ratio * base:
            mapping[w] = best
    return mapping


# ---- word division -------------------------------------------------------
_PREFIXES = ("u|i|e|o|a|ku|kwa|ka|nga|ngu|ngo|nge|nje|se|sa|so|ne|na|no|"
             "le|la|lo|ze|zi|ye|yi|ba|be|bo|wa|we|wo|ya|yo")
# separator: hyphen, spaces, or one line break - never a blank line (paragraph boundary)
_SPLIT_PREFIX = re.compile(rf"\b({_PREFIXES})(?:-|[ \t]+|[ \t]*\n[ \t]*)(?=[A-Z])")


def join_prefixes(text: str) -> str:
    """'ku Magosa' / 'u-Maseya' -> 'kuMagosa' / 'uMaseya' (prefix + capitalised word)."""
    return _SPLIT_PREFIX.sub(r"\1", text)


def normalise_text(text: str, mapping: dict[str, str], join: bool = True) -> str:
    """Spelling first (the map was built on un-joined tokens), then word division."""

    def repl(m: re.Match) -> str:
        w = m.group(0)
        new = mapping.get(w.lower())
        return _match_case(w, new) if new else w

    text = WORD_RE.sub(repl, text)
    return join_prefixes(text) if join else text


def coverage(counts: Counter, lexicon: Counter, min_count: int = 1) -> float:
    """Share of running tokens whose type is attested in the lexicon."""
    tot = sum(counts.values())
    hit = sum(c for w, c in counts.items() if lexicon.get(w, 0) >= min_count)
    return hit / tot if tot else 0.0
