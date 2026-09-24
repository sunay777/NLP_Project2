"""Gopher quality rules (Rae et al., 2021, App. A), as commonly re-implemented
(e.g. HuggingFace datatrove). Thresholds are the published defaults.

Note for the write-up: the stop-word rule requires >= 2 *English* stop words.
Applied to non-English text it is not a quality rule but a language filter.
`stopwords` is a parameter so the ablation "English stop words vs isiXhosa
stop words (top-k frequent modern isiXhosa words)" is one flag.
"""
from __future__ import annotations

from collections import Counter

EN_STOPWORDS = frozenset({"the", "be", "to", "of", "and", "that", "have", "with"})


def _dup_frac(items: list[str]) -> tuple[float, float]:
    """(fraction of items that are duplicates, fraction of chars in duplicates)."""
    if not items:
        return 0.0, 0.0
    c = Counter(items)
    dup_items = sum(v - 1 for v in c.values() if v > 1)
    dup_chars = sum(len(k) * (v - 1) for k, v in c.items() if v > 1)
    tot_chars = sum(len(x) for x in items) or 1
    return dup_items / len(items), dup_chars / tot_chars


def _top_ngram_frac(words: list[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    g, cnt = grams.most_common(1)[0]
    return cnt * sum(len(w) for w in g) / (sum(len(w) for w in words) or 1)


def _dup_ngram_frac(words: list[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    c = Counter(grams)
    covered = [False] * len(words)
    for i, g in enumerate(grams):
        if c[g] > 1:
            for j in range(i, i + n):
                covered[j] = True
    return sum(len(w) for w, cv in zip(words, covered) if cv) / (sum(len(w) for w in words) or 1)


def gopher_checks(text: str, stopwords=EN_STOPWORDS) -> dict[str, bool]:
    """Return {rule_name: passed}. A document is kept only if all pass."""
    words = text.split()
    n = len(words)
    alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
    lines = [l for l in text.splitlines() if l.strip()]
    paras = [p for p in text.split("\n\n") if p.strip()]
    mean_len = sum(len(w) for w in words) / n if n else 0
    symbols = text.count("#") + text.count("...") + text.count("…")
    low = [w.lower().strip(".,;:!?\"'()") for w in words]

    line_dup, line_char_dup = _dup_frac(lines)
    para_dup, para_char_dup = _dup_frac(paras)

    checks = {
        "word_count": 50 <= n <= 100_000,
        "mean_word_len": 3 <= mean_len <= 10,
        "symbol_ratio": (symbols / n if n else 1) < 0.1,
        "bullet_lines": (sum(l.lstrip().startswith(("•", "-", "*")) for l in lines) / len(lines) if lines else 0) <= 0.9,
        "ellipsis_lines": (sum(l.rstrip().endswith(("...", "…")) for l in lines) / len(lines) if lines else 0) <= 0.3,
        "alpha_words": (len(alpha_words) / n if n else 0) >= 0.8,
        "stopwords": sum(1 for w in low if w in stopwords) >= 2,
        "dup_lines": line_dup <= 0.3 and line_char_dup <= 0.2,
        "dup_paras": para_dup <= 0.3 and para_char_dup <= 0.2,
    }
    for k, thr in ((2, 0.20), (3, 0.18), (4, 0.16)):
        checks[f"top_{k}gram"] = _top_ngram_frac(low, k) <= thr
    for k, thr in zip(range(5, 11), (0.15, 0.14, 0.13, 0.12, 0.11, 0.10)):
        checks[f"dup_{k}gram"] = _dup_ngram_frac(low, k) <= thr
    return checks


def keep(text: str, stopwords=EN_STOPWORDS) -> bool:
    return all(gopher_checks(text, stopwords).values())
