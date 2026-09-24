"""Text cleaning and tokenisation for OCR'd 19th-century newspaper text.

Cleaning is deliberately conservative: isiXhosa uses hyphens legitimately
(u-Maseya, ahlulwa-hlulwa), so we only repair hyphens that are clearly OCR /
layout artefacts (line-break hyphenation). Every cleaning step is a flag so
its effect can be ablated.
"""
from __future__ import annotations

import re
import unicodedata

_SOFT_HYPHEN = "­"
# "Asi-\nkalungi" -> "Asikalungi": hyphen at end of line followed by a lowercase letter.
_LINEBREAK_HYPHEN = re.compile(r"(\w)-\s*\n\s*([a-z])")
# "ngezi- tili" -> "ngezitili": reflow artefact, hyphen + space + lowercase inside a line.
_REFLOW_HYPHEN = re.compile(r"(\w)- ([a-z])")
_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")

# Words: letters (incl. accented), optional internal apostrophes/hyphens.
WORD_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)


def clean(text: str, dehyphenate: bool = True, fix_reflow: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text).replace(_SOFT_HYPHEN, "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if dehyphenate:
        text = _LINEBREAK_HYPHEN.sub(r"\1\2", text)
    if fix_reflow:
        text = _REFLOW_HYPHEN.sub(r"\1\2", text)
    text = _WS.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def words(text: str, lower: bool = True) -> list[str]:
    ws = WORD_RE.findall(text)
    return [w.lower() for w in ws] if lower else ws


def paragraphs(text: str, min_words: int = 1) -> list[str]:
    """Split on blank lines; join wrapped lines inside a paragraph."""
    out = []
    for block in re.split(r"\n\s*\n", text):
        p = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if len(p.split()) >= min_words:
            out.append(p)
    return out


def ocr_noise(text: str) -> float:
    """Crude OCR-noise proxy: share of whitespace tokens that are not clean words
    or ordinary punctuation-wrapped words (e.g. 'carrv', 'w^(r)eni', 'Ngesrmso' pass,
    but 'w^(r)eni' / '1l1' do not). Use as a covariate / stratifier, not a truth."""
    toks = text.split()
    if not toks:
        return 0.0
    ok = sum(1 for t in toks if WORD_RE.fullmatch(t.strip(".,;:!?\"'()[]“”‘’—-")))
    return 1.0 - ok / len(toks)
