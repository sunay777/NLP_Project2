"""Fast unit tests: python -m pytest -q"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import corpus, orthography, text  # noqa: E402
from src.filters import badwords, gopher  # noqa: E402
from src.lm import CharKN  # noqa: E402


def test_parse_name():
    m = corpus.parse_name("IMVO_1887-02-16_p004.txt")
    assert m["paper"] == "IMVO" and m["year"] == 1887 and m["page"] == 4
    assert m["edition"] == "IMVO_1887-02-16"


def test_clean_dehyphenates_linebreak_only():
    assert text.clean("Asi-\nkalungi") == "Asikalungi"
    assert "u-Maseya" in text.clean("wati u-Maseya")          # real hyphen kept
    assert "ahlulwa-hlulwa" in text.clean("ahlulwa-hlulwa")


def test_aspiration_candidates():
    c = set(orthography.candidates("pakati"))
    assert "phakathi" in c and "pakati" in c and len(c) == 8   # p, k, t sites
    assert "tsh" in "".join(orthography.candidates("ts"))


def test_build_map_and_normalise():
    modern = Counter({"kuthi": 50, "phakathi": 40, "umthetho": 30, "uThixo": 0, "thixo": 20, "into": 90})
    hist = Counter({"kuti": 10, "pakati": 5, "umteto": 4, "tixo": 3, "into": 7, "xyz": 1})
    m = orthography.build_map(hist, modern)
    assert m == {"kuti": "kuthi", "pakati": "phakathi", "umteto": "umthetho", "tixo": "thixo"}
    out = orthography.normalise_text("Kuti u Tixo pakati, into.", m)
    assert out == "Kuthi uThixo phakathi, into."


def test_gopher_english_stopword_rule_rejects_isixhosa():
    xh = " ".join(["Umfo abati ngu Maseya indoda eselilixego ite yafuna ukongezelela"] * 8)
    checks = gopher.gopher_checks(xh)
    assert checks["stopwords"] is False
    assert gopher.gopher_checks(xh, stopwords={"ngu", "ite"})["stopwords"] is True


def test_badwords():
    f = badwords.BadWordFilter({"badword", "two words"})
    assert f.hits("a BadWord here") == ["badword"] and not f.keep("two words")
    assert f.keep("clean text")


def test_kn_normalised():
    lm = CharKN(order=3).fit(["abcab", "bca", "cab"])
    for h in ("\x02\x02", "ab", "zz"):
        total = sum(lm.prob(h, ch) for ch in lm.vocab)
        assert abs(total - 1) < 1e-9, (h, total)
