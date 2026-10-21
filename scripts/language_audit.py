"""Hand audit of the archive's isiXhosa/English split (PLAN section 5).

  --make  sample 25 blocks the archive labels isiXhosa and 25 it labels English
          (stratified by paper, fixed seed) into a Word form. A team member marks what
          each block really is. The form quotes archive text, so it is written outside
          the repository by default and must not be committed.
  --score read the returned form and report the share of blocks with the wrong label
          (95% Wilson intervals) -> results/eda/language_audit.json

Run:  python scripts/language_audit.py --make  [--out ../language_audit.docx]
      python scripts/language_audit.py --score ../language_audit.docx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, corpus  # noqa: E402

N_PER_LABEL = 25
SNIPPET_WORDS = 40
LABELS = {"xh": "isiXhosa", "en": "English"}
ANSWER_HINT = "isiXhosa / English / mixed / unreadable"


def sample_blocks(seed: int = config.SEED):
    pages = corpus.load_pages()
    rng = np.random.default_rng(seed)
    out = []
    for lang in ("xh", "en"):
        u = corpus.to_units(pages, lang, "paragraph", min_words=10)
        # stratify by paper in proportion to its blocks of this language
        share = u.paper.value_counts(normalize=True)
        for paper, frac in share.items():
            k = int(round(N_PER_LABEL * frac))
            pool = u[u.paper == paper]
            pick = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
            for i in pick:
                r = pool.iloc[i]
                out.append({"unit_id": r.unit_id, "label": LABELS[lang],
                            "text": " ".join(r.text.split()[:SNIPPET_WORDS]) + " ..."})
    order = rng.permutation(len(out))  # mix the two labels so the label is not guessable by position
    return [out[i] for i in order]


def make(path: Path) -> None:
    import docx
    from docx.shared import Pt

    blocks = sample_blocks()
    d = docx.Document()
    d.styles["Normal"].font.name = "Arial"
    d.styles["Normal"].font.size = Pt(10)
    d.add_heading("Language-split audit (50 blocks)", level=1)
    d.add_paragraph(
        "Each block below is the start of a paragraph that the archive's language split labels as isiXhosa or "
        "English. Read the snippet and write what it actually is in the last column: isiXhosa, English, mixed "
        "(both languages), or unreadable (OCR garbage). This takes about 10 minutes and needs no isiXhosa "
        "fluency. Please don't change the other columns.")
    d.add_paragraph("Checked by: ______________________      Date: ____________")
    t = d.add_table(rows=1, cols=5)
    t.style = "Table Grid"
    for c, h in zip(t.rows[0].cells, ["#", "Block id", "Archive label", "Snippet", f"Actually ({ANSWER_HINT})"]):
        c.text = h
    for i, b in enumerate(blocks, 1):
        cells = t.add_row().cells
        for c, v in zip(cells, [str(i), b["unit_id"], b["label"], b["text"], ""]):
            c.text = v
    d.save(path)
    print(f"wrote {len(blocks)} blocks to {path}")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p, den = k / n, 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / den, z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [round(mid - half, 4), round(mid + half, 4)]


def score(path: Path) -> None:
    import docx

    t = docx.Document(str(path)).tables[0]
    rows = [[c.text.strip() for c in r.cells] for r in t.rows[1:]]
    done = [r for r in rows if r[4]]
    if not done:
        raise SystemExit("no answers filled in")
    res = {"n_checked": len(done)}
    for label in LABELS.values():
        g = [r for r in done if r[2] == label]
        wrong = [r for r in g if not r[4].lower().startswith(label.lower())]
        res[label] = {"n": len(g), "n_wrong": len(wrong), "error_rate": len(wrong) / len(g) if g else None,
                      "ci": wilson(len(wrong), len(g)),
                      "wrong_as": {a: sum(r[4].lower().startswith(a) for r in wrong)
                                   for a in ("isixhosa", "english", "mixed", "unreadable")}}
    wrong_all = sum(res[l]["n_wrong"] for l in LABELS.values())
    res["overall"] = {"error_rate": wrong_all / len(done), "ci": wilson(wrong_all, len(done))}
    out = config.RESULTS_DIR / "eda" / "language_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--out", type=Path, default=config.ROOT.parent / "language_audit.docx")
    ap.add_argument("--score", type=Path)
    args = ap.parse_args()
    if args.make:
        make(args.out)
    elif args.score:
        score(args.score)
    else:
        ap.error("use --make or --score FILE")


if __name__ == "__main__":
    main()
