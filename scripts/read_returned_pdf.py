"""Read a returned gold-set form that came back as a PDF (exported from Word or Google
Docs, or filled in with PDF text boxes) and write it as a .docx-equivalent TSV that
scripts/collect_gold_forms.py can merge.

Answers are located by position: each word row is anchored by its row number in the
first column, and every piece of text on that row (page text or FreeText annotation)
is assigned to the "Modern spelling" or "Comment" column by its x position relative to
the column headers.

Run:  python scripts/read_returned_pdf.py FORM.pdf [FORM2.pdf ...] --out DIR
      (writes DIR/<form name>.tsv with columns n, historical, modern, note, checked_by)
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def items(page):
    """(x0, y0, x1, y1, text) for page words and FreeText annotations."""
    out = [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]
    for a in page.annots() or []:
        txt = (a.info.get("content") or "").strip()
        if txt:
            r = a.rect
            out.append((r.x0, r.y0, r.x1, r.y1, txt))
    return out


def read_pdf(path: Path) -> tuple[str, list[dict]]:
    import pymupdf

    doc = pymupdf.open(str(path))
    name, rows, cols = "", {}, None
    for page in doc:
        its = items(page)
        heads = {t: x0 for x0, _, _, _, t in its if t in ("Modern", "Comment", "1880s")}
        if {"Modern", "Comment"} <= heads.keys():
            cols = (heads["Modern"] - 3, heads["Comment"] - 3)
        # name line: anything to the right of "name:" on the same line, before "Is isiXhosa"
        for x0, y0, x1, y1, t in its:
            if t == "name:":
                ny, nx = (y0 + y1) / 2, x1
                stop = min([a for a, b, c, d, s in its if s == "Is" and abs((b + d) / 2 - ny) < 8] or [1e9])
                parts = [s for a, b, c, d, s in sorted(its) if a >= nx - 1 and a < stop
                         and abs((b + d) / 2 - ny) < 10 and not set(s) <= set("_")]
                name = " ".join(p.replace("_", "") for p in parts).strip() or name
        if cols is None:
            continue
        anchors = [((y0 + y1) / 2, int(t)) for x0, y0, x1, y1, t in its if t.isdigit() and x0 < cols[0] - 80
                   and 1 <= int(t) <= 200]
        if not anchors:
            continue
        anchors.sort()
        half = min((b[0] - a[0] for a, b in zip(anchors, anchors[1:])), default=28) / 2
        for x0, y0, x1, y1, t in its:
            yc = (y0 + y1) / 2
            ay, n = min(anchors, key=lambda a: abs(a[0] - yc))
            if abs(ay - yc) > half or (t.isdigit() and int(t) == n and x0 < cols[0] - 80):
                continue
            r = rows.setdefault(n, {"n": n, "historical": [], "modern": [], "note": []})
            key = "historical" if x0 < cols[0] else "modern" if x0 < cols[1] else "note"
            r[key].append((x0, t))
    out = []
    for n in sorted(rows):
        r = rows[n]
        out.append({k: (" ".join(t for _, t in sorted(r[k])) if k != "n" else n) for k in r})
    return name, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.pdfs:
        name, rows = read_pdf(f)
        dst = args.out / (f.stem + ".tsv")
        with open(dst, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, ["n", "historical", "modern", "note", "checked_by"], delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow({**r, "checked_by": name})
        print(f"{f.name}: name={name!r}, {len(rows)} rows, {sum(bool(r['modern']) for r in rows)} filled -> {dst}")


if __name__ == "__main__":
    main()
