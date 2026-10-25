"""Merge returned gold-set forms (filled in by volunteers, .docx or .pdf) into
data/gold_normalisation.tsv, ready for `python experiments/b0_gold.py`.

For every word row in a returned form:
  historical  taken from the blank form by row number (data/gold_forms/), so a
              volunteer's edits or misplaced PDF text boxes cannot change it
  modern      what the volunteer wrote, lower-cased; "a / b" alternatives are kept
              as "a|b"; "same"/"unchanged" means the historical spelling
  checked_by  the volunteer's initials (full names go only in the acknowledgements)
  note        the volunteer's comment (name / English / OCR error / not sure / ...)
Rows with no modern answer are skipped. Rows already in the TSV for the same word
are replaced, so re-running is safe.

Run:  python scripts/collect_gold_forms.py path/to/returned_forms/ [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config  # noqa: E402

GOLD = config.ROOT / "data" / "gold_normalisation.tsv"
FORMS = config.ROOT / "data" / "gold_forms"
SAME_WORDS = {"same", "unchanged", "-", "–", "—", "ditto", "nochange", "no change"}
NOT_ANSWERS = {"not sure", "unsure", "english", "ocr error", "name", "?"}


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace(" ", " ")).strip()


def blank_words() -> dict[int, str]:
    """Row number -> 1880s word, from the blank forms that were sent out."""
    import docx

    out = {}
    for f in FORMS.glob("isiXhosa_spelling_check_set*.docx"):
        for t in docx.Document(str(f)).tables:
            for r in t.rows[1:]:
                c = [clean(x.text) for x in r.cells]
                if len(c) >= 2 and c[0].isdigit():
                    out[int(c[0])] = c[1]
    return out


def read_docx(path: Path) -> tuple[str, list[dict]]:
    import docx

    d = docx.Document(str(path))
    name = ""
    for para in d.paragraphs:
        m = re.match(r"\s*Your name:\s*(.*?)(Is isiXhosa your home language\?.*)?$", para.text)
        if m:
            name = clean(m.group(1).replace("_", ""))
            break
    rows = []
    for table in d.tables:
        for r in table.rows[1:]:
            c = [clean(x.text) for x in r.cells]
            if len(c) >= 4 and c[0].isdigit():
                rows.append({"n": int(c[0]), "historical": c[1], "modern": c[2], "note": c[3]})
    return name, rows


def read_any(path: Path) -> tuple[str, list[dict]]:
    if path.suffix.lower() == ".pdf":
        from read_returned_pdf import read_pdf

        return read_pdf(path)
    return read_docx(path)


def initials(name: str) -> str:
    """Initials for a full name; a single (first) name is kept as given."""
    parts = [p for p in name.split() if p[:1].isalpha()]
    return parts[0] if len(parts) == 1 else "".join(p[0].upper() for p in parts) or "?"


def normalise_answer(hist: str, modern: str, hist_field: str) -> str:
    """Lower-case answer; PDF text boxes that spill into the 1880s column are recovered."""
    m = clean(modern)
    if not m:  # answer box overlapping the 1880s column (PDF text boxes)
        extra = clean(hist_field)
        if extra.lower().startswith(hist.lower()):
            toks = clean(extra[len(hist):]).split()
            m = toks[0] if toks and len(set(toks)) == 1 else " ".join(toks)  # a box can be read twice
    m = m.lower().strip(".,;")
    if m in SAME_WORDS:
        return hist.lower()
    if m in NOT_ANSWERS:
        return ""
    alts = [clean(a) for a in re.split(r"\s*/\s*", m) if clean(a)]
    return "|".join(alts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path, help="folder holding the returned .docx / .pdf forms")
    ap.add_argument("--dry-run", action="store_true", help="print what would be merged, write nothing")
    args = ap.parse_args()

    words = blank_words()
    new = []
    for f in sorted(args.folder.iterdir()):
        if f.suffix.lower() not in (".docx", ".pdf") or f.name.startswith("~$"):
            continue
        name, rows = read_any(f)
        nums = sorted(r["n"] for r in rows if r["n"] in words)
        # keep only the 25 rows of the set this form belongs to (drops stray numbers in the intro text)
        set_id = max(set((n - 1) // 25 for n in nums), key=lambda s: sum((n - 1) // 25 == s for n in nums))
        rows = [r for r in rows if (r["n"] - 1) // 25 == set_id]
        kept = 0
        for r in rows:
            hist = words[r["n"]]
            note = clean(r["note"])
            modern_raw = clean(r["modern"])
            if modern_raw.lower() in NOT_ANSWERS and modern_raw.lower() not in note.lower():
                note = (note + "; " + modern_raw).strip("; ")  # comment typed into the answer box
            modern = normalise_answer(hist, modern_raw, r["historical"])
            if modern:
                kept += 1
                new.append({"historical": hist.lower(), "modern": modern, "checked_by": initials(name),
                            "note": note, "set": set_id + 1})
        print(f"{f.name}: set {set_id + 1}, checker={name!r} ({initials(name)}), {kept}/{len(rows)} answered")
    if not new:
        raise SystemExit("no filled-in rows found")
    new = pd.DataFrame(new).drop_duplicates("historical", keep="last")

    old = pd.read_csv(GOLD, sep="\t", dtype=str).fillna("")
    old = old[[c for c in old.columns if c in ("historical", "modern", "checked_by", "note", "set")]]
    merged = pd.concat([old[~old.historical.str.lower().isin(new.historical)], new], ignore_index=True)
    print(f"{len(new)} answered words from forms; gold file goes from {len(old)} to {len(merged)} rows")
    if args.dry_run:
        print(new.to_string(index=False))
        return
    merged.to_csv(GOLD, sep="\t", index=False)
    print("wrote", GOLD, "- now run: python experiments/b0_gold.py")


if __name__ == "__main__":
    main()
