"""Write modern isiXhosa corpora to resources/modern/<source>.txt (one doc per line).

Sources (edit SOURCES to add/remove; verify dataset ids and licences):
  masakhanews_xho : news articles, MasakhaNEWS (Adelani et al., 2023)
  wiki_xh         : isiXhosa Wikipedia (wikimedia/wikipedia, 20231101.xh)
Other good options to add manually as .txt files:
  * NCHLT isiXhosa text corpus (SADiLaR repository - government documents)
  * Vukuzenzele / ZA-gov multilingual corpora (dsfsi on GitHub)

Keeping sources separate lets you report results per source: government,
news and Wikipedia isiXhosa are themselves different registers.

Run:  python scripts/prepare_modern.py [--max-docs 20000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

SOURCES = {
    "masakhanews_xho": dict(path="masakhane/masakhanews", name="xho", fields=("headline", "text")),
    "wiki_xh": dict(path="wikimedia/wikipedia", name="20231101.xh", fields=("text",)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=20000)
    args = ap.parse_args()
    from datasets import load_dataset

    config.MODERN_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in SOURCES.items():
        try:
            ds = load_dataset(spec["path"], spec["name"])
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {name}: {e}")
            continue
        n = 0
        with open(config.MODERN_DIR / f"{name}.txt", "w", encoding="utf-8") as f:
            for split in ds:
                for ex in ds[split]:
                    doc = " ".join(str(ex.get(k, "")) for k in spec["fields"])
                    doc = " ".join(doc.split())
                    if doc:
                        f.write(doc + "\n")
                        n += 1
                    if n >= args.max_docs:
                        break
        print(f"{name}: {n} docs")


if __name__ == "__main__":
    main()
