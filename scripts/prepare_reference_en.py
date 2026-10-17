"""Write an English Wikipedia sample to resources/reference/wiki_en.txt (one doc per line).

Used only as the CCNet-style perplexity reference for the historical English
sections, so that hist_en passes through the same four gates as isiXhosa.
Kept outside resources/modern/ so it never enters the isiXhosa lexicon.

Run:  HF_HOME=resources/hf_cache python scripts/prepare_reference_en.py [--max-docs 3000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=3000)
    args = ap.parse_args()
    from datasets import load_dataset

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    config.REF_EN.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(config.REF_EN, "w", encoding="utf-8") as f:
        for ex in ds:
            doc = " ".join(ex["text"].split())
            if len(doc.split()) >= 150:
                f.write(doc + "\n")
                n += 1
            if n >= args.max_docs:
                break
    print(f"wiki_en: {n} docs -> {config.REF_EN}")
    import os
    os._exit(0)  # streaming iterator can hang on interpreter shutdown


if __name__ == "__main__":
    main()
