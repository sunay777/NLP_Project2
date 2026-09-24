"""Central paths and defaults.

Every path can be overridden with an environment variable so the code runs on
any machine without edits:

    XHOSA_DATA   folder containing XhTxt/, EnTxt/ and LanguageSplitTracker70-94.csv
    XHOSA_RES    folder holding downloaded resources (LID models, word lists, modern corpora)
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Historical corpus (not committed: it lives next to the repo by default).
DATA_ROOT = Path(os.environ.get("XHOSA_DATA", ROOT.parent / "TxtLangFiles1870-1894_TB"))
XH_DIR = DATA_ROOT / "XhTxt"
EN_DIR = DATA_ROOT / "EnTxt"
TRACKER_CSV = DATA_ROOT / "LanguageSplitTracker70-94.csv"

# Downloaded resources (see scripts/download_resources.sh). Git-ignored.
RES_DIR = Path(os.environ.get("XHOSA_RES", ROOT / "resources"))
LID_NLLB = RES_DIR / "lid" / "nllb_lid218.bin"      # facebook/fasttext-language-identification
LID_GLOT = RES_DIR / "lid" / "glotlid.bin"          # cis-lmu/glotlid
BADWORDS_DIR = RES_DIR / "badwords"                 # LDNOOBW lists, one file per language code
MODERN_DIR = RES_DIR / "modern"                     # modern isiXhosa .txt files (one doc per line)

# Outputs.
CACHE_DIR = ROOT / "data" / "cache"
SPLITS_JSON = ROOT / "data" / "splits.json"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "results" / "figures"

SEED = 0
PAPERS = ("ISIG", "IMVO")
PAPER_NAMES = {"ISIG": "Isigidimi samaXosa", "IMVO": "Imvo Zabantsundu"}
# Years both papers were in print: the natural controlled comparison window.
OVERLAP_YEARS = (1884, 1888)


def ensure_dirs() -> None:
    for d in (CACHE_DIR, RESULTS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
