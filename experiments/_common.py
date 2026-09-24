"""Shared helpers for experiment scripts (run them from the repo root)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

PAPER_COLORS = {"ISIG": "#3b6ea8", "IMVO": "#c0622f", "modern": "#5a8f4e", "EN": "#7a7a7a"}


def outdir(name: str) -> Path:
    d = config.RESULTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                          "axes.spines.right": False, "savefig.bbox": "tight"})
    return _plt
