"""Train / validation / test splits, grouped by edition to avoid leakage.

Two schemes:
  * temporal (default): within each paper, the chronologically last editions go
    to test and the ones before them to validation. Every split keeps both
    papers, and "test" asks a real question: does what we learn generalise
    forward in time?
  * random: editions shuffled with a fixed seed (a sanity-check baseline).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config


def temporal_per_paper(df: pd.DataFrame, frac_val: float = 0.15, frac_test: float = 0.15) -> dict:
    split = {"train": [], "val": [], "test": []}
    for _, g in df.groupby("paper"):
        eds = sorted(g.edition.unique())  # edition ids sort chronologically
        n = len(eds)
        n_test, n_val = round(n * frac_test), round(n * frac_val)
        split["test"] += eds[n - n_test:]
        split["val"] += eds[n - n_test - n_val:n - n_test]
        split["train"] += eds[:n - n_test - n_val]
    return split


def random_by_edition(df: pd.DataFrame, frac_val=0.15, frac_test=0.15, seed=config.SEED) -> dict:
    rng = np.random.default_rng(seed)
    split = {"train": [], "val": [], "test": []}
    for _, g in df.groupby("paper"):  # stratify by paper
        eds = np.array(sorted(g.edition.unique()))
        rng.shuffle(eds)
        n = len(eds)
        n_test, n_val = round(n * frac_test), round(n * frac_val)
        split["test"] += eds[:n_test].tolist()
        split["val"] += eds[n_test:n_test + n_val].tolist()
        split["train"] += eds[n_test + n_val:].tolist()
    return split


def make_and_save(df: pd.DataFrame) -> dict:
    splits = {"temporal": temporal_per_paper(df), "random": random_by_edition(df)}
    config.SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.SPLITS_JSON.write_text(json.dumps(splits, indent=1))
    return splits


def load(scheme: str = "temporal") -> dict:
    return json.loads(config.SPLITS_JSON.read_text())[scheme]


def assign(df: pd.DataFrame, scheme: str = "temporal") -> pd.Series:
    s = load(scheme)
    lookup = {e: name for name, eds in s.items() for e in eds}
    return df.edition.map(lookup)
