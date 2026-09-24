"""Uncertainty for every headline number.

Resampling unit = edition (pages of one issue are not independent).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_bootstrap(df: pd.DataFrame, stat, cluster: str = "edition", n_boot: int = 2000,
                      seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and percentile CI of stat(df), resampling whole clusters."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster)}
    keys = np.array(list(groups))
    est = stat(df)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(keys, size=len(keys), replace=True)
        boots.append(stat(pd.concat([groups[k] for k in sample], ignore_index=True)))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(est), float(lo), float(hi)


def permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 10_000, seed: int = 0) -> float:
    """Two-sided p-value for a difference in means."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        hits += abs(pooled[:len(a)].mean() - pooled[len(a):].mean()) >= obs
    return (hits + 1) / (n_perm + 1)


def rate_per_100k(counts: int, total: int) -> float:
    return 1e5 * counts / total if total else float("nan")
