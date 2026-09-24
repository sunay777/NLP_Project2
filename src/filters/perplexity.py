"""CCNet-style perplexity filter (Wenzek et al., 2020).

CCNet trains an LM on a "high-quality" reference corpus (Wikipedia) per
language, scores every web document, and splits them into head / middle / tail
by perplexity percentiles of that reference; the tail is usually discarded.
Here the reference is *modern* isiXhosa, which is exactly the point: text that
is "unlike Wikipedia" is treated as low quality, and 1880s orthography is
unlike Wikipedia by construction.
"""
from __future__ import annotations

import numpy as np

from ..lm import CharKN


class PerplexityFilter:
    def __init__(self, reference_texts, order: int = 5, max_chars: int = 5_000_000,
                 calib_frac: float = 0.1, seed: int = 0):
        texts = list(reference_texts)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(texts))
        n_cal = max(1, int(len(texts) * calib_frac))
        calib = [texts[i] for i in idx[:n_cal]]
        train = [texts[i] for i in idx[n_cal:]]
        self.lm = CharKN(order).fit(train, max_chars=max_chars)
        ref = np.array([self.score(t) for t in calib])
        # CCNet buckets: head <= 30th pct, middle <= 60th, tail above.
        self.p30, self.p60 = np.percentile(ref, [30, 60])

    def score(self, text: str) -> float:
        return self.lm.bits_per_char([text])

    def bucket(self, text: str) -> str:
        s = self.score(text)
        return "head" if s <= self.p30 else "middle" if s <= self.p60 else "tail"

    def keep(self, text: str) -> bool:
        return self.bucket(text) != "tail"
