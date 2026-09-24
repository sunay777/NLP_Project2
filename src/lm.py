"""Character n-gram language model with interpolated Kneser-Ney smoothing.

Pure Python (no KenLM build needed). Character level is chosen on purpose:
  * no OOV problem on noisy OCR text;
  * works identically for raw and normalised text, so conditions in the
    2x2 experiment differ only in the data, not in the model.

For comparability across the normalised / raw conditions report
**bits per original word** (total bits / number of words in the *raw* test
text) rather than per-character perplexity: normalisation adds 'h's, which
changes the character count but not the content. See PLAN.md, B-design.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

BOS, EOS = "\x02", "\x03"


class CharKN:
    def __init__(self, order: int = 5, discount: float = 0.75):
        self.n, self.d = order, discount

    def fit(self, texts, max_chars: int | None = None) -> "CharKN":
        n = self.n
        self.counts = [Counter() for _ in range(n + 1)]   # counts[k][ngram] for k-grams
        seen = 0
        for t in texts:
            s = BOS * (n - 1) + t + EOS
            for i in range(n - 1, len(s)):
                for k in range(1, n + 1):
                    self.counts[k][s[i - k + 1:i + 1]] += 1
            seen += len(t)
            if max_chars and seen >= max_chars:
                break
        self.vocab = {g for g in self.counts[1]} | {EOS}
        self.V = len(self.vocab)
        # Kneser-Ney statistics.
        self.ctx_total = [Counter() for _ in range(n + 1)]   # c(h)
        self.ctx_types = [Counter() for _ in range(n + 1)]   # N1+(h .)
        self.cont = [Counter() for _ in range(n + 1)]        # N1+(. w) for lower orders
        self.cont_ctx = [Counter() for _ in range(n + 1)]    # N1+(. h .)
        self.cont_ctx_types = [Counter() for _ in range(n + 1)]
        for k in range(1, n + 1):
            for g, c in self.counts[k].items():
                h = g[:-1]
                self.ctx_total[k][h] += c
                self.ctx_types[k][h] += 1
                if k >= 2:
                    lower = g[1:]
                    self.cont[k - 1][lower] += 1
        for k in range(1, n):
            for g, c in self.cont[k].items():
                self.cont_ctx[k][g[:-1]] += c
                self.cont_ctx_types[k][g[:-1]] += 1
        return self

    def prob(self, hist: str, ch: str) -> float:
        return self._p(hist[-(self.n - 1):] if self.n > 1 else "", ch, top=True)

    def _p(self, h: str, ch: str, top: bool) -> float:
        k = len(h) + 1
        if k == 1:
            num = self.cont[1].get(ch, 0) if self.n > 1 else self.counts[1].get(ch, 0)
            den = sum(self.cont[1].values()) if self.n > 1 else sum(self.counts[1].values())
            return (num + 1) / (den + self.V)  # add-one at the unigram floor
        if top:
            c_hw, c_h, types = self.counts[k].get(h + ch, 0), self.ctx_total[k].get(h, 0), self.ctx_types[k].get(h, 0)
        else:
            c_hw, c_h, types = self.cont[k].get(h + ch, 0), self.cont_ctx[k].get(h, 0), self.cont_ctx_types[k].get(h, 0)
        lower = self._p(h[1:], ch, top=False)
        if c_h == 0:
            return lower
        return max(c_hw - self.d, 0) / c_h + self.d * types / c_h * lower

    def logprob2(self, text: str) -> float:
        """Total log2 probability of text (including end-of-text)."""
        s = BOS * (self.n - 1) + text + EOS
        return sum(math.log2(self.prob(s[:i], s[i])) for i in range(self.n - 1, len(s)))

    def bits_per_char(self, texts) -> float:
        bits = chars = 0
        for t in texts:
            bits -= self.logprob2(t)
            chars += len(t) + 1
        return bits / chars
