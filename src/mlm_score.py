"""Pseudo-log-likelihood (PLL) of text under a pretrained masked LM
(Salazar et al., 2020): mask each token in turn and sum log P(token | rest).

Used in B3 as a zero-training probe: does a modern African-language MLM
(e.g. AfroXLMR) find normalised historical text more "natural" than raw?
Report PLL per *original word* so raw and normalised sentences are comparable.
Expensive (one forward pass per token) -> run on a fixed random sample of
sentences, e.g. 500 per condition, and bootstrap over editions.
"""
from __future__ import annotations


def pseudo_log_likelihood(text: str, tok, model, device: str = "cpu", max_len: int = 256) -> tuple[float, int]:
    import torch

    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len)
    ids = enc["input_ids"][0]
    n = len(ids)
    special = set(tok.all_special_ids)
    positions = [i for i in range(n) if int(ids[i]) not in special]
    if not positions:
        return 0.0, 0
    batch = ids.repeat(len(positions), 1)
    for row, pos in enumerate(positions):
        batch[row, pos] = tok.mask_token_id
    with torch.no_grad():
        logits = model(batch.to(device)).logits
    logp = torch.log_softmax(logits, dim=-1)
    total = sum(float(logp[row, pos, ids[pos]]) for row, pos in enumerate(positions))
    return total, len(positions)
