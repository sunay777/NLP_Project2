"""Tokeniser "tax" on historical isiXhosa (Experiment B1).

Metrics from Rust et al. (2021), "How Good is Your Tokenizer?":
  * fertility: subword tokens per word;
  * continued-word proportion: share of words split into >= 2 pieces.
Higher = the model sees the text as more foreign (and it costs more compute and
context window per word -- a concrete, fairness-relevant cost).

Compare, per tokenizer: modern isiXhosa (control) vs historical raw vs historical
normalised. If normalisation closes most of the gap, the penalty is orthographic.
"""
from __future__ import annotations

import numpy as np

# Check these exist / are the versions you want before the final run.
# AfroXLMR (Davlan/afro-xlmr-base) reuses the XLM-R tokenizer unchanged, so the
# XLM-R row covers it. Serengeti (UBC-NLP/serengeti) fails to load under
# transformers>=4.5x; pass it via --tokenizers to retry.
DEFAULT_TOKENIZERS = {
    "XLM-R/AfroXLMR": "xlm-roberta-base",
    "NLLB-200": "facebook/nllb-200-distilled-600M",
    "mT5": "google/mt5-small",
    "BLOOM": "bigscience/bloom-560m",
    "GPT-2": "openai-community/gpt2",
    "Qwen2.5": "Qwen/Qwen2.5-0.5B",
}


def load_tokenizer(name_or_path: str):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name_or_path)
    except (TypeError, ValueError):  # some older repos only load as slow (sentencepiece) tokenizers
        return AutoTokenizer.from_pretrained(name_or_path, use_fast=False)


def word_pieces(tok, words: list[str]) -> np.ndarray:
    """Number of subword pieces per word (tokenised word-by-word, no specials)."""
    enc = tok(words, add_special_tokens=False, is_split_into_words=False)["input_ids"]
    return np.array([len(e) for e in enc])


def fertility_stats(tok, words: list[str]) -> dict:
    pieces = word_pieces(tok, words)
    return {
        "fertility": float(pieces.mean()),
        "continued_words": float((pieces > 1).mean()),
        "n_words": len(words),
    }
