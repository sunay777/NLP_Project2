"""Language-identification filter (the first gate of most pretraining pipelines).

Wraps fastText-format LID models:
  * NLLB LID-218  (facebook/fasttext-language-identification) labels like __label__xho_Latn
  * GlotLID       (cis-lmu/glotlid)                             labels like __label__xho_Latn

Typical pipeline rule: keep a document if its top label is the target language
and the probability is >= a threshold (CCNet used 0.5; many later pipelines
use 0.65 or higher). We record the full top-k so the threshold can be swept
afterwards rather than fixed up front.
"""
from __future__ import annotations

from pathlib import Path


class FastTextLID:
    def __init__(self, model_path: Path, name: str | None = None):
        import fasttext  # imported lazily: optional dependency

        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} missing - run scripts/download_resources.sh")
        fasttext.FastText.eprint = lambda *a, **k: None  # silence deprecation spam
        self.model = fasttext.load_model(str(model_path))
        self.name = name or Path(model_path).stem

    def predict(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        text = " ".join(text.split())  # fastText needs a single line
        labels, probs = self.model.predict(text, k=k)
        return [(l.replace("__label__", ""), float(p)) for l, p in zip(labels, probs)]


def keep(preds: list[tuple[str, float]], target: str, threshold: float) -> bool:
    lab, p = preds[0]
    return lab.split("_")[0] == target.split("_")[0] and p >= threshold
