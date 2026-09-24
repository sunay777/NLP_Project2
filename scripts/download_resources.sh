#!/usr/bin/env bash
# Download external resources into resources/ (git-ignored).
# Run from the repo root on your own machine:  bash scripts/download_resources.sh
# Check each URL/licence before using results in the write-up.
set -euo pipefail
RES="${XHOSA_RES:-resources}"
mkdir -p "$RES/lid" "$RES/badwords" "$RES/modern"

# --- Language ID ---------------------------------------------------------
# NLLB LID-218 (fastText) - labels like __label__xho_Latn
curl -L -o "$RES/lid/nllb_lid218.bin" \
  https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin
# GlotLID (fastText) - 1,600+ languages
curl -L -o "$RES/lid/glotlid.bin" \
  https://huggingface.co/cis-lmu/glotlid/resolve/main/model.bin

# --- C4 bad-words list (LDNOOBW) ----------------------------------------
curl -L -o "$RES/badwords/en" \
  https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en

# --- Modern isiXhosa -----------------------------------------------------
# Then run:  python scripts/prepare_modern.py
# (it pulls MasakhaNEWS xho + Wikipedia xh via HuggingFace `datasets`).
echo "Done. Now run: python scripts/prepare_modern.py"
