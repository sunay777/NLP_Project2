NLP Project 2: Language in Culture and Responsible NLP
COMS4054A / COMS7062A / COMS7066A, University of the Witwatersrand

Erased and penalised: 19th-century isiXhosa newspapers in modern NLP pipelines
(working title; see PLAN.md for hypotheses, design and threats to validity)

--------------------------------------------------------------------------
1. WHAT THIS DOES
--------------------------------------------------------------------------
Uses the isiXhosa Intellectual Traditions corpus (Isigidimi samaXosa 1870-1888,
Imvo Zabantsundu 1884-1894; Schoots et al., 2023) to measure:

  A. how much of the archive would survive a standard LLM pretraining filter
     cascade (language ID -> Gopher quality rules -> C4 bad words -> CCNet
     perplexity), compared with modern isiXhosa and the English sections;
  B. how much of the modelling penalty (tokeniser fertility, LM bits/word)
     is due to the pre-reform orthography, using a transparent historical ->
     modern spelling normaliser and a controlled {raw, normalised} x
     {share of modern data} experiment at a fixed training budget.

--------------------------------------------------------------------------
2. SETUP
--------------------------------------------------------------------------
Python 3.10-3.12 (fasttext-wheel has no 3.14 build).

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Data: the corpus is NOT in this repository. By default the code expects the
folder next to the repo:

    ../TxtLangFiles1870-1894_TB/{XhTxt/, EnTxt/, LanguageSplitTracker70-94.csv}

or set  export XHOSA_DATA=/path/to/TxtLangFiles1870-1894_TB

External resources (LID models, bad-word list, modern isiXhosa):

    bash scripts/download_resources.sh
    export HF_HOME=resources/hf_cache     # if ~/.cache is not writable
    python scripts/prepare_modern.py      # MasakhaNEWS, Wikipedia, Vuk'uzenzele (isiXhosa)
    python scripts/prepare_reference_en.py  # English Wikipedia sample (perplexity reference for hist_en)

--------------------------------------------------------------------------
3. RUN ORDER  (all commands from the repo root)
--------------------------------------------------------------------------
    python -m pytest -q                               # sanity tests
    python experiments/e0_eda.py                      # corpus statistics
    python experiments/e1_splits.py                   # edition-grouped splits
    python experiments/b0_normalise.py                # spelling map + normalised corpus
    python experiments/b0_gold.py --make              # gold-set candidates to hand-check
    python scripts/collect_gold_forms.py FORMS_DIR    # merge returned volunteer forms (data/gold_forms/)
    python experiments/b0_gold.py                     # normaliser precision/recall on checked rows
    python experiments/a1_pipeline_survival.py        # A, raw text (150-word windows)
    python experiments/a1_pipeline_survival.py --normalised
    python experiments/a1_pipeline_survival.py --xh-stopwords               # Gopher ablation
    python experiments/a1_pipeline_survival.py --normalised --xh-stopwords
    python experiments/a1_pipeline_survival.py --unit paragraph             # robustness
    python experiments/a1_pipeline_survival.py --unit paragraph --normalised
    python experiments/b1_tokenizer_fertility.py      # B1
    python experiments/b2_lm_2x2.py --jobs 3          # B2 (~20 min on 3 CPU cores)
    python experiments/c1_summary.py                  # gap decomposition + paper figures
    python scripts/language_audit.py --make           # 50-block language-split audit form (outside repo)
    python scripts/language_audit.py --score FORM     # error rate of the archive's language split
    python experiments/b3_mlm_pll.py                  # B3 (optional, not run)

Outputs go to results/<experiment>/ and figures to results/figures/.
Every script prints what it saved.

--------------------------------------------------------------------------
4. LAYOUT
--------------------------------------------------------------------------
src/config.py          paths (env-overridable), seeds, overlap window
src/corpus.py          load pages + metadata; page/paragraph units
src/text.py            conservative OCR cleaning, tokenisation, OCR-noise proxy
src/splits.py          temporal-per-paper and random splits, grouped by edition
src/orthography.py     historical -> modern normaliser (aspiration, word division)
src/modern.py          modern isiXhosa loader / lexicon
src/lm.py              character Kneser-Ney n-gram LM (pure Python)
src/filters/           lid.py, gopher.py, badwords.py, perplexity.py, pipeline.py
src/tokenizers_eval.py fertility / continued-word metrics
src/mlm_score.py       masked-LM pseudo-log-likelihood
src/stats.py           edition-cluster bootstrap, permutation test
experiments/           one script per experiment (see run order)
scripts/               resource download / preparation
tests/                 unit tests
paper/                 extended_abstract/ (main.tex, references.bib), ethics_drafts.md,
                       contribution_statement/
STATUS.md              what is done / what still needs a human
data/                  splits.json, gold_normalisation.tsv (cache/ is ignored)

--------------------------------------------------------------------------
5. NOTES
--------------------------------------------------------------------------
* The unit of independence is the edition (paper + date): all splits and
  bootstrap CIs resample editions, never pages.
* Filters are re-implementations using published thresholds; see docstrings.
* Generative AI was used to help write code; see the AI ethics statement
  in the extended abstract.
