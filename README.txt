NLP Project 2: Language in Culture and Responsible NLP
COMS4054A / COMS7062A / COMS7066A, University of the Witwatersrand
Sunay Master (2677874), Matthew Purkiss (2697424), Harshil Vallabh (2656158)

Erased and Penalised: 19th-Century isiXhosa Newspapers in a Modern LLM Data Pipeline
==================================================================================

We pass the isiXhosa Intellectual Traditions newspapers (Isigidimi samaXosa
1870-1888, Imvo Zabantsundu 1884-1894; Schoots et al., 2023) through
re-implementations of a standard LLM pretraining filter cascade, and measure
how much of the penalty is due to their pre-reform spelling. The extended
abstract (NLP_Project2_Extended_Abstract.pdf in the submitted zip) reports that:

  * none of 5,000 historical isiXhosa 150-word windows survives the cascade
    (language ID -> Gopher rules -> C4 bad words -> CCNet perplexity), against
    32.6% of the English columns of the same pages and, with isiXhosa stop
    words, 25.6% of held-out modern isiXhosa;
  * the perplexity gate (a Wikipedia reference LM) is where the archive is
    singled out; language ID is not the bottleneck;
  * a historical -> modern spelling normaliser closes about a fifth of the
    perplexity gap but leaves survival near zero; tokenisers charge isiXhosa
    more per letter than English but barely notice the 1880s spelling.

Contents
  1. What is in this folder
  2. How to run the code (step by step, with run times)
  3. Notes (design decisions, steps that needed people)

==================================================================
1. WHAT IS IN THIS FOLDER
==================================================================
The submitted zip has four items at the top:
  NLP_Project2_Extended_Abstract.pdf   the extended abstract
  README.txt                           this file
  requirements.txt                     the pinned packages
  code/                                the full code
code/ is laid out exactly like the GitHub repository
(https://github.com/sunay777/NLP_Project2; it has its own copy of README.txt
and requirements.txt), so every path below is relative to code/.

The newspaper corpus itself is NOT included (it comes from the archive and
Moodle as TxtLangFiles1870-1894_TB); step 1 below says where to put it.

Inside code/ (the GitHub repository):
README.txt         this file
requirements.txt   the pinned packages
PLAN.md            the research plan written before the experiments
                   (hypotheses, design, threats to validity); the changes
                   made since are listed in the paper's contribution statement

src/               library code used by every script
  config.py          paths (each can be overridden by an environment variable)
  corpus.py          loads the pages and the archive's language split; cuts
                     them into page, paragraph or 150-word window units
  text.py            OCR clean-up, word splitting, OCR-noise measure
  splits.py          temporal and random train/val/test splits, by edition
  modern.py          modern isiXhosa in disjoint roles (perplexity reference,
                     normaliser lexicon, held-out control)
  orthography.py     the historical -> modern spelling normaliser
  filters/           the four gates: lid.py (NLLB, GlotLID), gopher.py,
                     badwords.py (C4 / LDNOOBW), perplexity.py (CCNet-style),
                     pipeline.py (applies them singly and in sequence)
  lm.py              character Kneser-Ney n-gram language model
  tokenizers_eval.py tokens per letter, fertility
  stats.py           bootstrap over editions, permutation test
  mlm_score.py       masked-LM scoring (optional B3, not in the paper)

experiments/       one script per experiment, run as  python experiments/<name>.py
  e0_eda.py                 corpus statistics (Methods: pages, editions, words)
  e1_splits.py              edition-grouped splits -> data/splits.json
  b0_normalise.py           B0: builds the spelling normaliser
  b0_gold.py                B0: scores it on the volunteer-checked gold set
  a1_pipeline_survival.py   A: the filter cascade (Fig. 1, H1, H2)
  b1_tokenizer_fertility.py B1: six tokenisers
  b2_lm_2x2.py              B2: character LMs, {raw, normalised} x modern share (H4)
  c1_summary.py             combines A, B1 and B2: gap decomposition (H3),
                            Figures 1 and 2, headline numbers
  b3_mlm_pll.py             optional B3, not run for the paper

scripts/           data preparation and the two hand-checking steps
  download_resources.sh     language-ID models and the bad-word list
  prepare_modern.py         modern isiXhosa (MasakhaNEWS, Wikipedia, Vuk'uzenzele)
  prepare_reference_en.py   English Wikipedia sample (reference LM for English)
  collect_gold_forms.py     merges the volunteers' returned forms into the gold set
  language_audit.py         makes and scores the 50-block language-split audit
  read_returned_pdf.py      helper for forms returned as PDF

data/
  splits.json               the train/val/test splits used in the paper
  gold_candidates.tsv       the 200 word types sampled for checking
  gold_forms/               the blank forms and instructions sent to volunteers
  gold_normalisation.tsv    the gold set: words checked by six isiXhosa
                            speakers (initials only)

results/           outputs of the scripts; the files the paper quotes are:
  C1/headline.json, C1/gate_attribution.csv   survival per gate (Fig. 1, H1, H2)
  C1/lid_threshold_sweep.csv                  LID at thresholds 0.3-0.8
  C1/gap_decomposition.csv                    share closed by normalising (Fig. 2, H3)
  A1*/survival_ci.csv                         every cascade setting, with CIs
  B0/coverage.json, B0/gold_eval.json         the normaliser and its gold-set scores
  B1/fertility.csv                            tokenisers
  B2/lm_2x2.csv, B2/h4_interaction.csv        the LM experiment and the H4 test
  eda/language_audit.json                     the language-split audit
  figures/fig1_cascade.png, fig2_gap.png      Figures 1 and 2

tests/             unit tests

paper/extended_abstract/  LaTeX source of the abstract (main.tex), the NeurIPS
                   checklist (checklist.tex), the AI declaration
                   (ai_declaration.tex), references.bib, and the template's
                   ccn.sty and apacite.bst

==================================================================
2. HOW TO RUN THE CODE
==================================================================
Run every command from the code folder: code/ in the submitted zip (run
`cd code` first), or the top level of the GitHub repository. All scripts are
seeded, and each prints what it saved. Times are from an Apple M-series
laptop, CPU only (no GPU needed).

------------------------------------------------------------------
Step 1. Put the corpus in place
------------------------------------------------------------------
Unzip the corpus so that the folder TxtLangFiles1870-1894_TB (containing
XhTxt/, EnTxt/ and LanguageSplitTracker70-94.csv) sits NEXT TO code/, in the
same folder as this README:

    README.txt
    code/
    TxtLangFiles1870-1894_TB/

(For the GitHub repository: next to the repository folder.) If it is
anywhere else, point the code to it instead:

    export XHOSA_DATA=/path/to/TxtLangFiles1870-1894_TB

------------------------------------------------------------------
Step 2. Set up Python (about 10 minutes)
------------------------------------------------------------------
Use Python 3.11. (fasttext-wheel, needed for language ID, has no
Apple-silicon build for Python 3.12 and no build at all for 3.13 or later.)

    python3.11 -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python -m pytest -q                # ~15 s, expect: 9 passed

------------------------------------------------------------------
Step 3. Download the external resources (about 10 minutes, ~3 GB)
------------------------------------------------------------------
    bash scripts/download_resources.sh      # NLLB and GlotLID language-ID
                                            # models, LDNOOBW bad-word list
    export HF_HOME=resources/hf_cache       # keeps HuggingFace downloads here
    python scripts/prepare_modern.py        # modern isiXhosa, ~1 min
    python scripts/prepare_reference_en.py  # English Wikipedia sample, ~1 min

Everything goes into resources/ (not part of the repository).
prepare_modern.py should report 1476 MasakhaNEWS, 1883 Wikipedia and 186
Vuk'uzenzele documents. Keep HF_HOME set for step 4: B1 downloads the six
tokenisers into it on first use.

------------------------------------------------------------------
Step 4. Run the experiments (about 45 minutes in total)
------------------------------------------------------------------
Run them in this order; later steps read earlier outputs.

    python experiments/e0_eda.py           # ~15 s  corpus statistics -> results/eda/
    python experiments/e1_splits.py        # ~1 s   splits -> data/splits.json (unchanged)
    python experiments/b0_normalise.py     # ~15 s  normaliser -> results/B0/coverage.json
    python experiments/b0_gold.py          # ~1 s   gold-set scores -> results/B0/gold_eval.json

    # A: the filter cascade, 2-3 min each (the first ~7 min) -> results/A1*/
    python experiments/a1_pipeline_survival.py                              # raw spelling
    python experiments/a1_pipeline_survival.py --normalised                 # normalised
    python experiments/a1_pipeline_survival.py --xh-stopwords               # isiXhosa stop words
    python experiments/a1_pipeline_survival.py --normalised --xh-stopwords
    python experiments/a1_pipeline_survival.py --unit paragraph             # robustness check
    python experiments/a1_pipeline_survival.py --unit paragraph --normalised

    python experiments/b1_tokenizer_fertility.py   # ~2 min   B1 -> results/B1/
    python experiments/b2_lm_2x2.py --jobs 3       # ~20-30 min  B2 -> results/B2/
                                                   # (--jobs = conditions run in
                                                   #  parallel; each needs a few GB
                                                   #  of RAM at order 12. If the
                                                   #  machine starts swapping, use
                                                   #  --jobs 1: slower but safe)
    python experiments/c1_summary.py               # ~5 s    gap decomposition and
                                                   # Figures 1-2 -> results/C1/,
                                                   # results/figures/

------------------------------------------------------------------
Step 5. Compare with the paper
------------------------------------------------------------------
The results/ files in the repository are the ones the paper was written
from, and rerunning step 4 reproduces them. Each number in the extended
abstract can be read from the files listed under results/ in section 1, e.g.
  results/C1/gate_attribution.csv   "survives": 0.0 for hist_xh, 0.326 for
                                    hist_en, 0.256 for modern_xh (xh stop words)
  results/C1/gap_decomposition.csv  orth_share 0.22 for CCNet perplexity
  results/B2/h4_interaction.csv     0.007 / 0.013 / 0.019 / 0.17

------------------------------------------------------------------
Optional: rebuild the paper
------------------------------------------------------------------
    cd paper/extended_abstract
    pdflatex main && bibtex main && pdflatex main && pdflatex main

==================================================================
3. NOTES
==================================================================
Design
* The unit of independence is the edition (paper + date): all splits and
  bootstrap CIs resample editions, never pages.
* All corpora are compared on 150-word windows (5,000 per historical corpus),
  so Gopher's length rules cannot drive differences; paragraphs are a
  robustness check (--unit paragraph).
* Modern isiXhosa is used in disjoint roles (src/modern.py): Wikipedia trains
  the perplexity reference (as in CCNet); news and government text are halved
  by document hash into the normaliser lexicon and a held-out control.
* The filters are re-implementations using the published thresholds; see the
  docstrings in src/filters/.
* B2 tunes the n-gram order (3-12) and discount (0.6-0.95) on the validation
  split and touches the test split once.

Steps that needed people (already done; do not rerun them)
* Gold set: `python experiments/b0_gold.py --make` sampled the 200 word types
  in data/gold_candidates.tsv (with an earlier version of the normaliser, so
  rerunning it now samples different words), and data/gold_forms/ holds the
  forms sent to volunteer isiXhosa speakers. `python
  scripts/collect_gold_forms.py FORMS_DIR` merged the returned forms into
  data/gold_normalisation.tsv. Only `python experiments/b0_gold.py` (the
  scoring, in step 4) needs to be rerun.
* Language-split audit: `python scripts/language_audit.py --make` wrote a form
  of 50 random blocks (kept outside the repository, as it quotes archive
  text); two team members marked it, and `--score FORM` wrote
  results/eda/language_audit.json.

Generative AI was used to help write code and text; see the AI declaration at
the end of the extended abstract.
