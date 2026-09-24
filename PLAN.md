# Research plan: *Erased and penalised*: 19th-century isiXhosa in modern NLP pipelines

Working title. Group project, COMS4054A/7062A/7066A Project 2. **Due 27 Oct 2026, 17:00.**
Preliminary contribution statement: two weeks before (~13 Oct).

## 1. Research question

> When the isiXhosa newspapers of 1870–1894 (Isigidimi samaXosa, Imvo Zabantsundu) meet a modern NLP
> pipeline, **how much of the text is filtered out, and how much worse is it modelled?** How much of
> that penalty comes from **fixable orthography** rather than genuine linguistic or cultural change?

Why it matters (the "language in culture / responsible NLP" hook). Pretraining corpora are built by
filter cascades that are tuned on English web text. If those filters discard early African
journalism, the first African-owned newspaper, protest poetry and letters from 27+ towns, then
LLMs inherit an archive with those voices removed. This is **representational harm by curation**.
It is also cheap to measure, and nobody has to train a large model to show it.

## 2. Evidence from EDA (already observed)

| | Isigidimi (1870–88) | Imvo (1884–94) |
|---|---|---|
| isiXhosa words | ~1.5M | ~3.8M |
| English words | ~0.02M | ~1.5M |
| `kuti` / `kuthi` | 1693 / 0 | 2325 / 0 |
| `umteto` / `umthetho` | 624 / 0 | 1975 / 0 |
| vote words / 100k, **whole run** (`ivoti`) | 3.9 | 19.6 (~5×) |
| vote words / 100k, **1884–88 overlap only** | 25.3 | 35.3 (1.4×) |
| `abelungu`/`umlungu` (white people) / 100k, overlap | 8.6 | 25.8 (3.0×) |
| `Tixo` (God) / 100k, overlap | 98.0 | 30.2 (0.3×) |

Note how the vote effect shrinks from ~5× to 1.4× once you restrict to the years both papers were
in print: much of the naive difference is **time**, not editorial line. That is exactly why the
overlap window is the right control for any Isigidimi-vs-Imvo claim.

**First smoke-test result (A1, Gopher only, 300 paragraphs each):** 0% of historical isiXhosa
paragraphs pass Gopher vs 78% of the English sections on the same pages. 296/300 fail the
*English* stop-word rule, and 168/300 also fail the 50-word minimum. isiXhosa is agglutinative
(one orthographic word often corresponds to several English words), so a word-count threshold is itself language-biased.

Aspiration is never written. The whole corpus is in a pre-reform orthography, so a modern system will
see it as "unlike isiXhosa".

## 3. Hypotheses

- **H1 (erasure).** A standard cascade (LID → Gopher → C4 bad-words → CCNet perplexity) keeps a much
  smaller share of historical isiXhosa than of (a) modern isiXhosa and (b) the historical English
  sections that sit on the same pages.
- **H2 (which gate).** Most of the loss happens at *language-agnostic-looking* rules that are in fact
  English- or modern-biased: LID confidence, Gopher's English stop-word rule, and the Wikipedia-referenced
  perplexity filter. Bad-words removes little text, but what it removes is politically salient
  (quoted slurs in anti-colonial writing).
- **H3 (orthography).** Spelling normalisation (i) raises LID confidence and survival, (ii) lowers
  tokeniser fertility, and (iii) lowers LM bits/word. It closes a *measurable fraction* of the gap, but not
  all of it. The residual is lexical/cultural change plus OCR noise.
- **H4 (modern data).** With a fixed training budget, mixing in modern isiXhosa helps more *after*
  normalisation than before, because the two corpora then share surface forms.

## 4. Design

### A. Pipeline survival (`experiments/a1_pipeline_survival.py`)
- Units: paragraphs (≥20 words); pages as a robustness check.
- Corpora: `hist_xh`, `hist_en` (same pages, target=eng), `modern_xh` (control).
- Filters: each applied independently (attribution) and as a cascade (end-to-end survival).
- Metrics: survival rate by units **and by words**; per paper × year; LID label distribution
  (what *does* the model think this is?). Report 95% edition-cluster bootstrap CIs.
- Sweeps: LID threshold ∈ {0.3, 0.5, 0.65, 0.8}; Gopher stop-words EN vs isiXhosa (ablation).
- Qualitative: sample 10 removed paragraphs per gate, translate/gloss, say what was lost.
- Link to B: rerun with `--normalised`. **Headline figure candidate:** cascade survival for
  modern / historical-normalised / historical-raw.

### B. Orthography gap decomposition
- **B0 normaliser** (`b0_normalise.py`): rule candidates (insert aspiration h; join split prefixes)
  ranked by modern-lexicon frequency. Report coverage before/after. **Evaluate precision on a
  hand-checked gold set** (`data/gold_normalisation.tsv`, ~200 types stratified by frequency), ideally
  checked by an isiXhosa speaker. Credit them in the acknowledgements.
- **B1 tokeniser tax** (`b1_tokenizer_fertility.py`): fertility / continued-word rate for XLM-R,
  AfroXLMR, NLLB, Serengeti, mT5 on modern vs hist-raw vs hist-norm.
- **B2 controlled LM experiment** (`b2_lm_2x2.py`): {raw, norm} × modern fraction {0, .25, .5, .75}
  at a **fixed character budget**. Character KN n-gram. Order/discount tuned on val. Temporal test split
  (last 15% of editions of each paper). Metric: **bits per original word** (comparable across spellings).
- **B3 (optional)** (`b3_mlm_pll.py`): AfroXLMR pseudo-log-likelihood raw vs norm, zero training.

### Gap decomposition (the Discussion's spine)
For any metric M: `gap_total = M(hist_raw) − M(modern)` splits into
`orthography = M(hist_raw) − M(hist_norm)` plus `residual = M(hist_norm) − M(modern)`.
Report the orthographic share with CIs. The residual is where culture lives: obsolete vocabulary,
loanwords of the period (voti, rulumente), and OCR.

## 5. Controls and threats to validity (write these up, since they earn marks)
- **Leakage:** all splits and bootstraps grouped by edition.
- **Budget confound:** B2 fixes the training budget, so "modern helps" ≠ "more data helps".
- **Normaliser circularity:** the lexicon comes from modern text, so a normalised text that looks more
  modern is partly by construction. Mitigate by using separate modern sources for *lexicon* and *evaluation*
  (e.g. Wikipedia for the lexicon, MasakhaNEWS for the control), and report precision on the gold set.
- **OCR noise vs orthography:** stratify results by the `ocr_noise` covariate, or match on it.
- **Register:** modern news or Wikipedia is not 1880s journalism. Use multiple modern sources and report per source.
- **English control is not neutral:** 19th-century English is also "old", which is useful. If
  hist_en survives far better than hist_xh, then age alone doesn't explain the loss.
- **Our own filters are re-implementations** (thresholds from papers). Say so.
- **Language-split tracker errors:** audit ~50 blocks by hand and report the error rate.

## 6. Responsible NLP content for the Discussion
- Curation as a normative act: whose "quality" do Gopher and CCNet encode?
- Slur handling: blanket bad-word filters vs historical documents that contest those slurs.
- Should the archive be normalised at all? Normalisation helps tools but can erase the historical
  orthography, which is itself cultural evidence. Recommend keeping both layers (parallel text).
- Community and data sovereignty: the archive's custodians and the descendants of its authors. Refer to
  the Schoots et al. (2023) collection pipeline and its licensing.
- NeurIPS ethics checklist and Faculty AI statement (mandatory appendices).

## 7. Figures for 2 pages (max 3)
1. Cascade survival (modern / hist-norm / hist-raw / hist-en), with per-gate attribution.
2. Gap decomposition bar chart across metrics (LID confidence, fertility, bits/word).
3. (Optional) B2 line plot: bits/word vs modern fraction, raw vs norm.

## 8. Timeline
| Week | Work |
|---|---|
| 24–30 Sep | Read Schoots et al. Download resources. EDA, splits. Run A1 raw. |
| 1–7 Oct | B0 normaliser + gold set. A1 normalised. B1. |
| 8–14 Oct | B2 sweep. Ablations and threshold sweeps. **Contribution statement (~13 Oct).** |
| 15–21 Oct | Qualitative analysis of removed text. Figures. Draft abstract. |
| 22–27 Oct | Edit to 2 pages. Ethics checklist and AI statement. README/requirements. Submit. |

## 9. References to use (verify every entry before citing)
Schoots et al. 2023 (RAIL); Raffel et al. 2020 (C4); Rae et al. 2021 (Gopher); Wenzek et al. 2020 (CCNet);
Dodge et al. 2021 (documenting C4: filters remove minority-identity text); Kreutzer et al. 2022 (audit of
multilingual web corpora); NLLB Team 2022; Kargaran et al. 2023 (GlotLID); Rust et al. 2021 (tokenizer
quality); Petrov et al. 2023 (tokenizer unfairness across languages); Alabi et al. 2022 (AfroXLMR);
Adelani et al. 2023 (MasakhaNEWS); Salazar et al. 2020 (MLM scoring); Bird 2020 (decolonising
language technology); Bender & Friedman 2018 (data statements).
