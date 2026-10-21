# Project status (18 Oct 2026). Due 27 Oct, 17:00.

## Done (all numbers regenerate from the scripts; see README run order)

| Part | Owner (contribution statement) | State | Key output |
|---|---|---|---|
| Env + resources | all | done: LID models, LDNOOBW, modern isiXhosa (MasakhaNEWS 1.5k docs, Wikipedia 1.9k, Vuk'uzenzele 186 = 0.70M words), English Wikipedia reference | `resources/` (git-ignored) |
| EDA, splits | Harshil | done (before this session) | `results/eda/`, `data/splits.json` |
| B0 normaliser | Matthew | done: 3,942 types mapped, 9.0% of tokens changed, lexicon coverage 63.2% → 68.4% | `results/B0/` |
| B0 gold set | Matthew | **candidates generated, NOT checked** | `data/gold_candidates.tsv` |
| A1 filter audit | Sunay | done: raw / normalised / isiXhosa-stop-word ablation / paragraph units, LID threshold sweep, qualitative examples, bad-word hits | `results/A1*/`, `results/C1/` |
| B1 tokenisers | Matthew | done: 6 tokenisers × {hist_raw, hist_norm, modern, hist_en} | `results/B1/` |
| B2 char-LM 2×5 | Harshil | done (3M-char budget; tuning hit grid edge order 6 / d 0.9, so consider extending the grid to 7–8 / 0.95) | `results/B2/` |
| C1 gap decomposition + figures | Harshil | done | `results/C1/`, `results/figures/fig1_cascade.png`, `fig2_gap.png` |
| Extended abstract | all | **full draft** | `paper/extended_abstract/main.tex` |
| Ethics checklist + AI statement | Sunay / Matthew | **draft answers** | `paper/ethics_drafts.md` |

## Headline findings (rerun 18 Oct with disjoint modern-data roles; window units, n=5000 per historical corpus)
- Full cascade survival: 1880s isiXhosa **0.0%**, 1880s English on the same pages **32.6%** [31.2, 34.3], modern isiXhosa 0.3%.
- With **isiXhosa stop words** in Gopher (fair rule): modern isiXhosa **25.6%** [22.8, 28.3] survives, 1880s isiXhosa **0.0%** (0.1% normalised). This is the decisive H1 comparison and is now in Fig. 1.
- LID is not the bottleneck (92–95% of historical windows kept at thresholds 0.3–0.8; modern 98–100%).
- The isiXhosa stop-word list is itself modern (*kwaye*, *uthi*; the 1880s write *uti*): it removes 32% of historical vs 8% of modern windows.
- The perplexity filter (isiXhosa Wikipedia reference) removes 100% of historical vs 72% of modern windows. Normalisation closes **22% [19, 25]** of that gap.
- Normaliser (lexicon half only): 1,577 types, 7.1% of tokens; held-out coverage 52.2% → 56.2% (same gain as on its own lexicon).
- Tokenisers: isiXhosa costs 1.1–1.6× more tokens per letter than English; 1880s vs modern differ by <6%, so no orthographic share is reported.
- B2 (orders 3–12): in-domain raw = normalised (3.02 vs 3.01 bits/raw letter). Modern-only model: 4.85 on 1880s vs 2.30 on held-out modern; normalising closes 7%. H4 interaction 0.007/0.013/0.019/0.17 bits/letter at 25/50/75/100% modern, all CIs exclude 0 (small but real).
- Bad words (1.3% removed) are all false positives.

## Design changes vs PLAN.md (report these in the final contribution statement / limitations)
1. **Units = 150-word windows** for every corpus, not paragraphs. Paragraphs vs whole modern articles confounded Gopher's 50-word rule (56% vs 8% failing on length alone). Paragraph runs are kept as a robustness check.
2. **Modern data in disjoint roles** (`src/modern.py`): Wikipedia = perplexity reference; news + government text halved by document hash into the normaliser lexicon and a held-out control (A1 control, B1 modern, B2 modern test). Earlier runs scored the control in-sample and built the lexicon from the perplexity reference.
3. **B1 metric is tokens per letter**, not fertility. Disjunctive 1880s word division makes words shorter, so fertility came out *lower* for historical text and *rose* after normalisation.
4. B2 adds modern share = 1.0, a held-out modern test set, a raw-character budget for both spellings, orders 3–12, and a paired CI for the H4 interaction.
5. Bug fixed: prefix-joining used to merge words across paragraph breaks.

## Still needs a human (in priority order)
1. **Gold set: with an isiXhosa speaker** (form: `isiXhosa_spelling_check.docx`, sent 18 Oct). When it returns, fill `modern` + `checked_by`, append to `data/gold_normalisation.tsv`, run `python experiments/b0_gold.py`, and put precision/recall in the paper (replaces "is being checked"); credit the speaker in the acknowledgements.
2. **Compile on Overleaf and re-check 2 pages.** A local tectonic build fits the main text on exactly 2 pages after the title page; Overleaf's pdflatex may differ slightly. Check whether the two `\phantom{.}\newpage` lines (title page) are in the Moodle template.
3. ~~Verify every reference~~ Done 18 Oct: all 15 checked against publisher records; missing pages/DOIs added.
4. Paste the official NeurIPS checklist `.tex` and the Faculty AI form from Moodle, using `paper/ethics_drafts.md`. Check the dataset licences marked *verify*.
5. Hand-audit ~50 blocks of the archive's language split (PLAN §5) and report the error rate, or keep it as a limitation (currently listed).
6. Final contribution statement (list the design changes above).
7. Optional: B3 (AfroXLMR pseudo-log-likelihood), not run.
