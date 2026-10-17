# Project status (17 Oct 2026). Due 27 Oct, 17:00.

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

## Headline findings (window units, n=5000 per historical corpus)
- Survival of the full cascade: 1880s isiXhosa **0.0%**, modern isiXhosa 0.4%, 1880s English on the same pages **32.6%**.
- LID is not the bottleneck (92–95% of historical windows kept as isiXhosa at thresholds 0.3–0.8).
- Gopher's English stop-word rule removes about 96% of *all* isiXhosa. With isiXhosa stop words: modern keeps 90%, historical only 71%, because modern function words (*kwaye*) are rare in the 1880s.
- The CCNet-style perplexity filter (isiXhosa Wikipedia reference) removes 100% of historical vs 72% of modern windows. Normalisation closes **26% [22, 29]** of that gap but does not change survival.
- Tokenisers: isiXhosa costs 1.1–1.6× more tokens per letter than English; 1880s vs modern spelling barely matters.
- B2: in-domain char LM, raw = normalised (3.10 vs 3.11 bits/raw letter). A modern-only model pays 4.89 vs 2.49 on modern text; normalising closes 9%. Modern data at a fixed budget hurts the 1880s test set, and normalisation's benefit grows with the modern share.
- Bad words (1.3% removed) are all false positives: surnames (Dick, Cumming), "Rape" (the crop), *nambla* (= modern *namhla*, "today"), and one slur quoted in a report contesting it.

## Design changes vs PLAN.md (report these in the final contribution statement / limitations)
1. **Units = 150-word windows** for every corpus, not paragraphs. Paragraphs vs whole modern articles confounded Gopher's 50-word rule (56% vs 8% failing on length alone). Paragraph runs are kept as a robustness check.
2. **Modern sources split.** Wikipedia is the perplexity reference and news + government text is the control. The first smoke test scored the control in-sample.
3. **B1 metric is tokens per letter**, not fertility. Disjunctive 1880s word division makes words shorter, so fertility came out *lower* for historical text and *rose* after normalisation.
4. B2 adds modern share = 1.0 (a modern-only model reads 1880s text) and a held-out modern test set for the decomposition.
5. Bug fixed: prefix-joining used to merge words across paragraph breaks.

## Still needs a human (in priority order)
1. **Check the gold set.** Fill `modern` + `checked_by` in `data/gold_candidates.tsv`, append the rows to `data/gold_normalisation.tsv`, run `python experiments/b0_gold.py`, and put precision/recall in the paper. Ideally an isiXhosa speaker does this; credit them in the acknowledgements.
2. **Compile the paper** on Overleaf (no LaTeX on this machine): upload `paper/extended_abstract/` and `results/figures/fig1_cascade.png`, `fig2_gap.png` into `figures/`. Check it fits **2 pages**. The draft is ~1,450 words + 2 figures, so it probably needs trimming (cut Methods detail first).
3. **Verify every reference** in `references.bib` (the RAIL and MasakhaNEWS entries were checked online; the rest are from memory).
4. Paste the official NeurIPS checklist `.tex` and the Faculty AI form from Moodle, using `paper/ethics_drafts.md`. Check the dataset licences marked *verify*.
5. Hand-audit ~50 blocks of the archive's language split (PLAN §5) and report the error rate, or list it as a limitation (currently listed).
6. Final contribution statement.
7. Optional: B3 (AfroXLMR pseudo-log-likelihood), not run.
