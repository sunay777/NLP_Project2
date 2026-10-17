# Draft answers: NeurIPS 2024 checklist + Wits Faculty of Science AI ethics statement

These are drafts to paste into the official `.tex` checklist and the Faculty form from Moodle.
The checklist is **mandatory** (no checklist, no mark). All three members must read and agree.

## NeurIPS 2024 paper checklist (draft answers)

1. **Claims.** [Yes] Abstract/intro claims (cascade survival, per-gate attribution, orthographic share of gaps) are exactly the results in Figs 1–2 and the Results section, with CIs.
2. **Limitations.** [Yes] Discussion → Limitations: re-implemented filters, register mismatch of the modern control, OCR not separated from lexical change, normaliser scope and unverified gold precision, unaudited language split, the failed paragraph-vs-article design.
3. **Theory.** [NA] No theoretical results.
4. **Reproducibility.** [Yes] README.txt gives the run order. Every script is seeded (`SEED=0`), the splits are committed (`data/splits.json`), and thresholds and hyper-parameter grids are in the scripts.
5. **Open access to data and code.** [Yes] for code (repository). The historical corpus is not redistributed; it is obtained from the isiXhosa Intellectual Traditions archive (Schoots et al., 2023) under its terms. Modern corpora and models are public on HuggingFace (download scripts provided).
6. **Experimental details.** [Yes] Methods: units, thresholds, splits (temporal per paper, 70/15/15 by edition), tuning on validation only (order 3–6, discount 0.6/0.75/0.9), single test evaluation.
7. **Statistical significance.** [Yes] 95% percentile bootstrap CIs, resampling editions (or modern documents), 500–1000 replicates.
8. **Compute.** [Yes] Everything runs on a laptop CPU (Apple M-series, 16 GB). A1 takes about 5 min per configuration and B2 about 1 h with 5 processes. No GPUs.
9. **Code of ethics.** [Yes].
10. **Broader impacts.** [Yes] Positive: documents representational harm by data curation against African-language heritage text and gives concrete curation recommendations. Negative: a normaliser could be used to "modernise" and overwrite historical orthography; we recommend parallel layers. Bad-word analysis quotes a historical racial slur only as reported speech, in order to show that the filter cannot distinguish contesting a slur from using it.
11. **Safeguards.** [NA] No model or scraped dataset released.
12. **Licences for existing assets.** [Yes] Cite the archive (Schoots et al., 2023), MasakhaNEWS (CC BY-NC 4.0, *verify*), Vuk'uzenzele (CC BY 4.0, *verify*), Wikipedia (CC BY-SA), LDNOOBW list (CC BY 4.0, *verify*), NLLB LID (CC BY-NC 4.0, *verify*), GlotLID (Apache 2.0, *verify*).
13. **New assets.** [Yes] The code and the normalisation map are documented in the README. The gold set is documented in `data/gold_normalisation.tsv`.
14. **Crowdsourcing / human subjects.** [NA] The only human input is a team member or acknowledged speaker checking ~200 word spellings. *If an external speaker helps, confirm with the lecturer whether ethics clearance is required.*
15. **IRB.** [NA] (see 14).

## Faculty of Science AI ethics statement (draft)

> Generative AI (Anthropic's Claude, used through Claude Code) was used to help write and debug
> analysis code (data loading, filter re-implementations, experiment scripts, plotting), to run the
> experiments, and to draft the text of this extended abstract. All research questions and hypotheses
> come from the group's own plan. All AI-generated code was reviewed and executed by the authors, and
> every number in the paper comes from those executions (`results/`). Every reference was checked
> against its publisher page by the authors. The authors take full responsibility for the content.

*Edit this so it says exactly how each member used AI tools. The Faculty form may ask for specific fields.*
