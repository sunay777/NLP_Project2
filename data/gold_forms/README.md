# Gold-set forms given to isiXhosa speakers

The 200 word types in `data/gold_candidates.tsv`, shuffled (fixed seed) and with the
normaliser's output hidden, so checkers cannot be biased by it.

- `isiXhosa_spelling_check_set1.docx` ... `set8.docx`: 25 words each (words 1-25, 26-50, ...),
  one set per volunteer. This is the version that was sent out.
- `isiXhosa_spelling_check_all200.docx`: the same 200 words in one form.

Each form contains the full instructions shown to volunteers. Returned forms are merged into
`data/gold_normalisation.tsv` (one row per word, `checked_by` = the volunteer) and scored with
`python experiments/b0_gold.py`.
