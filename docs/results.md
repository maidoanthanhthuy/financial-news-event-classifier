# Results

All scores are on the blind test set `data/labeled/test_blind.csv` (100 items,
human labels, see [data_process.md](data_process.md)). Metrics come from
`python -m src.classify.evaluate`.

## Stage 2: LLM baseline

### Setup
- Classifier: `classify_news(title, sapo)` in `src/classify/llm_label.py`, model
  gpt-6-luna, structured output (Pydantic), 6 labels. The system prompt is
  `docs/labeling_guide.md` as of commit 767afad.
- Run: `python -m src.classify.llm_test` -> `data/processed/test_llm_v2.csv`
  (local). Cost: $0.0225 for 100 items; all answers parsed.
- **The labeling guide was not changed based on test-set errors.** The guide used
  here was fixed before this run, and the errors below are reported, not fixed in
  the guide, so the test score stays an honest estimate.

### Scores (v2, 6 labels, 100 items)
Accuracy 0.800, macro-F1 0.793.

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| business_results | 0.783 | 1.000 | 0.878 | 18 |
| dividends_issuance | 0.950 | 1.000 | 0.974 | 19 |
| macro_policy | 0.867 | 0.650 | 0.743 | 20 |
| ma_investment | 0.714 | 0.714 | 0.714 | 7 |
| legal_personnel | 0.882 | 1.000 | 0.938 | 15 |
| other | 0.556 | 0.476 | 0.513 | 21 |

Confusion matrix (rows = human label, columns = LLM label):

| | business_results | dividends_issuance | macro_policy | ma_investment | legal_personnel | other |
|---|---|---|---|---|---|---|
| business_results | 18 | 0 | 0 | 0 | 0 | 0 |
| dividends_issuance | 0 | 19 | 0 | 0 | 0 | 0 |
| macro_policy | 0 | 0 | 13 | 0 | 0 | 7 |
| ma_investment | 1 | 0 | 0 | 5 | 0 | 1 |
| legal_personnel | 0 | 0 | 0 | 0 | 15 | 0 |
| other | 4 | 1 | 2 | 2 | 2 | 10 |

### Comparison with the draft LLM labels
The draft labels were made during data collection with an earlier version of the
guide (`data/processed/test_llm_labels.csv`, local).

| Run | Accuracy (6 labels) | Macro-F1 (6 labels) | Accuracy (5 labels, no `other`) | Macro-F1 (5 labels, no `other`) |
|---|---|---|---|---|
| Draft (earlier guide) | 0.730 | 0.666 | 0.924 | 0.910 |
| v2 (current guide) | 0.800 | 0.793 | 0.886 | 0.919 |

- The 6-label comparison favors v2 by design: the test items were sampled from
  items the draft run did **not** label `other`, so the draft never predicts
  `other` on this set.
- The 5-label columns use the 79 items humans did not label `other`; a prediction
  of `other` counts as a miss. Here the two runs are close: v2 has a similar
  macro-F1 but lower accuracy, because it moves 8 event items to `other`.

### Main weaknesses of v2
1. **`macro_policy` pushed to `other`** (7 of 20 macro items). These are statements
   or directions from government leaders and comments on policy proposals with no
   concrete decision in the title (ids 32, 35, 68, 70, 86, 97, 99).
2. **`business_results` for look-back or non-event articles with profit figures**
   (4 of 21 `other` items): a project delivery outlook (id 13), profit of a foreign
   group's plant (id 57), an article on how a company might reach its profit plan
   (id 88) and a sector profit ranking (id 98).

The draft run had the opposite habit: it labeled many `other` items as
`ma_investment`.

## Stage 3: VN30 ticker extraction (dictionary)

### Setup
- `extract_tickers(title, sapo, aliases)` in `src/classify/tickers.py` matches the
  alias dictionary `data/reference/vn30_aliases.csv` (ticker, company names and
  common names, both tone-mark styles such as Hòa/Hoà).
- All-caps aliases (tickers, BIDV, TCBS, MB) must match case and stand alone, so
  "gas" does not match `GAS`; names match without case. Text is NFC-normalized.
- The longest alias wins ("Masan Consumer" -> `MCH`, not also `MSN`). Rows with an
  empty ticker block names of companies outside VN30 that contain a VN30 name
  (e.g. FPT Shop, FPT Long Châu, Masan MEATLife, Chứng khoán BIDV, The CrownX).
- Run: `python -m src.classify.tickers` (blind test set) and
  `python -m src.classify.tickers check` (train sample).

### Blind test set (100 items, `tickers` column)
Precision 1.00, recall 1.00, 100 of 100 items match exactly (14 items have
tickers). The dictionary was not changed based on test-set errors. The name table
in `data_process.md` was written while tagging the test set, so the dictionary and
the test tags are not fully independent, and only 14 items carry tickers.

### Manual check on train items (40 items)
- `extract_tickers` finds tickers in 89 of 560 train items. A random sample (seed
  42) of 20 items with tickers and 20 without was checked by hand against the
  tagging rules in `data_process.md`.
- Before the fix: precision 0.96 (24 of 25 predicted tickers correct), 39 of 40
  items exact. The one error: id 325, "The CrownX (TCX)" (a Masan subsidiary)
  matched the `TCX` ticker. Fixed by adding "The CrownX" and "The CrownX (TCX)"
  to the blocked names; after the fix all 40 items match. The fix was made on this
  same sample, so the 40/40 is not an independent score.
- **Recall 1.00 holds only for the 20 sampled items without tickers** (none of them
  names a VN30 company). It is not a recall estimate for the full train set,
  because the sample was split by predicted tickers.
- In 4 items the company is named, so the tag is correct under the rules, but it is
  not the main subject of the news: id 194 (Xanh SM; Vingroup named in the sapo),
  132 (a Masan member company), 274 (VinSpace; new CEO formerly at FPT) and 396
  (Aeon Financial; SeABank's finance company). Such items add noise to the event
  study.

### `main_tickers`: tickers named in the title
- `main_tickers(title, sapo, aliases)` keeps only the tickers from
  `extract_tickers` whose name or ticker appears in the title. `extract_tickers`
  is unchanged.
- Of the 4 items above, it drops 194 and 396 but keeps 132 (`MSN`) and 274
  (`FPT`), because Masan and FPT are named in those titles.
- On the blind test set it drops `CTG` from id 17 (named only in the sapo) and
  `HDB`, `TCB` from id 56 (keeps `VPB`); the other 12 tagged items are unchanged.
