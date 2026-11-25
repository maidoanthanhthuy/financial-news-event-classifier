# Results

Classification scores (Stages 2, 4 and 6) are on the blind test set
`data/labeled/test_blind.csv` (100 items, human labels, see
[data_process.md](data_process.md)), from `python -m src.classify.evaluate`.
Stage 5 (event study) uses 169 news events with human or LLM labels; see its
Setup section.

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

## Stage 3: news explanation for beginners

### Setup
- `explain_news(title, sapo, label, tickers)` in `src/classify/explain.py` returns
  at most 3 short Vietnamese sentences: what the news says (financial terms
  explained in parentheses) and what a beginner should understand from it. No price
  predictions, no buy/sell advice, no numbers that are not in the news.
- Model gpt-6-luna with `reasoning_effort="none"` (no reasoning tokens). The system
  prompt is `docs/explain_prompt.md`. Callers pass tickers from `main_tickers`.
- Run: `python -m src.classify.explain` -> `data/processed/explain_samples.csv`
  (local): 2 train items per event label (10 items, seed 42).

### Prompt tuning on 10 train items
The same 10 train items were used in every round; the explanations were read by
hand.

| Round | Prompt change | Cost (10 items) |
|---|---|---|
| v1 | First prompt: 2-3 sentences, explain terms, no prices or advice | $0.0015 |
| v2 | Short sentences, no "Bài viết cho biết" openers, keep the news wording, `reasoning_effort="none"` | $0.0010 |
| v3 | Explain only specialist financial terms, keep all title details, never comment on missing information | $0.0011 |
| v4 (final) | One worked example, terms in parentheses right after the word, title details only, concrete last sentence, neutral wording for sensational headlines | $0.0013 |

- v1 issues: long sentences joined by semicolons, generic openers, unexplained
  terms, one reinterpreted phrase ("trữ 12.000 tỷ" written as "dành"), and
  reasoning tokens up to 4x the visible text.
- v2 and v3 fixed openers and term notes, but the model dropped title details to
  avoid wording problems, still used semicolons, and its closing sentences were
  generic.

### Remaining limitations (v4)
- **Sometimes changes the original wording**: "cầm khối tiền" became "trữ" (id 24),
  likely copied from the "trữ" example in the prompt.
- **Drops minor details**: id 308 lost "5,4 tỷ tấn, hơn 1.500 năm" and rewrote
  "quả đấm thép" as "dự án".
- **Closing sentences repeat one pattern**: 7 of 10 end with "nên hiểu đây là X,
  không phải Y". Correct, but repetitive across many items.
- Only 10 items were checked by hand. There is no automatic check of the output.

## Stage 3: LangGraph pipeline

### Setup
- `src/classify/pipeline.py`, state: title, sapo, label, reason, tickers,
  main_tickers, explanation.
- Nodes: `classify` (`classify_news`) -> `tickers` (`extract_tickers` and
  `main_tickers`) -> branch: label `other` goes to a fixed explanation ("Tin này
  không thuộc 5 loại sự kiện doanh nghiệp mà công cụ theo dõi.") and ends; any
  other label goes to `explain` (`explain_news`, given `main_tickers`).
- `run_pipeline(title, sapo)` returns the final state for the app.
- The LLM functions are passed to `build_graph`, so tests replace them with fakes
  and run without an API key (both branches tested).
- Run: `python -m src.classify.pipeline` (4 random event items and 1 `other` item
  from the train set, seed 42, real API calls).

### Trial on 5 train items
All 5 predicted labels match the human labels.

| id | Human label | Predicted | main_tickers | Time | LLM calls | Cost |
|---|---|---|---|---|---|---|
| 133 | dividends_issuance | dividends_issuance | - | 5.0 s | 2 | $0.00033 |
| 32 | business_results | business_results | - | 5.2 s | 2 | $0.00034 |
| 211 | legal_personnel | legal_personnel | - | 4.2 s | 2 | $0.00031 |
| 487 | macro_policy | macro_policy | - | 5.2 s | 2 | $0.00038 |
| 539 | other | other | STB | 1.6 s | 1 | $0.00021 |

- An event item takes about 4-5 s and $0.0003 (two sequential API calls); an
  `other` item takes about 1.6 s and $0.0002 (classification only).
- Every number and detail in the explanations appears in the title or sapo.
- 5 items is a small trial; the 4 event items include no `ma_investment` item.

### Limitations
- **The prompt is not always followed**: id 211 ends by saying the news does not
  name the appointed people (comments on missing information are forbidden); id
  487 lists 7 place names from the sapo and says "gồm" although the news mentions
  10 areas.
- **Title-based rules can miss sapo details**: id 539 is labeled `other` (the title
  is about staff cuts), but the sapo also says Sacombank plans to buy a securities
  company, which could be an `ma_investment` event.

## Stage 4: fine-tuned PhoBERT

### Setup
- Model `vinai/phobert-base-v2`, 6 labels, trained on `data/labeled/news_labeled.csv`
  (560 items). Input: title + sapo, word-segmented with underthesea,
  `max_length=128` (longest item is 105 tokens, so nothing is truncated).
- Split: 85/15 by label with a fixed seed (42): 476 train, 84 validation items.
  The split is the same for every run; the training seed changes only model
  initialization and data order.
- Training (`src/finetune/train.py`): learning rate 2e-5, batch size 16, at most 10
  epochs, weight decay 0.01, early stopping on validation macro-F1 (patience 2),
  best epoch kept. Run on Colab (T4 GPU) with `notebooks/finetune_colab.ipynb`.

### Validation results (3 seeds)

| Seed | Macro-F1 | Accuracy | Best epoch | Epochs run |
|---|---|---|---|---|
| 42 | 0.830 | 0.845 | 8 | 10 |
| 43 | 0.840 | 0.857 | 6 | 8 |
| 44 | 0.851 | 0.869 | 5 | 7 |

- Validation macro-F1: **0.841 ± 0.010** (mean ± sample std over 3 seeds).
- Seed 44 is the best run and is published on the Hugging Face Hub as
  `MaiDoanThanhThuy/phobert-vn30-news-events`; `predict_phobert(title, sapo)` in
  `src/finetune/predict.py` loads it from there.
- Seed 42 used all 10 epochs (best at epoch 8), so early stopping did not trigger
  for that run.
- **The validation score is slightly optimistic**: the same 84 validation items
  were used to pick the best epoch and the best seed. The blind test set has not
  been used yet; its score is the one to compare with the LLM baseline.

## Stage 5: event study

### Setup

- Events: raw news with exactly one main VN30 ticker (422 items), labeled by a
  human where available (74) and by the LLM otherwise (348). `other` and
  `macro_policy` (5 items) are dropped. Same ticker and label news less than 7
  calendar days apart are merged (first kept): 175 events, 28 tickers
  (`python -m src.event_study.build_events events`).
- Prices: daily adjusted closes from the KBS API, 2024-09-04 to 2026-11-10, for
  the event tickers and VNINDEX, loaded into DuckDB
  (`src/event_study/fetch_prices.py`, `load_prices.py`). 6 events published
  before the stock was listed (TCX 5, VPL 1) are dropped: **169 events**.
- Trading calendar = dates with a VNINDEX price. Session 0 = first session on or
  after the publication date; news after 15:00 counts from the next day.
- Abnormal return (AR) for k = 1, 3, 5 = stock return minus VNINDEX return,
  from the close of session -1 to the close of session k-1
  (`src/event_study/event_returns.sql`). All 169 events have all 3 windows.
- By label and k: mean, median, sd, share of AR > 0, 95% t-based CI, one-sample
  t-test and Wilcoxon signed-rank test against 0 (`src/event_study/summary.py`,
  `data/processed/event_summary.csv`, `notebooks/event_study.ipynb`).
- Robustness: the same statistics without VIC, the most frequent ticker
  (32 of the 169 events: 20 `ma_investment`, 9 `business_results`, 3
  `dividends_issuance`).

### All events (169)

Mean AR in %, then p-values: t-test / Wilcoxon.

| Label | n | k = 1 | k = 3 | k = 5 |
|---|---|---|---|---|
| business_results | 65 | -0.16 (0.445 / 0.390) | -0.49 (0.178 / 0.054) | -0.10 (0.831 / 0.688) |
| dividends_issuance | 33 | +0.10 (0.744 / 0.502) | +0.60 (0.348 / 0.584) | +0.18 (0.851 / 0.874) |
| legal_personnel | 22 | -0.09 (0.847 / 0.566) | -0.25 (0.804 / 0.545) | -0.24 (0.855 / 0.633) |
| ma_investment | 49 | -0.02 (0.951 / 0.168) | -0.50 (0.132 / 0.108) | -0.29 (0.635 / 0.235) |

### Without VIC (137)

| Label | n | k = 1 | k = 3 | k = 5 |
|---|---|---|---|---|
| business_results | 56 | -0.16 (0.441 / 0.396) | -0.41 (0.261 / 0.090) | -0.10 (0.842 / 0.738) |
| dividends_issuance | 30 | -0.05 (0.857 / 0.360) | +0.15 (0.768 / 0.871) | -0.27 (0.712 / 0.730) |
| legal_personnel | 22 | -0.09 (0.847 / 0.566) | -0.25 (0.804 / 0.545) | -0.24 (0.855 / 0.633) |
| ma_investment | 29 | +0.04 (0.905 / 0.213) | -0.78 (0.038 / 0.018) | -1.05 (0.053 / 0.039) |

VIC has no `legal_personnel` event, so that row does not change.

### Findings

- **No label has a mean AR clearly different from 0.** In the full sample every
  95% CI contains 0, mean ARs are small (at most 0.6%) next to their sd
  (1.7-6.0%), and the share of positive ARs is 32-55%.
- **M&A news without VIC is only a hint.** Its AR is negative at k = 3 (-0.78%,
  t-test p = 0.038, Wilcoxon p = 0.018) and borderline at k = 5. There are 12
  tests per sample (4 labels x 3 windows), 24 in total, so 1-2 values below 0.05
  are expected by chance. With a Bonferroni correction (0.05 / 12 = 0.0042) no
  result is significant. The 20 VIC M&A events have a mean AR near 0 at k = 3
  (-0.09%) and positive at k = 5 (+0.81%), so the result depends on whether one
  ticker is included.
- Wilcoxon p-values are often lower than t-test p-values (e.g.
  `business_results` k = 3: 0.054 vs. 0.178): most ARs are slightly negative
  (median below 0) while a few large positive ARs pull the mean toward 0.

### Why no clear effect

1. **Good and bad news share a label and cancel out.** `business_results`
   holds both record profits and losses; `ma_investment` holds both purchases
   and divestments. The label gives the event type, not its direction.
2. **Prices may react before the news is published.** Results, dividends and
   deals are often expected or leaked; session -1 may already include the move.
3. **Small samples and noisy labels.** 22-65 events per label, and 79% of the
   labels come from the LLM (about 80% accurate on the blind test set), so some
   events sit in the wrong group and dilute any effect.

### Next steps

- Add a good/bad sentiment label per news item and measure AR by event type and
  direction.
- Measure a pre-event window (e.g. sessions -5 to -1) to see if prices move
  before publication.

## Stage 6: PhoBERT vs. LLM on the blind test set

### PhoBERT, 3 seeds (Colab, `notebooks/finetune_colab.ipynb`)

| Seed | Accuracy | Macro-F1 |
|---|---|---|
| 42 | 0.800 | 0.768 |
| 43 | 0.820 | 0.788 |
| 44 | 0.800 | 0.767 |

- Test accuracy **0.807 ± 0.012**, macro-F1 **0.774 ± 0.012** (mean ± sample std).
- **Seed 44 was the best seed on validation (macro-F1 0.851) but is the weakest on
  test** (0.767, just below seed 42 at 0.768). The ranking of seeds on 84
  validation items does not carry over to the test set; the seed differences are
  within noise.
- Seed 44 stays the published model: it was chosen on validation. Seed 43 scores
  higher on test, but switching to it would mean choosing a model on the test set.
- **`other` recall is low in all 3 seeds**: 0.381 (seed 42), 0.333 (43), 0.286 (44).
- The seed 44 scores on Colab (GPU) match the laptop run (CPU, model from the Hub)
  exactly, so the published model is the one trained on Colab.

### Published model (seed 44) vs. LLM v2
Run on a laptop: `python -m src.finetune.test_phobert` ->
`data/processed/test_phobert.csv` (local).

| Label | PhoBERT F1 | LLM v2 F1 | Support |
|---|---|---|---|
| business_results | 0.878 | 0.878 | 18 |
| dividends_issuance | 0.944 | 0.974 | 19 |
| macro_policy | 0.923 | 0.743 | 20 |
| ma_investment | 0.545 | 0.714 | 7 |
| legal_personnel | 0.882 | 0.938 | 15 |
| other | 0.429 | 0.513 | 21 |
| **Accuracy** | 0.800 | 0.800 | 100 |
| **Macro-F1** | 0.767 | 0.793 | 100 |

PhoBERT confusion matrix (rows = human label, columns = PhoBERT):

| | business_results | dividends_issuance | macro_policy | ma_investment | legal_personnel | other |
|---|---|---|---|---|---|---|
| business_results | 18 | 0 | 0 | 0 | 0 | 0 |
| dividends_issuance | 0 | 17 | 0 | 1 | 0 | 1 |
| macro_policy | 0 | 0 | 18 | 1 | 1 | 0 |
| ma_investment | 0 | 0 | 0 | 6 | 1 | 0 |
| legal_personnel | 0 | 0 | 0 | 0 | 15 | 0 |
| other | 5 | 0 | 1 | 7 | 2 | 6 |

### Is the difference significant?
`python -m src.classify.compare` (PhoBERT seed 44 = a, LLM v2 = b, same 100 items).

- **McNemar test (exact)**: both correct 72, both wrong 12, only PhoBERT correct
  8, only LLM correct 8, p = 1.00. The two models make the same number of errors,
  just on different items.
- **Bootstrap (10,000 resamples of the 100 items, seed 42), 95% percentile CIs**:

| | Macro-F1 | 95% CI |
|---|---|---|
| PhoBERT (seed 44) | 0.767 | 0.681 - 0.840 |
| LLM v2 | 0.793 | 0.697 - 0.865 |
| Difference (PhoBERT - LLM) | -0.026 | -0.112 - 0.064 |

- The CI of the difference contains 0 (PhoBERT is ahead in 28% of resamples).
  **On 100 test items there is no significant difference** between the two models
  in accuracy or macro-F1. The test set is too small to separate them; a
  difference of about 0.1 macro-F1 would be needed to be detectable.

### Cost and speed

| | PhoBERT (seed 44) | LLM v2 (gpt-6-luna) |
|---|---|---|
| Time per item | 0.045 s (mean, median 0.039 s; laptop CPU, 16 threads) | 2.5 s (mean, median 2.3 s, range 1.5-4.7 s; 20 test items) |
| Time for 1,000 items | about 45 s | about 42 min (one call at a time) |
| Cost for 1,000 items | $0 (local CPU; training on free Colab GPU) | about $0.22 (from the 100-item run, $0.0225) |

- LLM times were measured separately on the first 20 test items (cost $0.0044),
  because the earlier test run did not record time. They include network delay.
- The gpt-6-luna price comes from aggregator sites and is not yet checked on
  OpenAI's pricing page.
- LLM calls could run in parallel to cut the wall time; the cost stays the same.

### Findings
- **Same accuracy, lower macro-F1 for PhoBERT** (0.767 vs. 0.793). Over 3 seeds the
  PhoBERT macro-F1 is 0.774 ± 0.012, still below the LLM, but the difference is
  not significant on 100 items (McNemar p = 1.00, bootstrap CI includes 0).
- **PhoBERT is about 55x faster and free to run**, so for a large volume of news it
  is the practical choice if its accuracy is acceptable.
- **PhoBERT is better on `macro_policy`** (18 of 20 correct; the LLM moved 7 of 20
  to `other`).
- **PhoBERT's main weakness is `other`**: recall 0.29 to 0.38 across seeds. In seed
  44, 15 of 21 `other` items get an event label, 7 of them `ma_investment` (news
  with words like "dự án", "mở rộng", "ký thỏa thuận": ids 13, 17, 25, 38, 69, 77,
  81). This drops `ma_investment` precision to about 0.40 in every seed.
- **This weakness matters more on real data**: about 54% of raw news is `other`,
  against 21% in the test set and 18% in the training set (101 of 560).
- Test sizes are small (7 `ma_investment`, 21 `other` items), so per-label scores
  move a lot with a few items.
- The model and training setup were not changed based on these test results.

## Stage 6: explanation quality on the blind test set

### Setup
- `python -m src.classify.explain_review` runs `run_pipeline` on all 100 test items
  (`data/processed/test_pipeline.csv`, local) and samples 20 items the pipeline did
  not label `other` (seed 42) into `data/processed/explain_review.csv` (local).
- Cost: $0.0319 for 100 items (180 LLM calls: 100 classifications, 80
  explanations), about $0.0004 per event item.
- In this run 20 items were labeled `other` and classification accuracy was 0.78
  (0.80 in the earlier LLM run): the LLM is not fully deterministic.
- The 20 explanations were graded on `correct` (faithful to the news, nothing
  invented) and `no_advice` (no price prediction, no buy/sell advice). An LLM
  assistant proposed the grades; the author checked them and made the final
  decisions (id 40 was changed from error to omission). A third
  column, `easy`, was filled but is not used.

### Results

| Criterion | Pass |
|---|---|
| correct | **18 of 20 (90%)** |
| no_advice | **20 of 20 (100%)** |

Errors:
- id 37: calls Vingroup's divestment gain "tiền thu được từ việc bán vốn"
  (proceeds); a gain is not the money received.
- id 54: says the company estimates the 50 billion dong dividend payout; in the
  sapo the estimate is the article's.

Minor issues (graded as pass, noted in the file):
- id 40 covers only a detail from the sapo and drops the title's main points
  (counted as an omission, not an error).
- 2 items have a wrong event label from the classifier (ids 13, 88; human label
  `other`); only the explanation was graded.
- 3 items still comment on missing information although the prompt forbids it
  (ids 26, 67, 88).
- 3 items have a generic or repeated last sentence (ids 5, 11, 34).
- Some terms are not explained: "Quyền Tổng Giám đốc", "niên độ", "thị giá".
- 20 items is a small sample, graded once.
