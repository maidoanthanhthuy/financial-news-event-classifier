# Data Process: Labeling and Ticker Tagging

How the labeled datasets in `data/labeled/` were built. Label definitions and edge
cases are in [labeling_guide.md](labeling_guide.md).

## 1. Raw news
- Sources: CafeF and VnExpress Kinh doanh, 2024-11-01 to 2026-10-31
  (`src/scraping/`). Pages are sampled (CafeF: 1 page in every 10; VnExpress: page 1
  of 12 equal date windows) to spread items over the period.
- Fields kept: source, section, title, sapo, published_at, url.
- Result: 4,125 items (CafeF 3,090, VnExpress 1,035), no duplicate urls.
  Stored locally in `data/raw/news.csv` (not committed).

## 2. LLM pre-labeling
- Every raw item gets one of 6 labels (5 event types + `other`) from an OpenAI model
  with structured output (`src/classify/llm_label.py`). The labeling guide is the
  system prompt.
- About 54% of items were labeled `other`.
- Output stays local: `data/processed/news_prelabeled.csv`.

## 3. Training set: human review of LLM labels
The model uses 6 labels: 5 event types plus `other`, which is a real class, not a
filter.
- `python -m src.classify.review sample`: 100 random items per event label (seed 42,
  `other` skipped) -> `data/labeled/review.csv` (500 items, ids 1-500).
- `python -m src.classify.review sample-other`: 60 random items the LLM labeled
  `other` (seed 42), excluding the review and test items ->
  `data/labeled/review_other.csv` (ids 501-560).
- LLM pre-labels were reviewed item by item by the author, with an LLM assistant
  proposing corrections; the author made every final decision. `human_label` is
  filled only when the LLM label is wrong; **an empty `human_label` means the author
  agreed with the LLM label**.
- Corrections: 67 of 560 items (12.0%). By LLM label: ma_investment 29%,
  other 15%, macro_policy 10%, business_results 8%, legal_personnel 8%,
  dividends_issuance 3%. 50 event-labeled items were moved to `other`, and 9 `other`
  items were moved to an event label.
- Rules found during review were added to the labeling guide.
- `python -m src.classify.review merge` reads both review files ->
  `data/labeled/news_labeled.csv`: `final_label` = `human_label` if filled, else
  `llm_label`; all 6 labels are kept. The url is joined back from the pre-labeled
  data on (source, published_at, title).
- Result: 560 items: other 101, dividends_issuance 99, macro_policy 99,
  legal_personnel 95, business_results 94, ma_investment 72.
- Note: items were sampled by LLM label and unchanged items keep the LLM label, so
  this set may carry some LLM labeling habits.

## 4. Test set: blind human labels
- `python -m src.classify.review test`: 20 random items per LLM label (seed 42,
  `other` skipped), excluding every item in `review.csv`, shuffled so ids give no
  hint of the label -> `data/labeled/test_blind.csv` (100 items).
- The file has only id, title, sapo, `human_label`, `tickers`. LLM labels are saved
  separately in `data/processed/test_llm_labels.csv` (local) and were not seen
  while labeling.
- Every item was labeled by hand, using all 6 labels including `other`.
- Result: other 21, macro_policy 20, dividends_issuance 19, business_results 18,
  legal_personnel 15, ma_investment 7.
- Labels are frozen as of commit e0ae966.
- Reference point: the draft LLM labels (made with an earlier guide) scored
  accuracy 92.4% and macro-F1 0.910 on the 79 items not labeled `other` (5 labels).

## 5. Ticker tagging
- Reference list: the 30 VN30 tickers fixed as of 2026-08-03
  (`data/reference/vn30.csv`).
- Rules for the `tickers` column only (they do not affect event labels):
  - Tag a VN30 ticker only when the company name or ticker appears in the title or
    sapo. Mentioning a person (e.g. a company's chair) does not count.
  - Subsidiaries or related companies not in the list are not mapped to their
    parent (e.g. VinFast, V-Green or GSM are not tagged `VIC`; Highlands is not
    tagged).
  - Several tickers are separated by `;` (e.g. `VIC;STB;VJC`). Empty when none
    applies.
- Name lookup:

| Group | Tickers (company names) |
|---|---|
| Banks | ACB (Á Châu), BID (BIDV), CTG (VietinBank), HDB (HDBank), LPB (LPBank), MBB (MB), SHB, SSB (SeABank), STB (Sacombank), TCB (Techcombank), VCB (Vietcombank), VIB, VPB (VPBank) |
| Securities | SSI, TCX (TCBS, Chứng khoán Kỹ Thương) |
| Consumer | MCH (Masan Consumer), MSN (Masan Group), MWG (Thế Giới Di Động), SAB (Sabeco), VNM (Vinamilk) |
| Vingroup | VIC (Vingroup), VHM (Vinhomes), VRE (Vincom Retail), VPL (Vinpearl) |
| Other | FPT, HPG (Hòa Phát), GAS (PV Gas), BSR (Lọc hóa dầu Bình Sơn), GVR (Cao su Việt Nam), VJC (Vietjet) |

- So far only the test set is tagged: 14 of 100 items have tickers, 12 with exactly
  one. The event study keeps only items tied to exactly one VN30 ticker.
