# Project Plan: Financial News Event Classifier

## Stage 0. Scope
- The model classifies news into 6 labels: 5 event types plus `other`.
  1. Business results
  2. Dividends and share issuance
  3. Macro policy
  4. M&A, investment, new projects
  5. Legal and personnel
  6. Other: a real class the model learns and predicts, not just a filter
     (about half of all news has no event of types 1-5).
- VN30 basket fixed as of 03/08/2026 (`data/reference/vn30.csv`).

## Stage 1. Data
- Scrape CafeF and VnExpress Kinh doanh automatically, from 01/11/2024 to 31/10/2026.
- Store only: source, section, title, sapo (lead paragraph), datetime, link.
- Collected 4,125 raw articles; reviewed 500 LLM-labeled items (100 per event type)
  plus 60 items the LLM labeled `other`. Items judged `other` stay in the training
  set with the `other` label.
- LLM pre-labels, a human reviews (process: `docs/data_process.md`).

## Stage 2. LLM baseline
- Classification with OpenAI using structured output (Pydantic).

## Stage 3. LangGraph 3-step pipeline
1. Classify the event type.
2. Extract related VN30 tickers.
3. Explain for beginners (no price prediction, no investment advice).

## Stage 4. PhoBERT fine-tuning
- Fine-tune `vinai/phobert-base-v2` on Colab.
- Word segmentation with VnCoreNLP.

## Stage 5. Event study
- Abnormal returns vs. VN-Index after 1, 3 and 5 trading sessions.
- News published after market close counts from the next session.
- Reuse the PostgreSQL database and price data from Project 2.

## Stage 6. Evaluation
- PhoBERT vs. LLM: accuracy, speed, cost.
- Blind test set: 100 hand-labeled items not used in training (see `docs/data_process.md`).
- Statistical significance testing.

## Stage 7. Demo
- Streamlit app deployed on Streamlit Community Cloud.
- Event study results loaded from CSV.
