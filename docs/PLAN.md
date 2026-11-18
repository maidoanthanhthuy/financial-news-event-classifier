# Project Plan: Financial News Event Classifier

## Stage 0. Scope
- 5 event types:
  1. Business results
  2. Dividends and share issuance
  3. Macro policy
  4. M&A, investment, new projects
  5. Legal and personnel
- VN30 basket fixed as of 03/08/2026 (`data/reference/vn30.csv`).

## Stage 1. Data
- Scrape CafeF and VnExpress Kinh doanh automatically, from 01/11/2024 to 31/10/2026.
- Store only: title, sapo (lead paragraph), datetime, link.
- Collect 2,000-3,000 raw articles; select 500 (about 100 per event type).
- LLM pre-labels, a human reviews.

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
- 20% held-out test set.
- Statistical significance testing.

## Stage 7. Demo
- Streamlit app deployed on Streamlit Community Cloud.
- Event study results loaded from CSV.
