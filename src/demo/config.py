"""Settings for the Streamlit demo (app/app.py)."""

from src.classify.config import LABELED_CSV
from src.event_study.config import EVENT_SUMMARY_CSV

SAMPLES_CSV = LABELED_CSV  # train set only; never the blind test set
SAMPLE_IDS = [9, 108, 323, 366]  # one per label: results, dividends, M&A, other
SUMMARY_CSV = EVENT_SUMMARY_CSV
SUMMARY_K = 3
SUMMARY_SAMPLE = "all"
ALPHA = 0.05
NO_HISTORY_LABELS = ["other", "macro_policy"]
MAX_RUNS_PER_SESSION = 10

LABEL_VI = {
    "business_results": "Kết quả kinh doanh",
    "dividends_issuance": "Cổ tức, phát hành cổ phiếu",
    "macro_policy": "Chính sách vĩ mô",
    "ma_investment": "M&A, đầu tư",
    "legal_personnel": "Pháp lý, nhân sự",
    "other": "Tin khác",
}
