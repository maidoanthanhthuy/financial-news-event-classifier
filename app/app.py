"""Streamlit demo: classify a Vietnamese financial news item and explain it.

Run from the repository root: streamlit run app/app.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classify.pipeline import run_pipeline  # noqa: E402
from src.demo.config import (  # noqa: E402
    MAX_RUNS_PER_SESSION,
    SAMPLE_IDS,
    SAMPLES_CSV,
    SUMMARY_CSV,
)
from src.demo.helpers import label_name, past_reaction, pick_samples  # noqa: E402

DISCLAIMER = "Giải thích do AI tạo, có thể sai sót. Không phải lời khuyên đầu tư."
OWN_INPUT = "Tự nhập tin"
REPO_URL = "https://github.com/maidoanthanhthuy/financial-news-event-classifier"
FOOTER = f"Dữ liệu: CafeF, VnExpress (11/2024–10/2026). Mã nguồn: [GitHub]({REPO_URL})"


def set_api_key() -> None:
    """Put OPENAI_API_KEY in the environment from st.secrets, else from .env."""
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:  # no secrets.toml
        key = None
    if key:
        os.environ["OPENAI_API_KEY"] = key
    else:
        load_dotenv()


@st.cache_data
def load_data() -> tuple:
    """Sample news and the event study summary."""
    samples = pick_samples(pd.read_csv(SAMPLES_CSV), SAMPLE_IDS)
    return samples, pd.read_csv(SUMMARY_CSV)


st.set_page_config(page_title="Phân loại tin tài chính VN30", page_icon="📰")
set_api_key()
samples, summary = load_data()
st.session_state.setdefault("runs", 0)

st.title("Phân loại tin tài chính VN30")
st.write(
    "App đọc một tin tài chính tiếng Việt, cho biết đây là loại tin gì, tìm mã cổ "
    "phiếu VN30 liên quan và giải thích tin bằng lời dễ hiểu cho người mới."
)
st.write("App **không** dự đoán giá cổ phiếu và **không** đưa ra lời khuyên đầu tư.")

options = [f"Tin mẫu {i + 1}: {t}" for i, t in enumerate(samples["title"])]
choice = st.radio("Chọn tin", options + [OWN_INPUT])
if choice == OWN_INPUT:
    title = st.text_input("Tiêu đề")
    sapo = st.text_area("Sapo (đoạn mở đầu)")
    url = None
else:
    row = samples.iloc[options.index(choice)]
    title, sapo, url = row["title"], row["sapo"], row["url"]
    st.caption(sapo)

runs_left = MAX_RUNS_PER_SESSION - st.session_state["runs"]

if st.button("Phân tích", type="primary", disabled=runs_left <= 0):
    if not title.strip():
        st.warning("Hãy nhập tiêu đề.")
    else:
        st.session_state["runs"] += 1
        try:
            with st.spinner("Đang phân tích..."):
                out = run_pipeline(title, sapo)
        except Exception as exc:  # show any API error instead of a stack trace
            st.error(f"Không phân tích được: {exc}")
        else:
            st.subheader(f"Loại tin: {label_name(out['label'])}")
            tickers = ", ".join(out["main_tickers"]) or "Không tìm thấy mã VN30"
            st.markdown(f"**Mã chính:** {tickers}")
            st.markdown(f"**Giải thích:** {out['explanation']}")
            history = past_reaction(summary, out["label"])
            if history:
                st.info(history)
            st.caption(DISCLAIMER)
            if url:
                st.markdown(f"[Đọc bài gốc]({url})")
runs_left = MAX_RUNS_PER_SESSION - st.session_state["runs"]
if runs_left <= 0:
    st.warning("Bạn đã dùng hết lượt phân tích của phiên này.")
else:
    st.caption(f"Còn {runs_left}/{MAX_RUNS_PER_SESSION} lượt phân tích trong phiên.")

st.divider()
st.caption(FOOTER)
