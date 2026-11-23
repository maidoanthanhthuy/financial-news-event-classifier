"""Predict the event label of a news item with the fine-tuned PhoBERT model."""

import time
from functools import lru_cache

import numpy as np
import pandas as pd

from src.finetune.config import HF_REPO_ID, MAX_LENGTH
from src.finetune.data import build_text, underthesea_segment


def softmax(logits) -> np.ndarray:
    """Row-wise softmax that is stable for large logits."""
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def to_result(logits, id2label: dict) -> dict:
    """Top label and the probability of every label from one row of logits."""
    probs = softmax(logits)
    probs_by_label = {id2label[i]: float(p) for i, p in enumerate(probs)}
    return {"label": id2label[int(probs.argmax())], "probs": probs_by_label}


@lru_cache(maxsize=2)
def load_model(repo_id: str = HF_REPO_ID):
    """Download (once) and return the tokenizer and model from the Hub."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id).eval()
    return tok, model


def predict_phobert(
    title: str, sapo: str, repo_id: str = HF_REPO_ID, max_length: int = MAX_LENGTH
) -> dict:
    """Return the predicted label and per-label probabilities for one news item."""
    import torch

    tok, model = load_model(repo_id)
    text = underthesea_segment(build_text(title, sapo))
    enc = tok(text, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits[0].numpy()
    return to_result(logits, model.config.id2label)


def predict_frame(
    df: pd.DataFrame, repo_id: str = HF_REPO_ID, max_length: int = MAX_LENGTH
) -> pd.DataFrame:
    """Predict every row (id, title, sapo); one row per item with probs and seconds.

    The model is loaded before timing, so `seconds` is segmentation + inference.
    """
    load_model(repo_id)
    rows = []
    for row in df.itertuples(index=False):
        start = time.perf_counter()
        res = predict_phobert(row.title, row.sapo, repo_id, max_length)
        secs = time.perf_counter() - start
        probs = {f"prob_{k}": v for k, v in res["probs"].items()}
        rows.append({"id": row.id, "label": res["label"]} | probs | {"seconds": secs})
    return pd.DataFrame(rows)
