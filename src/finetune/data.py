"""Prepare news text for PhoBERT: join title + sapo, word-segment, split.

Usage: python -m src.finetune.data   # token length check to choose MAX_LENGTH
"""

from collections.abc import Callable

import pandas as pd

from src.finetune.config import (
    DATA_CSV,
    LABEL2ID,
    LABEL_COL,
    MODEL_NAME,
    SEED,
    VAL_SIZE,
)


def underthesea_segment(text: str) -> str:
    """Word-segment Vietnamese text with underthesea (compounds joined by '_')."""
    from underthesea import word_tokenize

    return word_tokenize(text, format="text")


def build_text(title, sapo) -> str:
    """Join title and sapo into one string; missing parts become empty."""
    parts = [p.strip() for p in (title, sapo) if isinstance(p, str) and p.strip()]
    return ". ".join(parts)


def prepare_dataset(
    df: pd.DataFrame,
    label_col: str = LABEL_COL,
    label2id: dict = LABEL2ID,
    segment_fn: Callable = underthesea_segment,
) -> pd.DataFrame:
    """Return id, segmented text and integer label for each row."""
    pairs = zip(df["title"], df["sapo"], strict=True)
    texts = [segment_fn(build_text(t, s)) for t, s in pairs]
    return pd.DataFrame(
        {
            "id": df["id"].to_numpy(),
            "text": texts,
            "label": df[label_col].map(label2id).to_numpy(),
        }
    )


def stratified_split(
    df: pd.DataFrame,
    val_size: float = VAL_SIZE,
    seed: int = SEED,
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/validation, taking val_size of each label for validation."""
    val = df.groupby(label_col).sample(frac=val_size, random_state=seed)
    train = df.drop(val.index)
    return train.reset_index(drop=True), val.reset_index(drop=True)


def length_stats(lengths: list, max_length: int) -> dict:
    """Percentiles of token lengths and the share longer than max_length."""
    s = pd.Series(lengths)
    return {
        "mean": float(s.mean()),
        "p50": float(s.quantile(0.5)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": int(s.max()),
        f"share_over_{max_length}": float((s > max_length).mean()),
    }


def load_splits(
    path: str = DATA_CSV, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the labeled CSV, prepare text and split train/validation."""
    return stratified_split(prepare_dataset(pd.read_csv(path)), seed=seed)


def main() -> None:
    """Print token length stats of the segmented texts with the PhoBERT tokenizer."""
    from transformers import AutoTokenizer

    from src.finetune.config import MAX_LENGTH

    data = prepare_dataset(pd.read_csv(DATA_CSV))
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    lengths = [len(tok(t)["input_ids"]) for t in data["text"]]
    print(f"items: {len(data)}")
    print(length_stats(lengths, MAX_LENGTH))
    train, val = stratified_split(data)
    print(f"train {len(train)}, validation {len(val)}")
    print(val["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
