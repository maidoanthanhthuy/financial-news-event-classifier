"""Fine-tune PhoBERT on the 6-label training set (run on a GPU, e.g. Colab).

Usage: python -m src.finetune.train --seed 42 [--output-dir models/phobert_seed42]
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.classify.evaluate import summary_scores
from src.finetune.config import (
    BATCH_SIZE,
    ID2LABEL,
    LABEL2ID,
    LEARNING_RATE,
    MAX_EPOCHS,
    MAX_LENGTH,
    MODEL_NAME,
    OUTPUT_DIR,
    PATIENCE,
    WEIGHT_DECAY,
)
from src.finetune.data import load_splits


def compute_metrics(eval_pred) -> dict:
    """Accuracy and macro-F1 from (logits, labels)."""
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(axis=-1)
    return summary_scores(
        pd.Series(labels), pd.Series(preds), labels=list(range(len(ID2LABEL)))
    )


def best_epoch(log_history: list, metric: str = "eval_macro_f1") -> dict:
    """The evaluation log entry with the highest metric (first one on ties)."""
    evals = [e for e in log_history if metric in e]
    return max(evals, key=lambda e: e[metric])


def summarize_seeds(results: list, metric: str = "macro_f1") -> dict:
    """Mean and sample std of a metric over seeds, plus the best seed."""
    s = pd.Series([r[metric] for r in results])
    best = max(results, key=lambda r: r[metric])
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "best_seed": best["seed"],
    }


class TextDataset:
    """Tokenized texts and labels in the format Trainer expects."""

    def __init__(self, texts: list, labels: list, tokenizer, max_length: int):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def train(seed: int, output_dir: str, max_length: int = MAX_LENGTH) -> dict:
    """Train with early stopping on validation macro-F1; save the best model.

    The train/validation split is fixed; `seed` changes initialization and order.
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(seed)
    train_df, val_df = load_splits()  # fixed split (config SEED); seed = training
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(ID2LABEL), id2label=ID2LABEL, label2id=LABEL2ID
    )
    ckpt_dir = Path(output_dir) / "checkpoints"
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=MAX_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
        logging_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=TextDataset(train_df["text"], train_df["label"], tok, max_length),
        eval_dataset=TextDataset(val_df["text"], val_df["label"], tok, max_length),
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )
    trainer.train()
    trainer.save_model(output_dir)
    shutil.rmtree(ckpt_dir, ignore_errors=True)

    best = best_epoch(trainer.state.log_history)
    result = {
        "seed": seed,
        "macro_f1": best["eval_macro_f1"],
        "accuracy": best["eval_accuracy"],
        "best_epoch": best["epoch"],
        "epochs_run": trainer.state.epoch,
        "train_items": len(train_df),
        "val_items": len(val_df),
    }
    Path(output_dir, "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def push_to_hub(model_dir: str, repo_id: str, token: str) -> None:
    """Upload a saved model and tokenizer to the Hugging Face Hub."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    AutoModelForSequenceClassification.from_pretrained(model_dir).push_to_hub(
        repo_id, token=token
    )
    AutoTokenizer.from_pretrained(model_dir).push_to_hub(repo_id, token=token)


def main() -> None:
    """Parse arguments and train one seed."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    out = args.output_dir or f"{OUTPUT_DIR}_seed{args.seed}"
    print(json.dumps(train(args.seed, out, args.max_length), indent=2))


if __name__ == "__main__":
    main()
