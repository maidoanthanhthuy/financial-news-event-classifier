"""Settings for PhoBERT fine-tuning."""

from src.classify.config import LABELED_CSV, LABELS

MODEL_NAME = "vinai/phobert-base-v2"
LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}

DATA_CSV = LABELED_CSV
LABEL_COL = "final_label"
VAL_SIZE = 0.15  # share of each label held out for validation
SEED = 42

MAX_LENGTH = 128  # set from the token length check (python -m src.finetune.data)
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_EPOCHS = 10
PATIENCE = 2  # epochs without macro-F1 gain before early stopping
WEIGHT_DECAY = 0.01

OUTPUT_DIR = "models/phobert"
HF_REPO_ID = "maidoanthanhthuy/phobert-vn30-news-events"  # change to your HF user
