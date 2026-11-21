"""Run the LLM classifier on the blind test set.

Usage: python -m src.classify.llm_test   # write data/processed/test_llm_v2.csv
"""

import pandas as pd

from src.classify.config import MODEL, TEST_BLIND_CSV, TEST_LLM_V2_CSV
from src.classify.llm_label import classify_news, cost_usd, get_client, load_guide


def main() -> None:
    """Classify every test item and save id, label, reason and token counts."""
    test = pd.read_csv(TEST_BLIND_CSV)
    client, guide = get_client(), load_guide()
    rows = []
    for i, row in enumerate(test.itertuples(index=False), start=1):
        rows.append({"id": row.id} | classify_news(row.title, row.sapo, client, guide))
        print(f"{i}/{len(test)}", end="\r")
    out = pd.DataFrame(rows).assign(model=MODEL)
    out.to_csv(TEST_LLM_V2_CSV, index=False)
    cost = cost_usd(out["input_tokens"].sum(), out["output_tokens"].sum())
    print(f"wrote {len(out)} rows to {TEST_LLM_V2_CSV}; cost ${cost:.4f}")
    print(f"unparsed answers: {out['label'].isna().sum()}")


if __name__ == "__main__":
    main()
