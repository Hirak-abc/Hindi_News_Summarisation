# Prepare deterministic, disjoint development and final LLM samples.

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TEST_SET = ROOT / "01_Dataset" / "processed" / "common_test_set.csv"
PROCESSED = TEST_SET.parent


def main() -> None:
    # Write reproducible, non-overlapping development and final sample IDs.
    test = pd.read_csv(TEST_SET, encoding="utf-8")
    if test["article_id"].duplicated().any():
        raise ValueError("common_test_set.csv contains duplicate article IDs.")

    # Sample development examples first, then draw final IDs only from the remainder.
    development = test.sample(n=50, random_state=42)
    remaining = test.drop(index=development.index)
    final = remaining.sample(n=100, random_state=42)

    development[["article_id"]].to_csv(
        PROCESSED / "llm_development_sample_ids.csv",
        index=False,
        encoding="utf-8",
    )
    final[["article_id"]].to_csv(
        PROCESSED / "llm_final_sample_ids.csv",
        index=False,
        encoding="utf-8",
    )
    print("Development sample: 50")
    print("Final sample: 100")
    print("Overlap:", len(set(development.article_id) & set(final.article_id)))


if __name__ == "__main__":
    main()
