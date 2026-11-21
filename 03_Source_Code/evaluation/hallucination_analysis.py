# Calculate hallucination rates from human faithfulness labels.

from __future__ import annotations

import argparse
from pathlib import Path

import openpyxl
import pandas as pd


def main() -> None:
    # Aggregate completed binary human judgments into per-model rates.
    parser = argparse.ArgumentParser()
    parser.add_argument("review", help="Human review CSV")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    review_path = Path(args.review)
    if review_path.suffix.lower() == ".xlsx":
        review = pd.read_excel(review_path, sheet_name="Reviews")
    else:
        review = pd.read_csv(review_path, encoding="utf-8-sig")
    # Require both a qualitative label and a valid binary flag for a counted review.
    labels = review["faithfulness_label"].fillna("").astype(str).str.strip()
    binary = review["hallucinated_binary"].fillna("").astype(str).str.strip().str.lower()
    completed = labels.ne("") & binary.isin({"0", "1", "yes", "no", "true", "false"})
    review = review[completed].copy()
    if review.empty:
        raise ValueError(
            "No completed reviews found. Fill in faithfulness_label and "
            "hallucinated_binary (0/1) before calculating rates."
        )

    binary = review["hallucinated_binary"].fillna("").astype(str).str.lower()
    review["is_hallucinated"] = binary.isin({"1", "yes", "true"})

    rows = []
    # Divide each model's hallucination count by its own number of completed reviews.
    for model, group in review.groupby("model", sort=True):
        total = len(group)
        hallucinated = int(group["is_hallucinated"].sum())
        rows.append({
            "model": model,
            "total_reviewed": total,
            "hallucinated": hallucinated,
            "hallucination_rate": hallucinated / total,
        })

    output = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Hallucination Rates"
    sheet.append(list(output.columns))
    for row in output.itertuples(index=False, name=None):
        sheet.append(list(row))
    workbook.save(output_path.with_suffix(".xlsx"))
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
