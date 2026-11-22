# Create the human faithfulness review template.

from __future__ import annotations

import argparse
from pathlib import Path

import openpyxl
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE = ROOT / "01_Dataset" / "processed" / "llm_final_sample_ids.csv"
DEFAULT_TEST_SET = ROOT / "01_Dataset" / "processed" / "common_test_set.csv"
DEFAULT_OUTPUT = ROOT / "06_Evaluation_Results" / "human_faithfulness_review.csv"
PREDICTION_FILES = {
    "TextRank": ROOT / "05_Predictions" / "textrank_predictions.csv",
    "Seq2Seq": ROOT / "05_Predictions" / "seq2seq_predictions_penalty_1.5.csv",
    "mT5": ROOT / "05_Predictions" / "mt5_predictions.csv",
    "LLM": ROOT / "05_Predictions" / "llm_predictions.csv",
}


def main() -> None:
    # Build the review CSV and workbook with source, reference, and model outputs.
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["TextRank", "Seq2Seq", "mT5", "LLM"],
    )
    args = parser.parse_args()

    test = pd.read_csv(DEFAULT_TEST_SET, encoding="utf-8")
    sample = pd.read_csv(DEFAULT_SAMPLE, encoding="utf-8")
    review = test.merge(sample, on="article_id", how="inner", validate="one_to_one")

    rows = []
    for model in args.models:
        if model not in PREDICTION_FILES:
            raise ValueError(f"No prediction file configured for model: {model}")
        predictions = pd.read_csv(PREDICTION_FILES[model], encoding="utf-8-sig")
        predictions.columns = [column.lstrip("\ufeff") for column in predictions.columns]
        required = {"article_id", "prediction"}
        if not required.issubset(predictions.columns):
            raise ValueError(
                f"{PREDICTION_FILES[model]} must contain article_id and prediction."
            )
        predictions = predictions[["article_id", "prediction"]]
        if predictions["article_id"].duplicated().any():
            raise ValueError(f"{model} predictions contain duplicate article IDs.")
        # Match every sample article to exactly one prediction for this model.
        joined = review.merge(
            predictions,
            on="article_id",
            how="left",
            validate="one_to_one",
        )
        if joined["prediction"].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"{model} has missing predictions in the 100-article sample.")
        for _, row in joined.iterrows():
            rows.append({
                "article_id": row["article_id"],
                "model": model,
                "article": row["article"],
                "reference": row["reference"],
                "prediction": row["prediction"],
                "faithfulness_label": "",
                "hallucinated_binary": "",
                "reviewer": "",
                "notes": "",
            })

    output = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve human annotations when refreshing the workbook from current outputs.
    if output_path.exists():
        existing = pd.read_csv(output_path, encoding="utf-8-sig")
        key = ["article_id", "model"]
        review_fields = [
            "faithfulness_label",
            "hallucinated_binary",
            "reviewer",
            "notes",
        ]
        if set(key + review_fields).issubset(existing.columns):
            previous = existing[key + review_fields].drop_duplicates(key)
            output = output.drop(columns=review_fields).merge(
                previous,
                on=key,
                how="left",
                validate="one_to_one",
            )

    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    workbook_path = output_path.with_suffix(".xlsx")
    workbook = openpyxl.Workbook()
    # Separate reviewer instructions from the 400 annotation rows.
    guide = workbook.active
    guide.title = "Instructions"
    guide.append(["Field", "Instructions"])
    guide.append([
        "faithfulness_label",
        "Choose Faithful, Partially Faithful, or Unfaithful.",
    ])
    guide.append([
        "hallucinated_binary",
        "Use 1 if any factual claim is unsupported or contradicted by the article; otherwise 0.",
    ])
    guide.append(["reviewer", "Reviewer initials or name."])
    guide.append(["notes", "Briefly note the unsupported or contradicted claim, if any."])
    reviews = workbook.create_sheet("Reviews")
    reviews.append(list(output.columns))
    for row in output.itertuples(index=False, name=None):
        reviews.append(list(row))
    reviews.freeze_panes = "A2"
    reviews.auto_filter.ref = reviews.dimensions
    workbook.save(workbook_path)
    print(f"Created {len(output)} review rows at {output_path}")
    print(f"Created review workbook at {workbook_path}")


if __name__ == "__main__":
    main()
