# Create the human faithfulness review template.
# This comment identifies the purpose of this module.

from __future__ import annotations
# Allow forward references and modern annotations without eager evaluation.

import argparse
# Import argparse to parse the output path and model selections.
from pathlib import Path
# Import Path to build paths and create output directories.

import openpyxl
# Import openpyxl to create the human-review Excel workbook.
import pandas as pd
# Import pandas to merge and format review data.


ROOT = Path(__file__).resolve().parents[2]
# Resolve the repository root from this source file.
DEFAULT_SAMPLE = ROOT / "01_Dataset" / "processed" / "llm_final_sample_ids.csv"
# Set the default deterministic 100-article sample path.
DEFAULT_TEST_SET = ROOT / "01_Dataset" / "processed" / "common_test_set.csv"
# Set the default source test dataset path.
DEFAULT_OUTPUT = ROOT / "06_Evaluation_Results" / "human_faithfulness_review.csv"
# Set the default review CSV and workbook base path.
PREDICTION_FILES = {
    # Map each model display name to its prediction CSV.
    "TextRank": ROOT / "05_Predictions" / "textrank_predictions.csv",
    "Seq2Seq": ROOT / "05_Predictions" / "seq2seq_predictions_penalty_1.5.csv",
    "mT5": ROOT / "05_Predictions" / "mt5_predictions.csv",
    "LLM": ROOT / "05_Predictions" / "llm_predictions.csv",
}


def main() -> None:
    # Build the review CSV and workbook with source, reference, and model outputs.
    parser = argparse.ArgumentParser()
    # Create a command-line argument parser.
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    # Configure the review CSV/XLSX output path.
    parser.add_argument(
        "--models",
        nargs="+",
        default=["TextRank", "Seq2Seq", "mT5", "LLM"],
    )
    # Configure the models included in the review.
    args = parser.parse_args()
    # Parse command-line arguments.

    test = pd.read_csv(DEFAULT_TEST_SET, encoding="utf-8")
    # Load the full common test set.
    sample = pd.read_csv(DEFAULT_SAMPLE, encoding="utf-8")
    # Load the deterministic review sample IDs.
    review = test.merge(sample, on="article_id", how="inner", validate="one_to_one")
    # Join test articles to sample IDs and require exactly one match per row.

    rows = []
    # Initialize the review row list.
    for model in args.models:
        if model not in PREDICTION_FILES:
            raise ValueError(f"No prediction file configured for model: {model}")
            # Reject model names that have no configured prediction file.
        predictions = pd.read_csv(PREDICTION_FILES[model], encoding="utf-8-sig")
        # Load the selected model's predictions.
        predictions.columns = [column.lstrip("\ufeff") for column in predictions.columns]
        # Remove Unicode BOM characters from column names.
        required = {"article_id", "prediction"}
        # Define the prediction columns required for the review.
        if not required.issubset(predictions.columns):
            raise ValueError(
                f"{PREDICTION_FILES[model]} must contain article_id and prediction."
            )
            # Reject prediction files missing required columns.
        predictions = predictions[["article_id", "prediction"]]
        # Keep only columns needed for joining and annotation.
        if predictions["article_id"].duplicated().any():
            raise ValueError(f"{model} predictions contain duplicate article IDs.")
            # Reject ambiguous predictions with duplicate IDs.
        # Match every sample article to exactly one prediction for this model.
        joined = review.merge(
            predictions,
            on="article_id",
            how="left",
            validate="one_to_one",
        )
        # Join predictions onto the 100 review articles.
        if joined["prediction"].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"{model} has missing predictions in the 100-article sample.")
            # Reject incomplete model coverage for the review sample.
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
            # Append one human-review record for this article/model pair.

    output = pd.DataFrame(rows)
    # Convert review records into a DataFrame.
    output_path = Path(args.output)
    # Convert the output destination to a Path.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Create the output directory.

    # Preserve human annotations when refreshing the workbook from current outputs.
    if output_path.exists():
        existing = pd.read_csv(output_path, encoding="utf-8-sig")
        # Load previously saved human-review annotations.
        key = ["article_id", "model"]
        # Define the unique article/model merge key.
        review_fields = [
            "faithfulness_label",
            "hallucinated_binary",
            "reviewer",
            "notes",
        ]
        # Define the human-annotation fields that should be retained.
        if set(key + review_fields).issubset(existing.columns):
            previous = existing[key + review_fields].drop_duplicates(key)
            # Keep one prior annotation record for every article/model pair.
            output = output.drop(columns=review_fields).merge(
                previous,
                on=key,
                how="left",
                validate="one_to_one",
            )
            # Merge prior annotations into refreshed model/article rows.

    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    # Save the review table to CSV with UTF-8 BOM.
    workbook_path = output_path.with_suffix(".xlsx")
    # Build the workbook path beside the CSV.
    workbook = openpyxl.Workbook()
    # Create a new Excel workbook.
    # Separate reviewer instructions from the 400 annotation rows.
    guide = workbook.active
    # Use the first worksheet for instructions.
    guide.title = "Instructions"
    # Name the instruction worksheet.
    guide.append(["Field", "Instructions"])
    # Add the instruction table header.
    guide.append([
        "faithfulness_label",
        "Choose Faithful, Partially Faithful, or Unfaithful.",
    ])
    # Explain the qualitative faithfulness label field.
    guide.append([
        "hallucinated_binary",
        "Use 1 if any factual claim is unsupported or contradicted by the article; otherwise 0.",
    ])
    # Explain the binary hallucination field.
    guide.append(["reviewer", "Reviewer initials or name."])
    # Explain the reviewer field.
    guide.append(["notes", "Briefly note the unsupported or contradicted claim, if any."])
    # Explain the optional reviewer notes field.
    reviews = workbook.create_sheet("Reviews")
    # Create the worksheet containing annotation rows.
    reviews.append(list(output.columns))
    # Write the review table column headers.
    for row in output.itertuples(index=False, name=None):
        reviews.append(list(row))
        # Append each human-review record as an Excel row.
    reviews.freeze_panes = "A2"
    # Freeze the header row while scrolling.
    reviews.auto_filter.ref = reviews.dimensions
    # Enable worksheet filtering across all used rows.
    workbook.save(workbook_path)
    # Save the review workbook.
    print(f"Created {len(output)} review rows at {output_path}")
    # Report the CSV review size and location.
    print(f"Created review workbook at {workbook_path}")
    # Report the workbook location.


if __name__ == "__main__":
    main()
    # Run the review builder only when executed directly.
