# Calculate hallucination rates from human faithfulness labels.
# This comment identifies the module's purpose.

from __future__ import annotations
# Allow modern annotations without eager evaluation.

import argparse
# Import argparse to read the review and output paths.
from pathlib import Path
# Import Path to validate and create output locations.

import openpyxl
# Import openpyxl to create the hallucination workbook.
import pandas as pd
# Import pandas to read and aggregate review judgments.


def main() -> None:
    # Aggregate completed binary human judgments into per-model rates.
    parser = argparse.ArgumentParser()
    # Create a command-line argument parser.
    parser.add_argument("review", help="Human review CSV")
    # Require the human review CSV or XLSX file.
    parser.add_argument("--output", required=True)
    # Require the output file path.
    args = parser.parse_args()
    # Parse command-line arguments.

    review_path = Path(args.review)
    # Convert the review filename into a Path.
    if review_path.suffix.lower() == ".xlsx":
        review = pd.read_excel(review_path, sheet_name="Reviews")
    else:
        review = pd.read_csv(review_path, encoding="utf-8-sig")
        # Read the review CSV or Excel Reviews sheet.
    # Require both a qualitative label and a valid binary flag for a counted review.
    labels = review["faithfulness_label"].fillna("").astype(str).str.strip()
    # Normalize qualitative labels for completion checking.
    binary = review["hallucinated_binary"].fillna("").astype(str).str.strip().str.lower()
    # Normalize binary labels and accepted text variants.
    completed = labels.ne("") & binary.isin({"0", "1", "yes", "no", "true", "false"})
    # Count a row only when both qualitative and binary fields are completed.
    review = review[completed].copy()
    # Keep only completed review rows.
    if review.empty:
        raise ValueError(
            "No completed reviews found. Fill in faithfulness_label and "
            "hallucinated_binary (0/1) before calculating rates."
        )
        # Stop when no usable human annotations exist.

    binary = review["hallucinated_binary"].fillna("").astype(str).str.lower()
    # Normalize binary values again for conversion to booleans.
    review["is_hallucinated"] = binary.isin({"1", "yes", "true"})
    # Mark accepted positive values as hallucinated.

    rows = []
    # Initialize per-model result rows.
    # Divide each model's hallucination count by its own number of completed reviews.
    for model, group in review.groupby("model", sort=True):
        total = len(group)
        # Count completed reviews for this model.
        hallucinated = int(group["is_hallucinated"].sum())
        # Count completed rows marked hallucinated.
        rows.append({
            "model": model,
            "total_reviewed": total,
            "hallucinated": hallucinated,
            "hallucination_rate": hallucinated / total,
        })
        # Append the model's completed-review count and hallucination rate.

    output = pd.DataFrame(rows)
    # Convert aggregate rows into a DataFrame.
    output_path = Path(args.output)
    # Convert the output path into a Path.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Create the output directory.
    output.to_csv(output_path, index=False)
    # Write hallucination rates to CSV.
    workbook = openpyxl.Workbook()
    # Create a new Excel workbook.
    sheet = workbook.active
    # Select the active worksheet.
    sheet.title = "Hallucination Rates"
    # Name the worksheet.
    sheet.append(list(output.columns))
    # Add the output table headers.
    for row in output.itertuples(index=False, name=None):
        sheet.append(list(row))
        # Append one model result per worksheet row.
    workbook.save(output_path.with_suffix(".xlsx"))
    # Save the workbook beside the CSV.
    print(output.to_string(index=False))
    # Print aggregate rates to the terminal.


if __name__ == "__main__":
    main()
    # Run analysis only when the file is executed directly.
