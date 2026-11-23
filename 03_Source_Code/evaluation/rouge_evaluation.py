# Shared Unicode-aware ROUGE evaluator for prediction files.
# This comment identifies the module's purpose.

from __future__ import annotations
# Allow modern annotations without eager evaluation.

import argparse
# Import argparse to read prediction and output paths.
import unicodedata
# Import unicodedata to normalize and categorize Hindi text.
from pathlib import Path
# Import Path to create output directories and paths.

import pandas as pd
# Import pandas to load prediction CSV files.
from rouge_score import rouge_scorer
# Import the ROUGE implementation used for n-gram scoring.


class HindiTokenizer:
    # Split text into Unicode-aware tokens for Hindi ROUGE scoring.
    # Define a custom tokenizer for the ROUGE library.

    def tokenize(self, text: str) -> list[str]:
        # Normalize text and keep letters, combining marks, and digits together.
        text = unicodedata.normalize("NFC", str(text).lower())
        # Normalize Unicode and lowercase the input.
        tokens = []
        # Initialize the completed-token list.
        current = []
        # Initialize the current-token character buffer.
        for character in text:
            category = unicodedata.category(character)
            # Obtain the Unicode category of each character.
            if category[0] in {"L", "M", "N"} or (
                character in "\u200c\u200d" and current
            ):
                current.append(character)
                # Append letters, marks, digits, and active joiners.
            elif current:
                tokens.append("".join(current))
                current = []
                # Save a completed token and reset the buffer.
        if current:
            tokens.append("".join(current))
            # Save the final buffered token.
        return tokens
        # Return the Unicode-aware token list.


def evaluate(predictions: pd.DataFrame) -> pd.DataFrame:
    # Return mean ROUGE precision, recall, and F1 for non-empty predictions.
    required = {"reference", "prediction"}
    # Define required prediction columns.
    missing = required - set(predictions.columns)
    # Find required columns absent from the DataFrame.
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
        # Reject input with missing required columns.

    # Exclude missing/blank outputs rather than silently counting them as summaries.
    predictions = predictions[
        predictions["prediction"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    # Remove missing or blank predictions before scoring.
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=HindiTokenizer(),
    )
    # Configure Hindi-aware ROUGE-1, ROUGE-2, and ROUGE-L scoring.
    totals = {
        "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
    }
    # Initialize precision, recall, and F-measure accumulators.

    for reference, prediction in zip(
        predictions["reference"].fillna(""),
        predictions["prediction"].fillna(""),
    ):
        result = scorer.score(str(reference), str(prediction))
        # Score this reference/prediction pair.
        for metric in totals:
            totals[metric]["precision"] += result[metric].precision
            totals[metric]["recall"] += result[metric].recall
            totals[metric]["fmeasure"] += result[metric].fmeasure
            # Add the pair's three values to the metric accumulators.

    row_count = len(predictions)
    # Count valid prediction/reference rows.
    if row_count == 0:
        raise ValueError("No non-empty predictions to evaluate.")
        # Reject a completely blank prediction set.
    return pd.DataFrame([
        {
            "metric": metric,
            "precision": values["precision"] / row_count,
            "recall": values["recall"] / row_count,
            "fmeasure": values["fmeasure"] / row_count,
        }
        for metric, values in totals.items()
    ])
    # Return one mean-score row per ROUGE metric.


def main() -> None:
    # Evaluate one prediction CSV and write a machine-readable score table.
    parser = argparse.ArgumentParser()
    # Create a command-line argument parser.
    parser.add_argument("predictions", help="Prediction CSV file")
    # Require the prediction CSV path.
    parser.add_argument("--output", required=True, help="Output CSV file")
    # Require the output score CSV path.
    args = parser.parse_args()
    # Parse command-line arguments.

    predictions = pd.read_csv(args.predictions, encoding="utf-8-sig")
    # Load prediction data with UTF-8 BOM support.
    scores = evaluate(predictions)
    # Calculate aggregate ROUGE values.
    output = Path(args.output)
    # Convert the output location to a Path.
    output.parent.mkdir(parents=True, exist_ok=True)
    # Create the output directory.
    scores.to_csv(output, index=False)
    # Save aggregate scores without a pandas index.
    print(scores.to_string(index=False))
    # Print the score table to the terminal.


if __name__ == "__main__":
    main()
    # Run evaluation only when executed directly.
