# Shared Unicode-aware ROUGE evaluator for prediction files.

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer


class HindiTokenizer:
    # Split text into Unicode-aware tokens for Hindi ROUGE scoring.

    def tokenize(self, text: str) -> list[str]:
        # Normalize text and keep letters, combining marks, and digits together.
        text = unicodedata.normalize("NFC", str(text).lower())
        tokens = []
        current = []
        for character in text:
            category = unicodedata.category(character)
            if category[0] in {"L", "M", "N"} or (
                character in "\u200c\u200d" and current
            ):
                current.append(character)
            elif current:
                tokens.append("".join(current))
                current = []
        if current:
            tokens.append("".join(current))
        return tokens


def evaluate(predictions: pd.DataFrame) -> pd.DataFrame:
    # Return mean ROUGE precision, recall, and F1 for non-empty predictions.
    required = {"reference", "prediction"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    # Exclude missing/blank outputs rather than silently counting them as summaries.
    predictions = predictions[
        predictions["prediction"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=HindiTokenizer(),
    )
    totals = {
        "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
    }

    for reference, prediction in zip(
        predictions["reference"].fillna(""),
        predictions["prediction"].fillna(""),
    ):
        result = scorer.score(str(reference), str(prediction))
        for metric in totals:
            totals[metric]["precision"] += result[metric].precision
            totals[metric]["recall"] += result[metric].recall
            totals[metric]["fmeasure"] += result[metric].fmeasure

    row_count = len(predictions)
    if row_count == 0:
        raise ValueError("No non-empty predictions to evaluate.")
    return pd.DataFrame([
        {
            "metric": metric,
            "precision": values["precision"] / row_count,
            "recall": values["recall"] / row_count,
            "fmeasure": values["fmeasure"] / row_count,
        }
        for metric, values in totals.items()
    ])


def main() -> None:
    # Evaluate one prediction CSV and write a machine-readable score table.
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", help="Prediction CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions, encoding="utf-8-sig")
    scores = evaluate(predictions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output, index=False)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
