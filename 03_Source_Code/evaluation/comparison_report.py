# Build same-sample four-model and full-test ROUGE tables and chart.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import openpyxl
import pandas as pd

from rouge_evaluation import evaluate

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "06_Evaluation_Results"
CHART_DIR = ROOT / "08_Report" / "charts"
SAMPLE_PATH = ROOT / "01_Dataset" / "processed" / "llm_final_sample_ids.csv"
PREDICTION_FILES = {
    "TextRank": ROOT / "05_Predictions" / "textrank_predictions.csv",
    "Seq2Seq": ROOT / "05_Predictions" / "seq2seq_predictions_penalty_1.5.csv",
    "mT5": ROOT / "05_Predictions" / "mt5_predictions.csv",
    "LLM": ROOT / "05_Predictions" / "llm_predictions.csv",
}


def read_scores(path: Path, model: str) -> dict[str, float]:
    # Read either F1-only or precision/recall/F1 score output formats.
    frame = pd.read_csv(path)
    if "f1" in frame.columns:
        values = dict(zip(frame["metric"], frame["f1"]))
    elif "fmeasure" in frame.columns:
        values = dict(zip(frame["metric"], frame["fmeasure"]))
    else:
        raise ValueError(f"Unknown score format: {path}")
    return {
        "model": model,
        "rouge1": float(values["rouge1"]),
        "rouge2": float(values["rouge2"]),
        "rougeL": float(values["rougeL"]),
    }


def main() -> None:
    # Score all four systems on shared IDs and write comparison artifacts.
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    rows = []
    for model, path in PREDICTION_FILES.items():
        predictions = pd.read_csv(path, encoding="utf-8-sig")
        predictions.columns = [column.lstrip("\ufeff") for column in predictions.columns]
        # Join on stable article IDs so each model is scored on exactly the same examples.
        sample_predictions = sample.merge(
            predictions,
            on="article_id",
            how="left",
            validate="one_to_one",
        )
        if sample_predictions["prediction"].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"{model} predictions are incomplete for the common sample.")
        scores = evaluate(sample_predictions)
        values = dict(zip(scores["metric"], scores["fmeasure"]))
        rows.append({
            "model": model,
            "rouge1": float(values["rouge1"]),
            "rouge2": float(values["rouge2"]),
            "rougeL": float(values["rougeL"]),
        })

    comparison = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "rouge_comparison.csv", index=False)

    # Full-test results are separate because the LLM only has sample predictions.
    full_test = pd.DataFrame([
        read_scores(
            ROOT / "06_Evaluation_Results" / "textrank_rouge_scores.csv",
            "TextRank",
        ),
        read_scores(
            ROOT / "06_Evaluation_Results" / "seq2seq_penalty_1.5_rouge_unicode.csv",
            "Seq2Seq",
        ),
        read_scores(
            ROOT / "06_Evaluation_Results" / "mt5_rouge_scores.csv",
            "mT5",
        ),
    ])
    full_test.to_csv(OUTPUT_DIR / "rouge_comparison_full_test.csv", index=False)

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Common Sample (100)"
    sheet.append(["model", "rouge1", "rouge2", "rougeL"])
    for row in comparison.to_dict("records"):
        sheet.append([row["model"], row["rouge1"], row["rouge2"], row["rougeL"]])
    full_sheet = workbook.create_sheet("Full Test (8847)")
    full_sheet.append(["model", "rouge1", "rouge2", "rougeL"])
    for row in full_test.to_dict("records"):
        full_sheet.append([row["model"], row["rouge1"], row["rouge2"], row["rougeL"]])
    workbook.save(OUTPUT_DIR / "rouge_comparison.xlsx")

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["rouge1", "rouge2", "rougeL"]
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    x_positions = range(len(rows))
    width = 0.24

    # Offset metric bars within each model group for side-by-side comparison.
    figure, axis = plt.subplots(figsize=(9, 5))
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        axis.bar(
            [position + (index - 1) * width for position in x_positions],
            comparison[metric],
            width,
            label=label,
        )

    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(comparison["model"])
    axis.set_ylabel("ROUGE F1")
    axis.set_title("Hindi News Summarization: Available Model Comparison")
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    figure.tight_layout()
    figure.savefig(CHART_DIR / "rouge_comparison_common100.png", dpi=200)
    plt.close(figure)

    print(comparison.to_string(index=False))
    print("Common-sample scores use the same 100 article IDs for all four models.")


if __name__ == "__main__":
    main()
