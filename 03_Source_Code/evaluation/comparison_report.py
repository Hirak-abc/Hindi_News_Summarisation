# Build same-sample four-model and full-test ROUGE tables and chart.

from __future__ import annotations

# Use path-safe filesystem operations for data locations and generated results.
from pathlib import Path
# Resolve input and output locations with platform-safe Path objects.

# Plotting and spreadsheet libraries produce the chart and workbook artifacts.
import matplotlib.pyplot as plt
# Import matplotlib for the ROUGE comparison chart.
import openpyxl
# Import openpyxl to create the Excel comparison workbook.
import pandas as pd
# Import pandas to merge and shape model score tables.

# Reuse the project's Unicode-aware Hindi ROUGE implementation.
from rouge_evaluation import evaluate

# Resolve project directories from this script, regardless of the launch directory.
ROOT = Path(__file__).resolve().parents[2]
# Resolve the project repository root from this script.
OUTPUT_DIR = ROOT / "06_Evaluation_Results"
# Set the directory for CSV and Excel comparison outputs.
CHART_DIR = ROOT / "08_Report" / "charts"
# Set the directory for report chart images.
SAMPLE_PATH = ROOT / "01_Dataset" / "processed" / "llm_final_sample_ids.csv"
# Set the path for the deterministic 100-article comparison sample.
# Map display names to each model's prediction file for the common-sample join.
PREDICTION_FILES = {
    "TextRank": ROOT / "05_Predictions" / "textrank_predictions.csv",
    "Seq2Seq": ROOT / "05_Predictions" / "seq2seq_predictions_penalty_1.5.csv",
    "mT5": ROOT / "05_Predictions" / "mt5_predictions.csv",
    "LLM": ROOT / "05_Predictions" / "llm_predictions.csv",
}


def read_scores(path: Path, model: str) -> dict[str, float]:
    """Read a model's saved ROUGE table and normalize it to three F1 values."""
    # Pandas reads whichever CSV format the model-specific evaluation script wrote.
    frame = pd.read_csv(path)
    # Newer score files may use "f1"; older files use "fmeasure".
    if "f1" in frame.columns:
        values = dict(zip(frame["metric"], frame["f1"]))
    elif "fmeasure" in frame.columns:
        values = dict(zip(frame["metric"], frame["fmeasure"]))
    else:
        # Fail clearly instead of silently producing a comparison with missing metrics.
        raise ValueError(f"Unknown score format: {path}")
    # Return one consistent row shape for the full-test comparison table.
    return {
        "model": model,
        "rouge1": float(values["rouge1"]),
        "rouge2": float(values["rouge2"]),
        "rougeL": float(values["rougeL"]),
    }


def main() -> None:
    """Build common-sample scores, full-test scores, an Excel workbook, and a chart."""
    # Load the 100 deterministic article IDs that are shared by all four systems.
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8-sig")
    # Collect one normalized score row for each model.
    rows = []
    for model, path in PREDICTION_FILES.items():
        # Remove byte-order-mark characters from column names before merging.
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
            # An incomplete model output would make the shared-sample comparison invalid.
            raise ValueError(f"{model} predictions are incomplete for the common sample.")
        # The shared evaluate() function returns precision, recall, and F1 for three metrics.
        scores = evaluate(sample_predictions)
        # Convert the metric table to columns used by CSV and chart output.
        values = dict(zip(scores["metric"], scores["fmeasure"]))
        rows.append({
            "model": model,
            "rouge1": float(values["rouge1"]),
            "rouge2": float(values["rouge2"]),
            "rougeL": float(values["rougeL"]),
        })

    # Convert the common-sample rows into a tabular dataframe.
    comparison = pd.DataFrame(rows)
    # Create the evaluation directory before writing machine-readable outputs.
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

    # Create a workbook that keeps the common-sample and full-test views separate.
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Common Sample (100)"
    sheet.append(["model", "rouge1", "rouge2", "rougeL"])
    # Write dict values into worksheet rows in model-comparison order.
    for row in comparison.to_dict("records"):
        sheet.append([row["model"], row["rouge1"], row["rouge2"], row["rougeL"]])
    # Full-test sheet covers systems for which all 8,847 predictions are available.
    full_sheet = workbook.create_sheet("Full Test (8847)")
    full_sheet.append(["model", "rouge1", "rouge2", "rougeL"])
    for row in full_test.to_dict("records"):
        full_sheet.append([row["model"], row["rouge1"], row["rouge2"], row["rougeL"]])
    workbook.save(OUTPUT_DIR / "rouge_comparison.xlsx")

    # Ensure chart output folders exist before saving the generated PNG.
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    # Define metric series and presentation labels once for reuse in loop and legend.
    metrics = ["rouge1", "rouge2", "rougeL"]
    labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    x_positions = range(len(rows))
    width = 0.24

    # Offset metric bars within each model group for side-by-side comparison.
    figure, axis = plt.subplots(figsize=(9, 5))
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        # Shift bars left/center/right within the model position.
        axis.bar(
            [position + (index - 1) * width for position in x_positions],
            comparison[metric],
            width,
            label=label,
        )

    # Configure chart axes, title, legend, grid, and output dimensions.
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(comparison["model"])
    axis.set_ylabel("ROUGE F1")
    axis.set_title("Hindi News Summarization: Available Model Comparison")
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    figure.tight_layout()
    # Save the chart at publication-friendly resolution and close its memory.
    figure.savefig(CHART_DIR / "rouge_comparison_common100.png", dpi=200)
    plt.close(figure)

    # Print the same data shown in the CSV, followed by an explicit shared-sample note.
    print(comparison.to_string(index=False))
    print("Common-sample scores use the same 100 article IDs for all four models.")


if __name__ == "__main__":
    main()
