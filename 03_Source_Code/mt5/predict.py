"""mT5-small prediction on the shared test set (Anshu, mt5-finetuning branch).

Writes article_id, reference, prediction to a CSV and scores with the
Unicode-aware HindiTokenizer agreed for the Seq2Seq baseline.

Usage:
  python 03_Source_Code/mt5/predict.py --model_dir /kaggle/working/mt5_checkpoints/final --output 05_Predictions/mt5_predictions.csv --scores_output 06_Evaluation_Results/mt5_rouge_scores.csv
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TEST_PATH = "01_Dataset/processed/common_test_set.csv"
MAX_SOURCE_LEN = 400
MAX_TARGET_LEN = 100
REPETITION_PENALTY = 1.5
NUM_BEAMS = 1  # greedy, matching the Seq2Seq best run


class HindiTokenizer:
    """Unicode-aware tokenizer for Hindi ROUGE evaluation (same as seq2seq/predict.py)."""

    def tokenize(self, text: str) -> list[str]:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output", default="05_Predictions/mt5_predictions.csv")
    parser.add_argument("--scores_output", default="06_Evaluation_Results/mt5_rouge_scores.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N rows (for a quick check)")
    args = parser.parse_args()

    test = pd.read_csv(TEST_PATH, encoding="utf-8")
    if args.limit:
        test = test.head(args.limit)
    print(f"Test rows: {len(test)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device).eval()

    articles = test["article"].fillna("").astype(str).tolist()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(articles), args.batch_size):
            batch = articles[start:start + args.batch_size]
            enc = tokenizer(batch, max_length=MAX_SOURCE_LEN, truncation=True,
                            padding=True, return_tensors="pt").to(device)
            out = model.generate(
                **enc,
                max_length=MAX_TARGET_LEN,
                num_beams=NUM_BEAMS,
                do_sample=False,
                repetition_penalty=REPETITION_PENALTY,
            )
            predictions.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
            if (start // args.batch_size) % 50 == 0:
                print(f"Generated {len(predictions)}/{len(articles)}")

    output = pd.DataFrame({
        "article_id": test["article_id"],
        "reference": test["reference"],
        "prediction": predictions,
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(output)} predictions to {args.output}")

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=HindiTokenizer()
    )
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for ref, hyp in zip(output["reference"].fillna(""), output["prediction"].fillna("")):
        s = scorer.score(str(ref), str(hyp))
        for m in totals:
            totals[m] += s[m].fmeasure
    n = len(output)
    scores = pd.DataFrame([{"metric": m, "f1": totals[m] / n} for m in totals])
    Path(args.scores_output).parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.scores_output, index=False)
    for m in totals:
        print(f"{m}: {totals[m] / n:.6f}")
    print(f"rows scored: {n}")


if __name__ == "__main__":
    main()
