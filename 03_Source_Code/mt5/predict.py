"""mT5-small prediction script for generating summaries and Hindi ROUGE scores."""
# This docstring describes the script, its purpose, and how it can be invoked.

import argparse
# Import argparse to read command-line options such as the model directory and output paths.
import unicodedata
# Import unicodedata to normalize Hindi text and inspect Unicode character categories.
from pathlib import Path
# Import Path to create output directories and work safely with filesystem paths.

import pandas as pd
# Import pandas and give it the short name pd for loading CSV data and creating result tables.
import torch
# Import PyTorch for CUDA/CPU device selection and disabled-gradient model inference.
from rouge_score import rouge_scorer
# Import the ROUGE scorer implementation used to calculate n-gram overlap scores.
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# Import Transformers classes that can load the mT5 model and matching tokenizer.

TEST_PATH = "01_Dataset/processed/common_test_set.csv"
# Store the shared test dataset path so the script always reads the same test set.
MAX_SOURCE_LEN = 400
# Limit each source article to 400 tokens before passing it to the model.
MAX_TARGET_LEN = 100
# Limit each generated summary to 100 tokens to keep predictions comparable.
REPETITION_PENALTY = 1.5
# Penalize repeated tokens while mT5 generates a summary.
NUM_BEAMS = 1
# Use greedy decoding with one beam, matching the Seq2Seq baseline configuration.


class HindiTokenizer:
    # Define the Unicode-aware tokenizer used by the ROUGE library for Hindi text.
    """Unicode-aware tokenizer for Hindi ROUGE evaluation."""
    # Describe the purpose and behavior of the HindiTokenizer class.

    def tokenize(self, text: str) -> list[str]:
        # Define a method that converts text into Unicode-aware Hindi tokens.
        text = unicodedata.normalize("NFC", str(text).lower())
        # Convert the value to text, lowercase it, and normalize its Unicode representation.
        tokens = []
        # Create an empty list that will hold completed tokens.
        current = []
        # Create an empty list for characters belonging to the token currently being built.
        for character in text:
            # Examine one character at a time from the normalized text.
            category = unicodedata.category(character)
            # Obtain the Unicode general category, such as letter or punctuation.
            if category[0] in {"L", "M", "N"} or (
                # Accept letters, marks, and numbers, or a joiner that follows a token.
                character in "\u200c\u200d" and current
            ):
                # Enter this block when the character should remain inside the current token.
                current.append(character)
                # Append the accepted character to the token currently being assembled.
            elif current:
                # Enter this block for a separator when a token has already been assembled.
                tokens.append("".join(current))
                # Join the current characters and add the completed token to the token list.
                current = []
                # Reset the character buffer for the next token.
        if current:
            # Check whether characters remain after the final loop iteration.
            tokens.append("".join(current))
            # Join and append the remaining characters as the final token.
        return tokens
        # Return all extracted Unicode-aware tokens to the ROUGE scorer.


def main():
    # Define the main function that orchestrates loading, prediction, and evaluation.
    parser = argparse.ArgumentParser()
    # Create a command-line argument parser.
    parser.add_argument("--model_dir", required=True)
    # Require a model directory containing the fine-tuned mT5 model and tokenizer.
    parser.add_argument("--output", default="05_Predictions/mt5_predictions.csv")
    # Set the default CSV path for article IDs, references, and predictions.
    parser.add_argument("--scores_output", default="06_Evaluation_Results/mt5_rouge_scores.csv")
    # Set the default CSV path for the final aggregate ROUGE scores.
    parser.add_argument("--batch_size", type=int, default=16)
    # Add a configurable batch size with a default of 16 articles.
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N rows (for a quick check)")
    # Add an optional limit for quickly scoring only the first N test rows.
    args = parser.parse_args()
    # Parse the supplied command-line arguments into an accessible namespace.

    test = pd.read_csv(TEST_PATH, encoding="utf-8")
    # Load the shared test CSV using UTF-8 text encoding.
    if args.limit:
        # Limit the test data only when the user supplied a nonzero limit.
        test = test.head(args.limit)
        # Keep only the requested first N rows of the test DataFrame.
    print(f"Test rows: {len(test)}")
    # Print the number of test rows that will be processed.

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Select CUDA GPU inference when available; otherwise select CPU inference.
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    # Load the tokenizer saved alongside the fine-tuned model directory.
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device).eval()
    # Load mT5, move it to the selected device, and switch it into evaluation mode.

    articles = test["article"].fillna("").astype(str).tolist()
    # Extract articles, replace missing values with empty strings, and convert them to a Python list.
    predictions = []
    # Initialize an empty list that will receive generated summaries in input order.
    with torch.no_grad():
        # Disable gradient calculation because prediction does not require backpropagation.
        for start in range(0, len(articles), args.batch_size):
            # Iterate through the article list in steps equal to the requested batch size.
            batch = articles[start:start + args.batch_size]
            # Slice the article list to obtain the articles in the current batch.
            enc = tokenizer(batch, max_length=MAX_SOURCE_LEN, truncation=True,
                            padding=True, return_tensors="pt").to(device)
            # Tokenize and pad the batch, truncate it to 400 tokens, and move tensors to the device.
            out = model.generate(
                # Generate summaries using the current batch of encoded articles.
                **enc,
                # Unpack the tokenizer output fields into the model's named generation arguments.
                max_length=MAX_TARGET_LEN,
                # Limit each generated summary to at most 100 tokens.
                num_beams=NUM_BEAMS,
                # Request greedy generation because NUM_BEAMS is 1.
                do_sample=False,
                # Disable sampling so generation is deterministic and beam-free.
                repetition_penalty=REPETITION_PENALTY,
                # Apply the repetition penalty to discourage repeated tokens.
            )
            # Close the model.generate call argument list.
            predictions.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
            # Decode generated token IDs into readable text and append each summary to predictions.
            if (start // args.batch_size) % 50 == 0:
                # Print progress only on the first batch and every 50 batches.
                print(f"Generated {len(predictions)}/{len(articles)}")
                # Report how many summaries have been generated out of the full article count.

    output = pd.DataFrame({
        # Create the prediction CSV table from the source article IDs, references, and predictions.
        "article_id": test["article_id"],
        # Add the original test article identifiers as a column.
        "reference": test["reference"],
        # Add the human-written reference summaries as a column.
        "prediction": predictions,
        # Add the model-generated summaries as a column.
    })
    # Close and finalize the prediction DataFrame dictionary.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Create the prediction output directory, including missing parent directories.
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    # Write predictions to CSV without a pandas index and with a UTF-8 BOM for compatibility.
    print(f"Saved {len(output)} predictions to {args.output}")
    # Confirm how many predictions were saved and report their destination.

    scorer = rouge_scorer.RougeScorer(
        # Create a ROUGE scorer configured for the three requested metrics.
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=HindiTokenizer()
        # Score unigrams, bigrams, and longest common subsequences without English stemming.
    )
    # Close the ROUGE scorer constructor.
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    # Initialize accumulators for the F1 score of each ROUGE metric.
    for ref, hyp in zip(output["reference"].fillna(""), output["prediction"].fillna("")):
        # Pair each reference with its prediction and replace missing text with an empty string.
        s = scorer.score(str(ref), str(hyp))
        # Calculate ROUGE scores for this reference/prediction pair.
        for m in totals:
            # Visit each configured ROUGE metric accumulator.
            totals[m] += s[m].fmeasure
            # Add this row's F1 measure to the metric's running total.
    n = len(output)
    # Store the number of prediction rows used in evaluation.
    scores = pd.DataFrame([{"metric": m, "f1": totals[m] / n} for m in totals])
    # Build a DataFrame containing each metric's average F1 score.
    Path(args.scores_output).parent.mkdir(parents=True, exist_ok=True)
    # Create the scores output directory if it does not already exist.
    scores.to_csv(args.scores_output, index=False)
    # Save the aggregate metric table to CSV without an index column.
    for m in totals:
        # Iterate through each ROUGE metric to print its average.
        print(f"{m}: {totals[m] / n:.6f}")
        # Print the metric name and its mean F1 value with six decimal places.
    print(f"rows scored: {n}")
    # Print the total number of rows included in ROUGE evaluation.


if __name__ == "__main__":
    # Run main only when this file is executed directly, not when imported as a module.
    main()
    # Invoke the prediction and evaluation workflow.
