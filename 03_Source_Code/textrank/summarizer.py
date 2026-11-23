# Extractive Hindi summarization using TextRank.
# This docstring identifies the file's extractive summarization purpose.

from __future__ import annotations
# Allow modern type annotations without eagerly evaluating referenced types.

import argparse
# Import argparse to read TextRank command-line options.
import unicodedata
# Import unicodedata to normalize and classify Unicode Hindi characters.
from pathlib import Path
# Import Path to build workspace-relative input and output paths.

import networkx as nx
# Import NetworkX to create and rank the sentence similarity graph.
import pandas as pd
# Import pandas to load test data and write prediction/evaluation CSV files.
from rouge_score import rouge_scorer
# Import the ROUGE scoring implementation.
from sklearn.feature_extraction.text import TfidfVectorizer
# Import TF-IDF vectorization for character-level sentence features.
from sklearn.metrics.pairwise import cosine_similarity
# Import cosine similarity to calculate pairwise sentence relationships.


ROOT = Path(__file__).resolve().parents[2]
# Resolve the repository root relative to this source file.
DEFAULT_TEST_SET = ROOT / "01_Dataset" / "processed" / "common_test_set.csv"
# Set the default shared test CSV path.
DEFAULT_OUTPUT = ROOT / "05_Predictions" / "textrank_predictions.csv"
# Set the default destination for generated TextRank summaries.
DEFAULT_SCORES_OUTPUT = ROOT / "06_Evaluation_Results" / "textrank_rouge_scores.csv"
# Set the default destination for aggregate ROUGE scores.


class HindiTokenizer:
    # Unicode-aware tokenizer for Hindi ROUGE evaluation.
    # Define the custom tokenizer supplied to the ROUGE library.

    def tokenize(self, text: str) -> list[str]:
        # Group Unicode letters, marks, and numbers into Hindi-aware tokens.
        text = unicodedata.normalize("NFC", str(text).lower())
        # Normalize text and convert it to lowercase before tokenization.
        tokens = []
        # Create the list that will contain completed tokens.
        current = []
        # Create a buffer for characters belonging to the current token.

        for character in text:
            # Examine one character at a time.
            category = unicodedata.category(character)
            # Obtain the Unicode category for the character.
            if category[0] in {"L", "M", "N"} or (
                character in "\u200c\u200d" and current
            ):
                # Include letters, marks, numbers, and active joiner characters.
                current.append(character)
                # Append the character to the token buffer.
            elif current:
                # End the current token when a separator is encountered.
                tokens.append("".join(current))
                # Join and save the buffered characters.
                current = []
                # Clear the buffer for the next token.

        if current:
            # Check whether a final token remains after the loop.
            tokens.append("".join(current))
            # Save the final buffered token.

        return tokens
        # Return the Hindi-aware token list to ROUGE.


def split_sentences(article: str) -> list[str]:
    # Split an article at Hindi danda and common terminal punctuation.
    article = unicodedata.normalize("NFC", str(article)).strip()
    # Normalize and trim the input article.
    if not article:
        return []
        # Return an empty list for a blank article.

    sentences = [
        sentence.strip()
        for sentence in __import__("re").split(r"(?<=[।!?])\s+|(?<=\.)\s+", article)
        if sentence.strip()
    ]
    # Split on terminal punctuation and discard whitespace-only fragments.
    return sentences
    # Return the ordered list of article sentences.


def summarize_text(
    article: str,
    max_sentences: int = 3,
    max_words: int = 100,
) -> str:
    # Return the highest-ranked sentences in their original article order.
    sentences = split_sentences(article)
    # Split the input article into candidate sentences.
    if not sentences:
        return ""
        # Return an empty summary when there are no sentences.
    if len(sentences) == 1:
        # Handle a one-sentence article separately.
        words = sentences[0].split()
        # Split the single sentence into words.
        return " ".join(words[:max_words])
        # Return only the requested maximum number of words.

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        lowercase=True,
    )
    # Configure character n-gram TF-IDF features for Hindi text.
    # Compare sentence character n-grams so Hindi spelling and inflection patterns
    # contribute to the graph even when word tokenization is imperfect.
    sentence_vectors = vectorizer.fit_transform(sentences)
    # Convert each sentence into a character n-gram TF-IDF vector.
    similarity = cosine_similarity(sentence_vectors)
    # Calculate cosine similarity for every pair of sentence vectors.

    # PageRank scores each sentence by how strongly it is connected to the others.
    graph = nx.from_numpy_array(similarity)
    # Build a complete weighted graph from the similarity matrix.
    scores = nx.pagerank(graph, max_iter=500)
    # Rank sentences with PageRank using a bounded iteration count.
    ranked = sorted(range(len(sentences)), key=lambda index: scores[index], reverse=True)
    # Sort sentence indices from highest to lowest PageRank score.

    selected = ranked[:max_sentences]
    # Select the requested number of top-ranked sentences.
    summary_sentences = [sentences[index] for index in sorted(selected)]
    # Restore original article order among selected sentences.
    summary = " ".join(summary_sentences)
    # Join the selected sentences into a summary string.

    words = summary.split()
    # Split the summary to enforce the word limit.
    if len(words) > max_words:
        # Check whether the summary exceeds the maximum word count.
        summary = " ".join(words[:max_words])
        # Truncate the summary to the requested word limit.

    return summary
    # Return the final extractive summary.


def score_predictions(output: pd.DataFrame) -> pd.DataFrame:
    # Average per-example Hindi ROUGE F1 scores over the prediction frame.
    scorer = rouge_scorer.RougeScorer(
        # Configure the ROUGE scorer for the requested Hindi metrics.
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=HindiTokenizer(),
    )
    # Finish configuring the scorer with the Hindi-aware tokenizer.
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    # Initialize accumulators for the three ROUGE F1 metrics.

    for reference, prediction in zip(
        output["reference"].fillna(""),
        output["prediction"].fillna(""),
    ):
        scores = scorer.score(str(reference), str(prediction))
        # Calculate ROUGE scores for one reference/prediction pair.
        for metric in totals:
            totals[metric] += scores[metric].fmeasure
            # Add the pair's F1 value to the matching metric accumulator.

    return pd.DataFrame([
        {"metric": metric, "f1": total / len(output)}
        for metric, total in totals.items()
    ])
    # Create a DataFrame containing mean F1 scores for every metric.


def main() -> None:
    # Generate TextRank predictions, score them, and write both output files.
    parser = argparse.ArgumentParser()
    # Create a command-line argument parser.
    parser.add_argument("--input", default=str(DEFAULT_TEST_SET))
    # Set the default or custom test CSV path.
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    # Set the default or custom prediction CSV path.
    parser.add_argument("--scores-output", default=str(DEFAULT_SCORES_OUTPUT))
    # Set the default or custom ROUGE scores CSV path.
    parser.add_argument("--max-sentences", type=int, default=3)
    # Configure the maximum number of ranked sentences per summary.
    parser.add_argument("--max-words", type=int, default=100)
    # Configure the maximum number of words per summary.
    parser.add_argument("--limit", type=int, default=None)
    # Add an optional test-row limit for quick runs.
    args = parser.parse_args()
    # Parse command-line arguments into the args namespace.

    test = pd.read_csv(args.input, encoding="utf-8")
    # Load the test CSV using UTF-8 encoding.
    if args.limit:
        test = test.head(args.limit)
        # Restrict the test frame to the requested number of initial rows.

    predictions = [
        summarize_text(
            article,
            max_sentences=args.max_sentences,
            max_words=args.max_words,
        )
        for article in test["article"].fillna("")
    ]
    # Generate one extractive summary for every test article.

    output = pd.DataFrame({
        "article_id": test["article_id"],
        "reference": test["reference"],
        "prediction": predictions,
    })
    # Build the prediction table from IDs, references, and summaries.

    output_path = Path(args.output)
    # Convert the prediction destination into a Path.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Create the prediction destination directory if necessary.
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    # Write predictions to CSV with UTF-8 BOM and no index.

    scores = score_predictions(output)
    # Calculate average Hindi ROUGE scores for the predictions.
    scores_path = Path(args.scores_output)
    # Convert the score destination into a Path.
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    # Create the score destination directory if necessary.
    scores.to_csv(scores_path, index=False)
    # Write aggregate ROUGE scores without an index.

    print(f"Saved {len(output)} predictions to {output_path}")
    # Report the number and location of saved predictions.
    for _, row in scores.iterrows():
        print(f"{row['metric']}: {row['f1']:.6f}")
        # Print each metric name and mean F1 score.


if __name__ == "__main__":
    main()
    # Execute TextRank prediction and evaluation only when run directly.
