# Extractive Hindi summarization using TextRank.

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import networkx as nx
import pandas as pd
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_SET = ROOT / "01_Dataset" / "processed" / "common_test_set.csv"
DEFAULT_OUTPUT = ROOT / "05_Predictions" / "textrank_predictions.csv"
DEFAULT_SCORES_OUTPUT = ROOT / "06_Evaluation_Results" / "textrank_rouge_scores.csv"


class HindiTokenizer:
    # Unicode-aware tokenizer for Hindi ROUGE evaluation.

    def tokenize(self, text: str) -> list[str]:
        # Group Unicode letters, marks, and numbers into Hindi-aware tokens.
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


def split_sentences(article: str) -> list[str]:
    # Split an article at Hindi danda and common terminal punctuation.
    article = unicodedata.normalize("NFC", str(article)).strip()
    if not article:
        return []

    sentences = [
        sentence.strip()
        for sentence in __import__("re").split(r"(?<=[।!?])\s+|(?<=\.)\s+", article)
        if sentence.strip()
    ]
    return sentences


def summarize_text(
    article: str,
    max_sentences: int = 3,
    max_words: int = 100,
) -> str:
    # Return the highest-ranked sentences in their original article order.
    sentences = split_sentences(article)
    if not sentences:
        return ""
    if len(sentences) == 1:
        words = sentences[0].split()
        return " ".join(words[:max_words])

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        lowercase=True,
    )
    # Compare sentence character n-grams so Hindi spelling and inflection patterns
    # contribute to the graph even when word tokenization is imperfect.
    sentence_vectors = vectorizer.fit_transform(sentences)
    similarity = cosine_similarity(sentence_vectors)

    # PageRank scores each sentence by how strongly it is connected to the others.
    graph = nx.from_numpy_array(similarity)
    scores = nx.pagerank(graph, max_iter=500)
    ranked = sorted(range(len(sentences)), key=lambda index: scores[index], reverse=True)

    selected = ranked[:max_sentences]
    summary_sentences = [sentences[index] for index in sorted(selected)]
    summary = " ".join(summary_sentences)

    words = summary.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words])

    return summary


def score_predictions(output: pd.DataFrame) -> pd.DataFrame:
    # Average per-example Hindi ROUGE F1 scores over the prediction frame.
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=HindiTokenizer(),
    )
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for reference, prediction in zip(
        output["reference"].fillna(""),
        output["prediction"].fillna(""),
    ):
        scores = scorer.score(str(reference), str(prediction))
        for metric in totals:
            totals[metric] += scores[metric].fmeasure

    return pd.DataFrame([
        {"metric": metric, "f1": total / len(output)}
        for metric, total in totals.items()
    ])


def main() -> None:
    # Generate TextRank predictions, score them, and write both output files.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_TEST_SET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--scores-output", default=str(DEFAULT_SCORES_OUTPUT))
    parser.add_argument("--max-sentences", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    test = pd.read_csv(args.input, encoding="utf-8")
    if args.limit:
        test = test.head(args.limit)

    predictions = [
        summarize_text(
            article,
            max_sentences=args.max_sentences,
            max_words=args.max_words,
        )
        for article in test["article"].fillna("")
    ]

    output = pd.DataFrame({
        "article_id": test["article_id"],
        "reference": test["reference"],
        "prediction": predictions,
    })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    scores = score_predictions(output)
    scores_path = Path(args.scores_output)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(scores_path, index=False)

    print(f"Saved {len(output)} predictions to {output_path}")
    for _, row in scores.iterrows():
        print(f"{row['metric']}: {row['f1']:.6f}")


if __name__ == "__main__":
    main()
