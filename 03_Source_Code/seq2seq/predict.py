"""Generate Hindi summaries using configurable constrained decoding."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from rouge_score import rouge_scorer

from model import Seq2SeqAttention
from train import ROOT, collate_batch, encode


DEFAULT_CHECKPOINT = (
    ROOT / "04_Models" / "seq2seq_attention"
    / "checkpoints" / "best_model.pt"
)

DEFAULT_TEST_SET = (
    ROOT / "01_Dataset" / "processed" / "common_test_set.csv"
)

DEFAULT_SETTINGS = (
    ROOT / "04_Models" / "seq2seq_attention"
    / "training_settings.json"
)

DEFAULT_OUTPUT = (
    ROOT / "05_Predictions"
    / "seq2seq_predictions_trigram_blocking.csv"
)

DEFAULT_SCORES_OUTPUT = (
    ROOT / "06_Evaluation_Results"
    / "seq2seq_trigram_blocking_rouge.csv"
)


class HindiTokenizer:
    """Unicode-aware tokenizer for Hindi ROUGE evaluation."""

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


def decode(ids: list[int], inverse_vocabulary: list[str]) -> str:
    tokens = []

    for token_id in ids:
        token = inverse_vocabulary[token_id]

        if token in {"<pad>", "<bos>"}:
            continue

        if token == "<eos>":
            break

        tokens.append(token)

    text = " ".join(tokens)
    return re.sub(r"\s+([।,!?;:])", r"\1", text).strip()


def _repeated_ngram_continuations(
    token_ids: list[int],
    ngram_size: int,
    eos_id: int,
) -> set[int]:
    """Return continuations that would repeat an n-gram, except EOS."""
    if ngram_size <= 0 or len(token_ids) < ngram_size:
        return set()

    prefix_size = ngram_size - 1
    current_prefix = tuple(token_ids[-prefix_size:]) if prefix_size else ()
    blocked = set()

    for start in range(len(token_ids) - ngram_size + 1):
        prefix = tuple(token_ids[start:start + prefix_size])
        continuation = token_ids[start + prefix_size]
        if prefix == current_prefix and continuation != eos_id:
            blocked.add(continuation)

    return blocked


@torch.inference_mode()
def generate_with_penalty(
    model: Seq2SeqAttention,
    sources: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_length: int,
    penalty: float,
    no_repeat_ngram_size: int = 3,
    min_length: int = 0,
) -> list[list[int]]:
    """Batched greedy decoding with optional n-gram and minimum-length constraints."""
    if max_length < 1:
        raise ValueError("Maximum summary length must be at least one.")
    if penalty <= 0:
        raise ValueError("Repetition penalty must be greater than zero.")
    if no_repeat_ngram_size < 0:
        raise ValueError("No-repeat n-gram size cannot be negative.")
    if min_length < 0 or min_length > max_length:
        raise ValueError("Minimum summary length must be between 0 and max_length.")

    model.eval()

    encoded, source_mask, hidden, cell = model.encode(sources)

    batch_size = sources.size(0)

    token = torch.full(
        (batch_size,),
        bos_id,
        dtype=torch.long,
        device=sources.device,
    )

    generated = [[bos_id] for _ in range(batch_size)]

    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=sources.device,
    )

    for _ in range(max_length):
        logits, hidden, cell = model.decode_step(
            token.unsqueeze(1),
            encoded,
            source_mask,
            hidden,
            cell,
        )

        logits = logits.clone()

        # Apply repetition penalty independently to each article.
        for i in range(batch_size):
            if finished[i]:
                continue

            seen_tokens = set(generated[i])
            seen_tokens.discard(bos_id)
            seen_tokens.discard(pad_id)
            seen_tokens.discard(eos_id)

            for token_id in seen_tokens:
                if logits[i, token_id] < 0:
                    logits[i, token_id] *= penalty
                else:
                    logits[i, token_id] /= penalty

            content_tokens = generated[i][1:]
            blocked_tokens = _repeated_ngram_continuations(
                content_tokens,
                no_repeat_ngram_size,
                eos_id,
            )
            if blocked_tokens:
                logits[i, list(blocked_tokens)] = float("-inf")

            if len(content_tokens) < min_length:
                logits[i, eos_id] = float("-inf")

        # Never generate padding or BOS.
        logits[:, pad_id] = float("-inf")
        logits[:, bos_id] = float("-inf")

        next_token = logits.argmax(dim=-1)

        next_token = torch.where(
            finished,
            torch.full_like(next_token, eos_id),
            next_token,
        )

        for i, token_id in enumerate(next_token.tolist()):
            generated[i].append(token_id)

        finished |= next_token.eq(eos_id)

        if finished.all():
            break

        token = next_token

    return generated


def load_test_rows(
    path: Path,
    limit: int | None,
) -> pd.DataFrame:

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Test dataset not found: {path}")

    frame = pd.read_csv(path, dtype={"article_id": str})

    required = {"article_id", "reference"}

    if not required.issubset(frame.columns):
        raise ValueError("Test CSV must contain article_id and reference.")

    if "article" not in frame and "text" not in frame:
        raise ValueError("Test CSV must contain article or text.")

    if frame.empty:
        raise ValueError("Test dataset is empty.")

    if frame[["article_id", "reference"]].isna().any().any():
        raise ValueError("Missing article IDs or references.")

    if limit is not None:
        frame = frame.head(limit)

    return frame


def predict(args: argparse.Namespace) -> Path:

    # Load training and inference settings.
    with open(args.settings, "r", encoding="utf-8") as f:
        settings = json.load(f)

    inference_settings = settings.get("inference", {})

    penalty = (
        args.repetition_penalty
        if args.repetition_penalty is not None
        else inference_settings.get("repetition_penalty", 1.5)
    )

    max_length = (
        args.max_length
        if args.max_length is not None
        else inference_settings.get(
            "max_length",
            settings["max_target_length"],
        )
    )
    no_repeat_ngram_size = (
        args.no_repeat_ngram_size
        if args.no_repeat_ngram_size is not None
        else inference_settings.get("no_repeat_ngram_size", 3)
    )
    min_length = (
        args.min_length
        if args.min_length is not None
        else inference_settings.get("min_length", 0)
    )

    if penalty <= 0:
        raise ValueError("Repetition penalty must be greater than zero.")
    if max_length < 1:
        raise ValueError("Maximum summary length must be at least one.")
    if no_repeat_ngram_size < 0:
        raise ValueError("No-repeat n-gram size cannot be negative.")
    if min_length < 0 or min_length > max_length:
        raise ValueError("Minimum summary length must be between 0 and max_length.")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint: dict[str, Any] = torch.load(
        args.checkpoint,
        map_location=device,
    )

    vocabulary = checkpoint["vocabulary"]
    config = checkpoint["config"]

    model = Seq2SeqAttention(
        len(vocabulary),
        config["embedding_dim"],
        config["hidden_size"],
        vocabulary["<pad>"],
        config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    frame = load_test_rows(args.test_set, args.limit)

    source_column = "article" if "article" in frame else "text"

    inverse_vocabulary = [""] * len(vocabulary)

    for token, index in vocabulary.items():
        inverse_vocabulary[index] = token

    predictions = []

    print(f"Device: {device}")
    print(f"Repetition penalty: {penalty}")
    print(f"No-repeat n-gram size: {no_repeat_ngram_size}")
    print(f"Minimum summary length: {min_length}")
    print(f"Maximum summary length: {max_length}")
    print(f"Test articles: {len(frame)}")

    # Generate predictions in batches.
    from tqdm.auto import tqdm

    for start in tqdm(
        range(0, len(frame), args.batch_size),
        desc="Generating summaries",
    ):
        batch_rows = frame.iloc[
            start:start + args.batch_size
        ]

        encoded_sources = [
            encode(
                article,
                vocabulary,
                config["max_source_length"],
            )
            for article in batch_rows[source_column]
        ]

        dummy_targets = [
            [vocabulary["<bos>"], vocabulary["<eos>"]]
            for _ in encoded_sources
        ]

        sources, _ = collate_batch(
            list(zip(encoded_sources, dummy_targets)),
            vocabulary["<pad>"],
        )

        generated = generate_with_penalty(
            model=model,
            sources=sources.to(device),
            bos_id=vocabulary["<bos>"],
            eos_id=vocabulary["<eos>"],
            pad_id=vocabulary["<pad>"],
            max_length=max_length,
            penalty=penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            min_length=min_length,
        )

        predictions.extend(
            decode(sequence, inverse_vocabulary)
            for sequence in generated
        )

    # Save predictions.
    output = pd.DataFrame({
        "article_id": frame["article_id"].astype(str),
        "reference": frame["reference"].astype(str),
        "prediction": predictions,
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(
        args.output,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved {len(output)} predictions to {args.output}")

    # Hindi-aware ROUGE evaluation.
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=HindiTokenizer(),
    )

    scores = [
        scorer.score(ref, pred)
        for ref, pred in zip(
            output["reference"],
            output["prediction"],
        )
    ]

    score_rows = []

    print("\nFINAL ROUGE RESULTS")

    for metric in ("rouge1", "rouge2", "rougeL"):
        precision = sum(
            score[metric].precision for score in scores
        ) / len(scores)

        recall = sum(
            score[metric].recall for score in scores
        ) / len(scores)

        fmeasure = sum(
            score[metric].fmeasure for score in scores
        ) / len(scores)

        print(
            f"{metric}: "
            f"precision={precision:.6f}, "
            f"recall={recall:.6f}, "
            f"fmeasure={fmeasure:.6f}"
        )

        score_rows.append({
            "metric": metric,
            "precision": precision,
            "recall": recall,
            "fmeasure": fmeasure,
        })

    args.scores_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(score_rows).to_csv(
        args.scores_output,
        index=False,
    )

    print(f"\nSaved ROUGE results to {args.scores_output}")

    return args.output


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--test-set",
        type=Path,
        default=DEFAULT_TEST_SET,
    )

    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--scores-output",
        type=Path,
        default=DEFAULT_SCORES_OUTPUT,
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--no-repeat-ngram-size", type=int)
    parser.add_argument("--min-length", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))

    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())
