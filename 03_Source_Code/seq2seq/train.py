"""Train the Hindi XL-Sum attention-based seq2seq model."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
import unicodedata

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import Seq2SeqAttention


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = ROOT / "04_Models" / "seq2seq_attention" / "checkpoints"
SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
XLSUM_REVISION = "651dcf16e996d411b12f0bb7b8fc671788f24193"
XLSUM_HINDI_BASE_URL = (
    "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/"
    f"{XLSUM_REVISION}/hindi/"
)
XLSUM_HINDI_FILES = {
    "train": [
        f"{XLSUM_HINDI_BASE_URL}xlsum-train-00000-of-00002.parquet",
        f"{XLSUM_HINDI_BASE_URL}xlsum-train-00001-of-00002.parquet",
    ],
    "validation": f"{XLSUM_HINDI_BASE_URL}xlsum-validation.parquet",
}


def tokenize(text: str) -> list[str]:
    """Split Unicode text while keeping Hindi combining marks with their words."""
    tokens: list[str] = []
    current: list[str] = []
    for character in str(text).lower():
        category = unicodedata.category(character)
        if category[0] in {"L", "M", "N"} or (
            character in "\u200c\u200d" and current
        ):
            current.append(character)
        elif character.isspace():
            if current:
                tokens.append("".join(current))
                current = []
        else:
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(character)
    if current:
        tokens.append("".join(current))
    return tokens


def encode(text: str, vocabulary: dict[str, int], max_length: int) -> list[int]:
    ids = [vocabulary["<bos>"]]
    ids.extend(vocabulary.get(token, vocabulary["<unk>"]) for token in tokenize(text))
    ids.append(vocabulary["<eos>"])
    return ids[:max_length - 1] + [vocabulary["<eos>"]] if len(ids) > max_length else ids


class SummaryDataset(Dataset[tuple[list[int], list[int]]]):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        vocabulary: dict[str, int],
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.rows = rows
        self.vocabulary = vocabulary
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        row = self.rows[index]
        if not str(row.get("text", "")).strip() or not str(row.get("summary", "")).strip():
            raise ValueError(f"Empty article-summary pair at dataset row {index}.")
        return (
            encode(row["text"], self.vocabulary, self.max_source_length),
            encode(row["summary"], self.vocabulary, self.max_target_length),
        )


def collate_batch(
    batch: list[tuple[list[int], list[int]]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    source_max = max(len(source) for source, _ in batch)
    target_max = max(len(target) for _, target in batch)
    sources = [source + [pad_id] * (source_max - len(source)) for source, _ in batch]
    targets = [target + [pad_id] * (target_max - len(target)) for _, target in batch]
    return torch.tensor(sources, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def build_vocabulary(
    rows: Sequence[dict[str, Any]], max_size: int, min_frequency: int
) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for row in rows:
        for token in tokenize(f"{row['text']} {row['summary']}"):
            frequencies[token] = frequencies.get(token, 0) + 1
    tokens = sorted(
        (token for token, count in frequencies.items() if count >= min_frequency),
        key=lambda token: (-frequencies[token], token),
    )
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token in tokens[: max(0, max_size - len(vocabulary))]:
        vocabulary[token] = len(vocabulary)
    return vocabulary


def limit_rows(rows: Any, limit: int | None) -> Any:
    if limit is None:
        return rows
    return rows.select(range(min(limit, len(rows))))


def is_valid_summary_pair(row: dict[str, Any]) -> bool:
    return bool(str(row.get("text") or "").strip()) and bool(
        str(row.get("summary") or "").strip()
    )


def filter_valid_summary_pairs(rows: Any, split_name: str) -> Any:
    """Drop rows without both article text and a reference summary."""
    filtered_rows = rows.filter(
        is_valid_summary_pair,
        desc=f"Filtering blank article-summary pairs from {split_name}",
    )
    removed_count = len(rows) - len(filtered_rows)
    if removed_count:
        print(f"Skipped {removed_count} blank article-summary pair(s) in {split_name}.")
    return filtered_rows


def load_xlsum_hindi() -> Any:
    """Load the published Parquet splits without the removed dataset script."""
    return load_dataset("parquet", data_files=XLSUM_HINDI_FILES)


def evaluate_loss(
    model: Seq2SeqAttention,
    loader: Iterable[Any],
    criterion: nn.Module,
    device: torch.device,
    pad_id: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for source, target in loader:
            source, target = source.to(device), target.to(device)
            logits = model(source, target, teacher_forcing_ratio=1.0)
            labels = target[:, 1:]
            token_count = int(labels.ne(pad_id).sum().item())
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            total_loss += loss.item() * token_count
            total_tokens += token_count
    if not total_tokens:
        raise ValueError("Validation split contains no target tokens.")
    return total_loss / total_tokens


def train(args: argparse.Namespace) -> Path:
    if args.max_source_length < 2 or args.max_target_length < 2:
        raise ValueError("Maximum source and target lengths must be at least 2.")
    if not 0.0 <= args.teacher_forcing <= 1.0:
        raise ValueError("Teacher-forcing ratio must be between 0 and 1.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training on {device}")

    dataset = load_xlsum_hindi()
    train_rows = limit_rows(
        filter_valid_summary_pairs(dataset["train"], "train"),
        args.max_train_samples,
    )
    validation_rows = limit_rows(
        filter_valid_summary_pairs(dataset["validation"], "validation"),
        args.max_validation_samples,
    )
    if not train_rows or not validation_rows:
        raise ValueError("Training and validation splits must both contain rows.")

    vocabulary = build_vocabulary(train_rows, args.max_vocab_size, args.min_frequency)
    pad_id = vocabulary["<pad>"]
    train_data = SummaryDataset(
        train_rows, vocabulary, args.max_source_length, args.max_target_length
    )
    validation_data = SummaryDataset(
        validation_rows, vocabulary, args.max_source_length, args.max_target_length
    )
    if len(train_data) == 0 or len(validation_data) == 0:
        raise ValueError("Training and validation splits must both contain rows.")
    collate = lambda batch: collate_batch(batch, pad_id)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    model = Seq2SeqAttention(
        len(vocabulary), args.embedding_dim, args.hidden_size, pad_id, args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "best_model.pt"
    history_path = args.checkpoint_dir / "training_history.csv"
    history_path.write_text("epoch,train_loss,validation_loss\n", encoding="utf-8")
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        token_count = 0
        for source, target in train_loader:
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, target, teacher_forcing_ratio=args.teacher_forcing)
            labels = target[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            tokens = int(labels.ne(pad_id).sum().item())
            running_loss += loss.item() * tokens
            token_count += tokens

        validation_loss = evaluate_loss(
            model, validation_loader, criterion, device, pad_id
        )
        train_loss = running_loss / max(token_count, 1)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}"
        )
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(f"{epoch},{train_loss:.6f},{validation_loss:.6f}\n")
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocabulary": vocabulary,
                    "config": {
                        "embedding_dim": args.embedding_dim,
                        "hidden_size": args.hidden_size,
                        "dropout": args.dropout,
                        "max_source_length": args.max_source_length,
                        "max_target_length": args.max_target_length,
                        "seed": args.seed,
                        "best_validation_loss": best_loss,
                    },
                },
                checkpoint_path,
            )
            (args.checkpoint_dir / "training_settings.json").write_text(
                json.dumps(vars(args) | {"device": str(device)}, indent=2, default=str),
                encoding="utf-8",
            )
    print(f"Best checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--teacher-forcing", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-source-length", type=int, default=400)
    parser.add_argument("--max-target-length", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
