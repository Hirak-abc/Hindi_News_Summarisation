"""mT5-small fine-tuning on XL-Sum Hindi (Anshu, mt5-finetuning branch).

Usage:
  python 03_Source_Code/mt5/train.py --smoke --output_dir /content/mt5_smoke
  python 03_Source_Code/mt5/train.py --output_dir /content/mt5_checkpoints
"""
import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

MODEL_NAME = "google/mt5-small"
SEED = 42
MAX_SOURCE_LEN = 400   # same as Seq2Seq source limit
MAX_TARGET_LEN = 100   # same as Seq2Seq target limit
SETTINGS_PATH = "04_Models/mt5_small/training_settings.json"


def keep_non_empty(row):
    return bool(row["text"] and row["text"].strip()) and bool(
        row["summary"] and row["summary"].strip()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="500 train rows, 200 validation rows, 1 epoch")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--output_dir", required=True,
                        help="Checkpoints go here. Keep this OUTSIDE the repo.")
    parser.add_argument("--settings_out", default=SETTINGS_PATH)
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1

    set_seed(SEED)

    raw = load_dataset("csebuetnlp/xlsum", "hindi")
    train = raw["train"].filter(keep_non_empty)
    valid = raw["validation"].filter(keep_non_empty)
    print(f"Rows after filtering: train={len(train)}, validation={len(valid)}")

    if args.smoke:
        train = train.shuffle(seed=SEED).select(range(500))
        valid = valid.shuffle(seed=SEED).select(range(200))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        inputs = tokenizer(batch["text"], max_length=MAX_SOURCE_LEN, truncation=True)
        labels = tokenizer(text_target=batch["summary"],
                           max_length=MAX_TARGET_LEN, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    train_tok = train.map(preprocess, batched=True, remove_columns=train.column_names)
    valid_tok = valid.map(preprocess, batched=True, remove_columns=valid.column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        seed=SEED,
        fp16=False,                  # mT5 is unstable in fp16; keep fp32
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    # Resume after a disconnect if a checkpoint already exists
    last_checkpoint = None
    if Path(args.output_dir).exists():
        last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint:
        print(f"Resuming from {last_checkpoint}")

    start = time.time()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    train_minutes = (time.time() - start) / 60

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    settings = {
        "model": MODEL_NAME,
        "dataset": "csebuetnlp/xlsum (hindi)",
        "smoke_run": args.smoke,
        "train_rows": len(train_tok),
        "validation_rows": len(valid_tok),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": "AdamW (Trainer default)",
        "weight_decay": 0.01,
        "max_source_length": MAX_SOURCE_LEN,
        "max_target_length": MAX_TARGET_LEN,
        "seed": SEED,
        "precision": "fp32",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "train_minutes": round(train_minutes, 1),
        "checkpoint_location": str(final_dir),
    }
    Path(args.settings_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.settings_out, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)
    print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()