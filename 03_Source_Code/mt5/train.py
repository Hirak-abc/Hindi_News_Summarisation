"""mT5-small fine-tuning script for the Hindi XL-Sum dataset."""
# This docstring states that the script trains mT5 and reports its configuration.

import argparse
# Import argparse to parse command-line training options.
import json
# Import json to write the completed training settings to a JSON file.
import time
# Import time to measure how many minutes the training run takes.
from pathlib import Path
# Import Path to test and create checkpoint and settings directories.

import torch
# Import PyTorch for CUDA detection and GPU name reporting.
from datasets import load_dataset
# Import Hugging Face's dataset loader for reading the pinned Parquet files.
from transformers import (
    # Begin the grouped import of Hugging Face Transformers training components.
    AutoModelForSeq2SeqLM,
    # Import the sequence-to-sequence model loader.
    AutoTokenizer,
    # Import the matching tokenizer loader.
    DataCollatorForSeq2Seq,
    # Import the collator that batches and pads sequence-to-sequence examples.
    Seq2SeqTrainer,
    # Import the Trainer class designed for sequence-to-sequence models.
    Seq2SeqTrainingArguments,
    # Import the configuration class for training parameters.
    set_seed,
    # Import the utility that applies deterministic seeds across libraries.
)
# Finish the grouped Transformers import.
from transformers.trainer_utils import get_last_checkpoint
# Import the helper that finds the latest existing training checkpoint.

MODEL_NAME = "google/mt5-small"
# Store the Hugging Face model identifier used for the tokenizer and base model.
SEED = 42
# Set the shared random seed for reproducible data selection and training behavior.
MAX_SOURCE_LEN = 400
# Limit source articles to 400 tokens, matching the Seq2Seq source limit.
MAX_TARGET_LEN = 100
# Limit target summaries to 100 tokens, matching the Seq2Seq target limit.
SETTINGS_PATH = "04_Models/mt5/training_settings.json"
# Set the default location for the saved training configuration JSON.

XLSUM_REVISION = "651dcf16e996d411b12f0bb7b8fc671788f24193"
# Pin the XL-Sum dataset revision so later runs use the same source snapshot.
XLSUM_HINDI_BASE_URL = (
    # Combine the Hugging Face dataset URL prefix and pinned revision.
    "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/"
    # Provide the fixed dataset repository and resolve endpoint.
    f"{XLSUM_REVISION}/hindi/"
    # Append the pinned revision and Hindi dataset folder to the URL.
)
# Finish constructing the base URL for the Hindi XL-Sum files.
XLSUM_HINDI_FILES = {
    # Define the remote training and validation Parquet files used by this script.
    "train": [
        # Store both training Parquet file URLs in a list.
        f"{XLSUM_HINDI_BASE_URL}xlsum-train-00000-of-00002.parquet",
        # Add the first training Parquet file.
        f"{XLSUM_HINDI_BASE_URL}xlsum-train-00001-of-00002.parquet",
        # Add the second training Parquet file.
    ],
    # Finish the training file list.
    "validation": f"{XLSUM_HINDI_BASE_URL}xlsum-validation.parquet",
    # Store the validation Parquet file URL.
}
# Finish the dataset file mapping.


def keep_non_empty(row):
    # Define a filtering function that accepts only rows with usable article and summary text.
    return bool(row["text"] and row["text"].strip()) and bool(
        # Return true only when article text is nonempty and not whitespace.
        row["summary"] and row["summary"].strip()
        # Require summary text to be nonempty and not whitespace.
    )
    # Close the filtering expression.


def main():
    # Define the main function that runs dataset loading, training, and settings export.
    parser = argparse.ArgumentParser()
    # Create a command-line argument parser.
    parser.add_argument("--smoke", action="store_true",
                        help="500 train rows, 200 validation rows, 1 epoch")
    # Add a smoke-run flag that limits data and training time.
    parser.add_argument("--epochs", type=int, default=5)
    # Add the number of training epochs with a default of five.
    parser.add_argument("--batch_size", type=int, default=8)
    # Add the per-device batch size with a default of eight.
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    # Add the learning rate with a default of 0.0005.
    parser.add_argument("--output_dir", required=True,
                        help="Checkpoints go here. Keep this OUTSIDE the repo.")
    # Require an external output directory for checkpoints.
    parser.add_argument("--settings_out", default=SETTINGS_PATH)
    # Set the default path for the training settings JSON file.
    args = parser.parse_args()
    # Parse all command-line arguments into one namespace.

    if args.smoke:
        # Check whether the user requested the reduced smoke run.
        args.epochs = 1
        # Override the epoch count to one for smoke mode.

    set_seed(SEED)
    # Apply the fixed random seed across supported libraries.

    raw = load_dataset("parquet", data_files=XLSUM_HINDI_FILES)
    # Download and load the pinned XL-Sum Parquet files as a dataset.
    train = raw["train"].filter(keep_non_empty)
    # Keep only training rows with nonempty article text and summaries.
    valid = raw["validation"].filter(keep_non_empty)
    # Keep only validation rows with nonempty article text and summaries.
    print(f"Rows after filtering: train={len(train)}, validation={len(valid)}")
    # Print the number of usable training and validation rows.

    if args.smoke:
        # Check whether smoke mode should reduce the filtered datasets.
        train = train.shuffle(seed=SEED).select(range(500))
        # Shuffle and select exactly 500 training rows for smoke mode.
        valid = valid.shuffle(seed=SEED).select(range(200))
        # Shuffle and select exactly 200 validation rows for smoke mode.

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Load the mT5 tokenizer matching the selected base model.
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    # Load the pre-trained mT5 sequence-to-sequence model.

    def preprocess(batch):
        # Define the batched preprocessing function used for both dataset splits.
        inputs = tokenizer(batch["text"], max_length=MAX_SOURCE_LEN, truncation=True)
        # Tokenize source articles with truncation at 400 tokens.
        labels = tokenizer(text_target=batch["summary"],
                           max_length=MAX_TARGET_LEN, truncation=True)
        # Tokenize target summaries with truncation at 100 tokens.
        inputs["labels"] = labels["input_ids"]
        # Store target token IDs in the field expected by the model for training.
        return inputs
        # Return the tokenized input and label dictionary.

    train_tok = train.map(preprocess, batched=True, remove_columns=train.column_names)
    # Tokenize all training rows and remove original text columns.
    valid_tok = valid.map(preprocess, batched=True, remove_columns=valid.column_names)
    # Tokenize all validation rows and remove original text columns.

    training_args = Seq2SeqTrainingArguments(
        # Create the Hugging Face training configuration.
        output_dir=args.output_dir,
        # Write intermediate checkpoints to the requested output directory.
        num_train_epochs=args.epochs,
        # Train for the requested number of epochs.
        per_device_train_batch_size=args.batch_size,
        # Use the requested batch size for training.
        per_device_eval_batch_size=args.batch_size,
        # Use the requested batch size for validation.
        learning_rate=args.learning_rate,
        # Apply the requested optimizer learning rate.
        weight_decay=0.01,
        # Apply weight decay to reduce overfitting.
        seed=SEED,
        # Use the fixed seed for trainer-controlled randomness.
        fp16=False,
        # Keep full fp32 precision because mT5 is unstable in fp16.
        eval_strategy="epoch",
        # Evaluate the model after every epoch.
        save_strategy="epoch",
        # Save a checkpoint after every epoch.
        save_total_limit=2,
        # Retain at most the two newest checkpoints.
        logging_steps=50,
        # Print training logs every 50 update steps.
        report_to="none",
        # Disable external experiment trackers.
    )
    # Finish the Trainer argument configuration.

    trainer = Seq2SeqTrainer(
        # Create the sequence-to-sequence Trainer.
        model=model,
        # Provide the mT5 model to train.
        args=training_args,
        # Provide the configured training arguments.
        train_dataset=train_tok,
        # Provide the preprocessed training dataset.
        eval_dataset=valid_tok,
        # Provide the preprocessed validation dataset.
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        # Dynamically pad batches with the matching tokenizer and model settings.
    )
    # Finish creating the Trainer.

    last_checkpoint = None
    # Initialize the resume checkpoint to no checkpoint.
    if Path(args.output_dir).exists():
        # Check whether the checkpoint output directory already exists.
        last_checkpoint = get_last_checkpoint(args.output_dir)
        # Search the output directory for the latest saved checkpoint.
    if last_checkpoint:
        # Check whether a resumable checkpoint was found.
        print(f"Resuming from {last_checkpoint}")
        # Report the checkpoint used to resume interrupted training.

    start = time.time()
    # Record the training start time.
    trainer.train(resume_from_checkpoint=last_checkpoint)
    # Train the model, optionally resuming from the latest checkpoint.
    train_minutes = (time.time() - start) / 60
    # Convert elapsed training time from seconds to minutes.

    final_dir = Path(args.output_dir) / "final"
    # Build the path for the final saved model and tokenizer.
    trainer.save_model(str(final_dir))
    # Save the trained model and its final configuration.
    tokenizer.save_pretrained(str(final_dir))
    # Save the tokenizer alongside the trained model.

    settings = {
        # Create a dictionary summarizing the completed training run.
        "model": MODEL_NAME,
        # Record the base model identifier.
        "dataset": "csebuetnlp/xlsum (hindi)",
        # Record the dataset used.
        "smoke_run": args.smoke,
        # Record whether this was smoke mode.
        "train_rows": len(train_tok),
        # Record the number of tokenized training rows.
        "validation_rows": len(valid_tok),
        # Record the number of tokenized validation rows.
        "epochs": args.epochs,
        # Record the epoch count.
        "batch_size": args.batch_size,
        # Record the batch size.
        "learning_rate": args.learning_rate,
        # Record the learning rate.
        "optimizer": "AdamW (Trainer default)",
        # Record the default Transformers optimizer.
        "weight_decay": 0.01,
        # Record the weight-decay value.
        "max_source_length": MAX_SOURCE_LEN,
        # Record the source token limit.
        "max_target_length": MAX_TARGET_LEN,
        # Record the summary token limit.
        "seed": SEED,
        # Record the random seed.
        "precision": "fp32",
        # Record the precision used during training.
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        # Record the GPU name when CUDA is available or otherwise record CPU.
        "train_minutes": round(train_minutes, 1),
        # Record elapsed training time rounded to one decimal place.
        "checkpoint_location": str(final_dir),
        # Record the final model output location.
    }
    # Finish the training settings dictionary.
    Path(args.settings_out).parent.mkdir(parents=True, exist_ok=True)
    # Create the parent directory for the settings JSON file.
    with open(args.settings_out, "w", encoding="utf-8") as f:
        # Open the settings file for UTF-8 writing.
        json.dump(settings, f, indent=2, ensure_ascii=False)
        # Serialize the settings with indentation and preserve Unicode characters.
    print(json.dumps(settings, indent=2))
    # Print the settings JSON to the terminal.


if __name__ == "__main__":
    # Execute the training workflow only when this file is run directly.
    main()
    # Call the main function.