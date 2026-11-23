# Generate LLM summaries for the deterministic final Hindi sample.
# This comment identifies the script's purpose.

from __future__ import annotations
# Allow modern type annotations without eager evaluation.

import argparse
# Import argparse to read command-line generation options.
import json
# Import json for API payloads and generation log data.
import os
# Import os to read the Gemini API key from environment variables.
import re
# Import re to extract retry delays from API error messages.
import time
# Import time to pause between requests and retry attempts.
import urllib.request
# Import urllib.request to send HTTP requests to Gemini.
import urllib.error
# Import urllib.error to handle HTTP API failures.
from pathlib import Path
# Import Path to build and create input and output locations.

import pandas as pd
# Import pandas to merge test rows with the final sample IDs.

from prompts import EXPERIMENTAL_PROMPTS, SYSTEM_PROMPT
# Import prompt choices and shared system instructions from prompts.py.


ROOT = Path(__file__).resolve().parents[2]
# Resolve the repository root from this module's location.
DEFAULT_TEST_SET = ROOT / "01_Dataset" / "processed" / "common_test_set.csv"
# Set the default source test CSV path.
DEFAULT_SAMPLE = ROOT / "01_Dataset" / "processed" / "llm_final_sample_ids.csv"
# Set the default CSV containing selected article IDs.
DEFAULT_OUTPUT = ROOT / "05_Predictions" / "llm_predictions.csv"
# Set the default destination for LLM summaries.
DEFAULT_LOG = ROOT / "10_Documentation" / "llm_generation_log.json"
# Set the default JSON log destination.


class GeminiAPIError(RuntimeError):
    # Carry the HTTP status and retry hint returned by the Gemini API.
    # Define a custom exception for Gemini HTTP failures.

    def __init__(self, status: int, message: str, retry_after: float | None = None):
        # Store retry metadata while preserving the readable API error message.
        super().__init__(f"HTTP {status}: {message}")
        # Initialize RuntimeError with a status and response message.
        self.status = status
        # Store the HTTP status code.
        self.retry_after = retry_after
        # Store an optional provider-requested retry delay.


def list_available_models(api_key: str) -> list[str]:
    # Return models from this API key that support generateContent requests.
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models"
        f"?key={api_key}"
    )
    # Build the Gemini model-list endpoint and include the API key.
    request = urllib.request.Request(url)
    # Create an HTTP GET request for the model list.
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))
        # Download and decode the JSON model-list response.

    models = []
    # Initialize the list of usable model names.
    for item in result.get("models", []):
        name = item.get("name", "").removeprefix("models/")
        # Remove the API path prefix from the model identifier.
        methods = item.get("supportedGenerationMethods", [])
        # Read methods supported by this model.
        if name and "generateContent" in methods:
            models.append(name)
            # Include only named models that support text generation.
    return models
    # Return the list of available generation models.


def call_gemini(api_key: str, model: str, article: str, prompt_template: str) -> str:
    # Request one Hindi summary and surface API errors with useful response details.
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        ":generateContent"
        f"?key={api_key}"
    )
    # Build the generateContent endpoint for the selected model.
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt_template.format(article=article)}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 180},
    }
    # Build system instructions, the formatted article prompt, and deterministic generation settings.
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Create a JSON POST request for Gemini.
    # Preserve the provider's error body so quota and model errors are diagnosable.
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
            # Send the request and decode the successful JSON response.
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        # Read the provider error body with replacement for invalid text.
        retry_match = re.search(r"retry in ([\d.]+)s", body, re.IGNORECASE)
        # Search for the provider's suggested retry delay.
        retry_after = float(retry_match.group(1)) if retry_match else None
        # Convert the matched delay to a float when present.
        raise GeminiAPIError(exc.code, body, retry_after) from exc
        # Raise the custom API exception while retaining its cause.

    candidates = result.get("candidates", [])
    # Extract generated candidates from the response.
    if not candidates:
        raise RuntimeError(f"No candidate returned: {result}")
        # Reject a response that has no generated candidate.
    parts = candidates[0].get("content", {}).get("parts", [])
    # Extract text parts from the first candidate.
    return "".join(part.get("text", "") for part in parts).strip()
    # Concatenate text parts and trim surrounding whitespace.


def main() -> None:
    # Generate the selected sample, resume saved successes, and log any failures.
    parser = argparse.ArgumentParser()
    # Create the command-line argument parser.
    parser.add_argument("--prompt", choices=EXPERIMENTAL_PROMPTS, default="faithful")
    # Select the prompt experiment name.
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    # Select the Gemini model identifier.
    parser.add_argument("--list-models", action="store_true")
    # Add an option that only lists models supporting content generation.
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    # Set the prediction CSV destination.
    parser.add_argument("--limit", type=int, default=None)
    # Add an optional limit for quick-generation runs.
    parser.add_argument(
        "--sleep",
        type=float,
        default=5.0,
        help="Delay between requests; keep at least 4 seconds for a 15 RPM quota.",
    )
    # Configure delay between successful or failed requests.
    parser.add_argument("--max-retries", type=int, default=5)
    # Configure maximum additional retry attempts for transient API errors.
    args = parser.parse_args()
    # Parse command-line arguments into args.

    api_key = os.getenv("GEMINI_API_KEY")
    # Read the Gemini API key from the environment.
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running LLM generation.")
        # Stop safely when credentials are missing.

    if args.list_models:
        available = list_available_models(api_key)
        # Request the models available to this API key.
        print("Available generateContent models:")
        # Print the model-list heading.
        for model in available:
            print(model)
            # Print each available generation model.
        return
        # Exit after model discovery.

    test = pd.read_csv(DEFAULT_TEST_SET, encoding="utf-8")
    # Load the shared test CSV.
    sample = pd.read_csv(DEFAULT_SAMPLE, encoding="utf-8")
    # Load the deterministic sample article-ID CSV.
    output = test.merge(sample, on="article_id", how="inner", validate="many_to_one")
    # Inner-join test articles with sample IDs and validate join cardinality.
    if args.limit:
        output = output.head(args.limit)
        # Keep only the first requested sample rows.

    output_path = Path(args.output)
    # Convert the output destination to a Path.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Create the prediction output directory.
    # Reuse completed rows from a prior run so retries do not spend requests twice.
    existing = {}
    # Initialize a mapping of already-generated article IDs.
    if output_path.exists():
        old = pd.read_csv(output_path, encoding="utf-8-sig")
        # Load any prior prediction CSV.
        previous_log = {}
        # Initialize the prior generation log.
        if DEFAULT_LOG.exists():
            previous_log = json.loads(DEFAULT_LOG.read_text(encoding="utf-8"))
            # Read the prior model/prompt metadata when it exists.
        if (
            previous_log.get("model") != args.model
            or previous_log.get("prompt") != args.prompt
        ):
            # Reject reuse when model or prompt settings differ.
            raise SystemExit(
                "Existing predictions were generated with a different model or prompt. "
                "Choose a different --output path or move the old file before continuing."
            )
            # Explain how to avoid mixing incompatible predictions.
        existing = {
            str(row["article_id"]): str(row["prediction"])
            for _, row in old.iterrows()
            if str(row.get("prediction", "")).strip()
        }
        # Index prior nonempty predictions by article ID.

    completed = []
    # Initialize the list of successful predictions for this run.
    failures = []
    # Initialize the list of API or processing failures.
    for _, row in output.iterrows():
        article_id = str(row["article_id"])
        # Convert the article ID to a string key.
        if article_id in existing:
            completed.append({
                "article_id": article_id,
                "reference": row["reference"],
                "prediction": existing[article_id],
            })
            continue
            # Reuse an existing prediction and skip another API call.

        prediction = ""
        # Initialize this article's prediction text.
        # Retry transient throttling/server errors with provider hints or backoff.
        for attempt in range(args.max_retries + 1):
            try:
                prediction = call_gemini(
                    api_key,
                    args.model,
                    str(row["article"]),
                    EXPERIMENTAL_PROMPTS[args.prompt],
                )
                # Call Gemini with the selected model and prompt.
                if not prediction:
                    raise RuntimeError("The API returned an empty summary.")
                    # Treat an empty candidate as an error.
                break
                # Stop retrying after a successful nonempty response.
            except GeminiAPIError as exc:
                if exc.status not in {429, 500, 502, 503, 504} or attempt == args.max_retries:
                    failures.append({"article_id": article_id, "error": str(exc)})
                    break
                    # Record permanent or retry-exhausted Gemini API errors.
                delay = exc.retry_after or min(5 * (2 ** attempt), 90)
                # Use provider delay or exponential capped backoff.
                print(
                    f"Temporary API error for {article_id} "
                    f"(HTTP {exc.status}); retrying in {delay:.1f}s"
                )
                # Report a transient API error and planned retry.
                time.sleep(delay)
                # Wait before the next retry attempt.
            except Exception as exc:
                failures.append({"article_id": article_id, "error": str(exc)})
                break
                # Record unexpected local/request errors without retrying.

        if prediction:
            completed.append({
                "article_id": article_id,
                "reference": row["reference"],
                "prediction": prediction,
            })
            # Append the successful summary to completed results.
            existing[article_id] = prediction
            # Cache it for resume behavior.
            pd.DataFrame(completed).to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig",
            )
            # Incrementally save completed predictions after every success.
        if len(existing) % 10 == 0:
            print(f"Completed {len(existing)}/{len(output)}")
            # Print progress whenever the completed count reaches a multiple of ten.
        time.sleep(args.sleep)
        # Pause between article requests to respect the request quota.

    result = pd.DataFrame(completed)
    # Convert completed rows to a DataFrame.
    log = {
        "model": args.model,
        "prompt": args.prompt,
        "temperature": 0,
        "max_output_tokens": 180,
        "sample_size": len(output),
        "successful": len(existing),
        "failures": failures,
    }
    # Build generation metadata and failure details for the JSON log.
    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    # Create the log directory.
    DEFAULT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    # Write the JSON log with Unicode preserved and indentation.

    if failures:
        print(f"Partial results saved: {len(existing)}/{len(output)} summaries.")
        # Report partial completion when failures occurred.
    else:
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(result)} summaries to {output_path}")
        # Write the full result only when every selected article succeeded.
    print(f"Failures: {len(failures)}")
    # Print the final failure count.


if __name__ == "__main__":
    main()
    # Execute generation only when this module is run directly.
