# Generate LLM summaries for the deterministic final Hindi sample.

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd

from prompts import EXPERIMENTAL_PROMPTS, SYSTEM_PROMPT


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_SET = ROOT / "01_Dataset" / "processed" / "common_test_set.csv"
DEFAULT_SAMPLE = ROOT / "01_Dataset" / "processed" / "llm_final_sample_ids.csv"
DEFAULT_OUTPUT = ROOT / "05_Predictions" / "llm_predictions.csv"
DEFAULT_LOG = ROOT / "10_Documentation" / "llm_generation_log.json"


class GeminiAPIError(RuntimeError):
    # Carry the HTTP status and retry hint returned by the Gemini API.

    def __init__(self, status: int, message: str, retry_after: float | None = None):
        # Store retry metadata while preserving the readable API error message.
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.retry_after = retry_after


def list_available_models(api_key: str) -> list[str]:
    # Return models from this API key that support generateContent requests.
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models"
        f"?key={api_key}"
    )
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    models = []
    for item in result.get("models", []):
        name = item.get("name", "").removeprefix("models/")
        methods = item.get("supportedGenerationMethods", [])
        if name and "generateContent" in methods:
            models.append(name)
    return models


def call_gemini(api_key: str, model: str, article: str, prompt_template: str) -> str:
    # Request one Hindi summary and surface API errors with useful response details.
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        ":generateContent"
        f"?key={api_key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt_template.format(article=article)}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 180},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Preserve the provider's error body so quota and model errors are diagnosable.
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        retry_match = re.search(r"retry in ([\d.]+)s", body, re.IGNORECASE)
        retry_after = float(retry_match.group(1)) if retry_match else None
        raise GeminiAPIError(exc.code, body, retry_after) from exc

    candidates = result.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"No candidate returned: {result}")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts).strip()


def main() -> None:
    # Generate the selected sample, resume saved successes, and log any failures.
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", choices=EXPERIMENTAL_PROMPTS, default="faithful")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sleep",
        type=float,
        default=5.0,
        help="Delay between requests; keep at least 4 seconds for a 15 RPM quota.",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running LLM generation.")

    if args.list_models:
        available = list_available_models(api_key)
        print("Available generateContent models:")
        for model in available:
            print(model)
        return

    test = pd.read_csv(DEFAULT_TEST_SET, encoding="utf-8")
    sample = pd.read_csv(DEFAULT_SAMPLE, encoding="utf-8")
    output = test.merge(sample, on="article_id", how="inner", validate="many_to_one")
    if args.limit:
        output = output.head(args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Reuse completed rows from a prior run so retries do not spend requests twice.
    existing = {}
    if output_path.exists():
        old = pd.read_csv(output_path, encoding="utf-8-sig")
        previous_log = {}
        if DEFAULT_LOG.exists():
            previous_log = json.loads(DEFAULT_LOG.read_text(encoding="utf-8"))
        if (
            previous_log.get("model") != args.model
            or previous_log.get("prompt") != args.prompt
        ):
            raise SystemExit(
                "Existing predictions were generated with a different model or prompt. "
                "Choose a different --output path or move the old file before continuing."
            )
        existing = {
            str(row["article_id"]): str(row["prediction"])
            for _, row in old.iterrows()
            if str(row.get("prediction", "")).strip()
        }

    completed = []
    failures = []
    for _, row in output.iterrows():
        article_id = str(row["article_id"])
        if article_id in existing:
            completed.append({
                "article_id": article_id,
                "reference": row["reference"],
                "prediction": existing[article_id],
            })
            continue

        prediction = ""
        # Retry transient throttling/server errors with provider hints or backoff.
        for attempt in range(args.max_retries + 1):
            try:
                prediction = call_gemini(
                    api_key,
                    args.model,
                    str(row["article"]),
                    EXPERIMENTAL_PROMPTS[args.prompt],
                )
                if not prediction:
                    raise RuntimeError("The API returned an empty summary.")
                break
            except GeminiAPIError as exc:
                if exc.status not in {429, 500, 502, 503, 504} or attempt == args.max_retries:
                    failures.append({"article_id": article_id, "error": str(exc)})
                    break
                delay = exc.retry_after or min(5 * (2 ** attempt), 90)
                print(
                    f"Temporary API error for {article_id} "
                    f"(HTTP {exc.status}); retrying in {delay:.1f}s"
                )
                time.sleep(delay)
            except Exception as exc:
                failures.append({"article_id": article_id, "error": str(exc)})
                break

        if prediction:
            completed.append({
                "article_id": article_id,
                "reference": row["reference"],
                "prediction": prediction,
            })
            existing[article_id] = prediction
            pd.DataFrame(completed).to_csv(
                output_path,
                index=False,
                encoding="utf-8-sig",
            )
        if len(existing) % 10 == 0:
            print(f"Completed {len(existing)}/{len(output)}")
        time.sleep(args.sleep)

    result = pd.DataFrame(completed)
    log = {
        "model": args.model,
        "prompt": args.prompt,
        "temperature": 0,
        "max_output_tokens": 180,
        "sample_size": len(output),
        "successful": len(existing),
        "failures": failures,
    }
    DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    if failures:
        print(f"Partial results saved: {len(existing)}/{len(output)} summaries.")
    else:
        result.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(result)} summaries to {output_path}")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
