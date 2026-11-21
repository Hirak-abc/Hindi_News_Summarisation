import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "03_Source_Code" / "llm"))

from prompts import EXPERIMENTAL_PROMPTS


def test_all_prompts_include_article_placeholder():
    # Each selectable prompt must have a slot for its source article.
    for prompt in EXPERIMENTAL_PROMPTS.values():
        assert "{article}" in prompt


def test_default_prompt_requests_hindi_summary():
    # The baseline prompt must explicitly ask for a Hindi summary.
    assert "हिंदी सारांश" in EXPERIMENTAL_PROMPTS["baseline"]
