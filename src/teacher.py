"""Call OpenAI to produce validated aspect labels."""

from __future__ import annotations

import json

from openai import OpenAI

from .prompt import build_messages
from .schema import validate_label


def label_review(
    client: OpenAI,
    text: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 2,
) -> dict:
    """Return {overall, aspects, teacher_model} for one review."""
    last_error: Exception | None = None

    for _ in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=build_messages(text),
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = resp.choices[0].message.content or ""
            parsed = json.loads(content)
            cleaned = validate_label(parsed)
            return {**cleaned, "teacher_model": model}
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = e

    raise RuntimeError(f"failed to label after {max_retries + 1} attempts: {last_error}")
