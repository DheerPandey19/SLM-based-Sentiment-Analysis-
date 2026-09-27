"""Prompts for the teacher model."""

from __future__ import annotations

from schema import MOVIE_ASPECTS


SYSTEM_PROMPT = f"""You label movie reviews for sentiment and aspects.
Return ONLY a JSON object with this shape:
{{
  "overall": "positive" | "negative" | "mixed",
  "aspects": {{ "<aspect>": "positive" | "negative" | "mixed", ... }}
}}
Allowed aspects: {", ".join(sorted(MOVIE_ASPECTS))}
Allowed sentiments: positive, negative, mixed
Rules:
- Only include aspects the review actually discusses.
- Do not invent aspects or add extra keys.
- overall should reflect the review as a whole.
- If no aspects are clearly discussed, use an empty aspects object.
"""
FEW_SHOT_USER = """Review:
The acting was fantastic and the lead carried every scene, but the pacing dragged badly in the middle and the ending felt rushed."""
FEW_SHOT_ASSISTANT = """{
  "overall": "mixed",
  "aspects": {
    "acting": "positive",
    "pacing": "negative"
  }
}"""


def build_messages(review_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {"role": "user", "content": f"Review:\n{review_text}"},
    ]
