"""Prompts for the teacher model."""

from __future__ import annotations

from .schema import MOVIE_ASPECTS


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
- Prefer fewer aspects. Empty aspects {{}} is correct when nothing specific is discussed.
- Do not dump the full aspect list. Two or three tags is typical; seven is almost never right.
- Use ONLY the allowed aspect names above. Never invent keys
  like action, animation, humor, characterisation, etc.
  Map those ideas to the closest allowed aspect (e.g. animation→visuals,
  characterisation→acting, humor→dialogue, music→soundtrack).
- Do not add extra keys.
- overall should reflect the review as a whole.
- If no aspects are clearly discussed, use an empty aspects object.
"""

# Short system for prompt ablation: schema only, no few-shot scaffolding.
WEAK_SYSTEM_PROMPT = f"""Return ONLY JSON:
{{"overall": "positive"|"negative"|"mixed", "aspects": {{"<aspect>": "positive"|"negative"|"mixed"}}}}
Allowed aspects: {", ".join(sorted(MOVIE_ASPECTS))}
Only tag aspects the review clearly discusses; otherwise use {{}}.
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

# Second shot: sparse / empty aspects — teach when not to tag.
FEW_SHOT_SPARSE_USER = """Review:
Just an okay movie. Nothing special, nothing terrible. I watched it once and moved on."""
FEW_SHOT_SPARSE_ASSISTANT = """{
  "overall": "mixed",
  "aspects": {}
}"""


def build_messages(
    review_text: str, style: str = "full"
) -> list[dict[str, str]]:
    """Build chat messages for labeling.

    style:
      full — system rules + few-shot (mixed + empty) + review
      weak — short system only + review (no few-shot; for ablation)
    """
    if style == "weak":
        return [
            {"role": "system", "content": WEAK_SYSTEM_PROMPT},
            {"role": "user", "content": f"Review:\n{review_text}"},
        ]
    if style != "full":
        raise ValueError(f"unknown prompt style: {style!r} (expected 'full' or 'weak')")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {"role": "user", "content": FEW_SHOT_SPARSE_USER},
        {"role": "assistant", "content": FEW_SHOT_SPARSE_ASSISTANT},
        {"role": "user", "content": f"Review:\n{review_text}"},
    ]
