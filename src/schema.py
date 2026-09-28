"""Allowed label schema for movie review aspect tagging."""

from __future__ import annotations

MOVIE_ASPECTS = frozenset(
    {
        "plot",
        "acting",
        "visuals",
        "pacing",
        "dialogue",
        "soundtrack",
        "direction",
    }
)

SENTIMENTS = frozenset({"positive", "negative", "mixed"})

# Common teacher invents → roadmap aspects. Unmapped keys are dropped.
ASPECT_ALIASES: dict[str, str] = {
    "cinematography": "visuals",
    "animation": "visuals",
    "effects": "visuals",
    "cgi": "visuals",
    "characterisation": "acting",
    "characterization": "acting",
    "characters": "acting",
    "performance": "acting",
    "performances": "acting",
    "casting": "acting",
    "music": "soundtrack",
    "score": "soundtrack",
    "sound": "soundtrack",
    "story": "plot",
    "storyline": "plot",
    "script": "dialogue",
    "writing": "dialogue",
    "humor": "dialogue",
    "humour": "dialogue",
    "comedy": "dialogue",
    "editing": "pacing",
    "directing": "direction",
}


def validate_label(obj: object) -> dict:
    """Validate and return a cleaned {overall, aspects} dict.

    Unknown aspects are mapped via ASPECT_ALIASES or dropped.
    Raises ValueError only if overall/aspects shape is invalid.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"label must be a dict, got {type(obj).__name__}")

    overall = obj.get("overall")
    if overall not in SENTIMENTS:
        raise ValueError(f"invalid overall sentiment: {overall!r}")

    aspects = obj.get("aspects")
    if not isinstance(aspects, dict):
        raise ValueError("aspects must be a dict")

    cleaned: dict[str, str] = {}
    for key, value in aspects.items():
        aspect = key if key in MOVIE_ASPECTS else ASPECT_ALIASES.get(key)
        if aspect is None:
            continue  # drop invents like "action"
        if value not in SENTIMENTS:
            raise ValueError(f"invalid sentiment for {key}: {value!r}")
        # first mention wins if alias collapses onto an existing key
        cleaned.setdefault(aspect, value)

    return {"overall": overall, "aspects": cleaned}
