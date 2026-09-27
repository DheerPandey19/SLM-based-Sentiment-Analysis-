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
        "direction"
    }
)

SENTIMENTS = frozenset(
    {
        "positive", 
        "negative", 
        "mixed"
    }
)

def validate_label(obj : object)->dict:

    """Validate and return a cleaned {overall, aspects} dict.
    Raises ValueError if the shape or values are invalid.
    """

    if not isinstance(obj,dict):
         raise ValueError(f"label must be a dict, got {type(obj).__name__}")

    overall=obj.get("overall")
    if overall not in SENTIMENTS:
                raise ValueError(f"invalid overall sentiment: {overall!r}")

    aspects = obj.get("aspects")
    if not isinstance(aspects, dict):
        raise ValueError("aspects must be a dict")

    cleaned: dict[str, str] = {}
    for key, value in aspects.items():
        if key not in MOVIE_ASPECTS:
            raise ValueError(f"unknown aspect: {key!r}")
        if value not in SENTIMENTS:
            raise ValueError(f"invalid sentiment for {key}: {value!r}")
        cleaned[key] = value
    return {"overall": overall, "aspects": cleaned}



