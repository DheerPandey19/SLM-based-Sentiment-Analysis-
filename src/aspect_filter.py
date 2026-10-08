"""Drop predicted aspects that lack clear evidence in the review text."""

from __future__ import annotations

import re

from .schema import MOVIE_ASPECTS

# Cue phrases / stems per aspect. Match is case-insensitive substring or word-ish.
ASPECT_CUES: dict[str, tuple[str, ...]] = {
    "acting": (
        "acting",
        "actor",
        "actress",
        "actors",
        "actresses",
        "performance",
        "performances",
        "cast",
        "casting",
        "portrayal",
        "characterisation",
        "characterization",
        "characters",
        "lead",
        "star",
        "stars",
    ),
    "plot": (
        "plot",
        "story",
        "storyline",
        "narrative",
        "premise",
        "twist",
        "ending",
        "climax",
        "script's story",
    ),
    "visuals": (
        "visual",
        "visuals",
        "cinematography",
        "cinematograph",
        "camera",
        "shot",
        "shots",
        "imagery",
        "effects",
        "cgi",
        "animation",
        "special effects",
        "spectacle",
        "looks beautiful",
        "looks stunning",
    ),
    "pacing": (
        "pacing",
        "pace",
        "paced",
        "slow",
        "dragged",
        "dragging",
        "rushed",
        "runtime",
        "too long",
        "too short",
        "editing",
        "tedious",
        "boring middle",
    ),
    "dialogue": (
        "dialogue",
        "dialog",
        "script",
        "writing",
        "lines",
        "witty",
        "humor",
        "humour",
        "comedy",
        "jokes",
        "banter",
        "conversation",
    ),
    "soundtrack": (
        "soundtrack",
        "score",
        "music",
        "musical",
        "song",
        "songs",
        "sound design",
        "audio",
        "orchestra",
        "theme song",
    ),
    "direction": (
        "direction",
        "directing",
        "director",
        "helmed",
        "filmmaking",
        "filmmaker",
        "directorial",
    ),
}


def _review_has_cue(review_lower: str, cues: tuple[str, ...]) -> bool:
    for cue in cues:
        if " " in cue:
            if cue in review_lower:
                return True
        else:
            # Word-boundary-ish so "cast" doesn't need to match "broadcast"
            if re.search(rf"\b{re.escape(cue)}\b", review_lower):
                return True
    return False


def filter_aspects(label: dict, review_text: str) -> dict:
    """Return a copy of label keeping only aspects with evidence in the review.

    overall is unchanged. Unknown aspect keys are dropped.
    """
    aspects = label.get("aspects") or {}
    text = review_text.lower()
    kept: dict[str, str] = {}
    for aspect, sentiment in aspects.items():
        if aspect not in MOVIE_ASPECTS:
            continue
        cues = ASPECT_CUES.get(aspect, (aspect,))
        if _review_has_cue(text, cues):
            kept[aspect] = sentiment
    return {"overall": label["overall"], "aspects": kept}
