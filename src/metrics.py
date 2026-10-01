"""Eval metrics for aspect-sentiment labels."""

from __future__ import annotations

from .schema import MOVIE_ASPECTS


def aspect_pair_set(aspects: dict[str, str]) -> set[tuple[str, str]]:
    """{(aspect, sentiment), ...} for aspects present in the label."""
    return {(a, s) for a, s in aspects.items() if a in MOVIE_ASPECTS}


def score_example(gold: dict, pred: dict | None) -> dict:
    """Score one example. pred=None means invalid / failed prediction."""
    if pred is None:
        return {
            "valid": False,
            "overall_correct": False,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

    gold_pairs = aspect_pair_set(gold.get("aspects") or {})
    pred_pairs = aspect_pair_set(pred.get("aspects") or {})
    tp = len(gold_pairs & pred_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)

    return {
        "valid": True,
        "overall_correct": gold.get("overall") == pred.get("overall"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def aggregate(rows: list[dict]) -> dict:
    """Aggregate per-example score dicts into summary metrics."""
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "n_valid": 0,
            "json_validity": 0.0,
            "overall_accuracy": 0.0,
            "aspect_precision": 0.0,
            "aspect_recall": 0.0,
            "aspect_f1": 0.0,
            "mean_latency_s": None,
        }

    n_valid = sum(1 for r in rows if r["valid"])
    n_overall_ok = sum(1 for r in rows if r["valid"] and r["overall_correct"])
    tp = sum(r["tp"] for r in rows if r["valid"])
    fp = sum(r["fp"] for r in rows if r["valid"])
    fn = sum(r["fn"] for r in rows if r["valid"])

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )

    latencies = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    mean_latency = sum(latencies) / len(latencies) if latencies else None

    return {
        "n": n,
        "n_valid": n_valid,
        "json_validity": n_valid / n,
        "overall_accuracy": n_overall_ok / n_valid if n_valid else 0.0,
        "aspect_precision": precision,
        "aspect_recall": recall,
        "aspect_f1": f1,
        "mean_latency_s": mean_latency,
    }
