"""Download a balanced sample of IMDb movie reviews into JSONL.

Usage:
  python scripts/download_reviews.py
  python scripts/download_reviews.py --n 2500 --seed 42 --out data/raw/reviews.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        default=2500,
        help="Total reviews to keep (half positive, half negative).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/reviews.jsonl"),
    )
    return parser.parse_args()


def sample_balanced(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n % 2 != 0:
        raise ValueError(f"--n must be even so pos/neg stays balanced; got {n}")

    per_class = n // 2
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]

    if len(pos) < per_class or len(neg) < per_class:
        raise RuntimeError(
            f"Need {per_class} per class, but found {len(pos)} pos / {len(neg)} neg"
        )

    rng = random.Random(seed)
    chosen = rng.sample(pos, per_class) + rng.sample(neg, per_class)
    rng.shuffle(chosen)
    return chosen


def normalize(row: dict, index: int) -> dict:
    text = (row.get("review") or row.get("text") or "").strip()
    return {
        "id": f"imdb_{index:04d}",
        "source": "imdb",
        "domain": "movie",
        "text": text,
        "label": int(row["label"]),
    }


def main() -> None:
    args = parse_args()

    print("Loading ajaykarthick/imdb-movie-reviews ...")
    ds = load_dataset("ajaykarthick/imdb-movie-reviews")
    # Prefer train if present; otherwise use the only/default split.
    split_name = "train" if "train" in ds else next(iter(ds.keys()))
    rows = list(ds[split_name])
    print(f"Loaded {len(rows)} rows from split '{split_name}'")

    sampled = sample_balanced(rows, args.n, args.seed)
    records = [normalize(row, i) for i, row in enumerate(sampled)]
    records = [r for r in records if r["text"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_pos = sum(1 for r in records if r["label"] == 1)
    n_neg = sum(1 for r in records if r["label"] == 0)
    avg_len = sum(len(r["text"]) for r in records) / max(len(records), 1)

    print(f"Wrote {len(records)} reviews -> {args.out}")
    print(f"  pos={n_pos}  neg={n_neg}  avg_chars={avg_len:.0f}")


if __name__ == "__main__":
    main()
