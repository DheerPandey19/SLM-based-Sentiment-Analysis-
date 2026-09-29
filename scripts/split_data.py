"""Split labeled reviews into train/val/test JSONL (80/10/10).

Usage:
  python scripts/split_data.py
  python scripts/split_data.py --seed 42 --train 0.8 --val 0.1 --test 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", type=Path, default=Path("data/labeled/reviews.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/splits"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_rows(
    rows: list[dict],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.inp)
    if not rows:
        raise SystemExit(f"no rows in {args.inp}")

    train, val, test = split_rows(rows, args.train, args.val, args.test, args.seed)

    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "val.jsonl", val)
    write_jsonl(args.out_dir / "test.jsonl", test)

    print(f"total={len(rows)}")
    print(f"  train={len(train)} -> {args.out_dir / 'train.jsonl'}")
    print(f"  val={len(val)}   -> {args.out_dir / 'val.jsonl'}")
    print(f"  test={len(test)}  -> {args.out_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
