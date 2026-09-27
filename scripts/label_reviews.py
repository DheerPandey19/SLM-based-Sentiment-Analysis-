"""Label raw reviews with a teacher model (GPT).

Usage:
  python scripts/label_reviews.py --limit 5
  python scripts/label_reviews.py --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Allow `from src...` when running as a script from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.teacher import label_review


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", type=Path, default=Path("data/raw/reviews.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/labeled/reviews.jsonl"))
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--limit", type=int, default=None, help="Label at most N reviews")
    p.add_argument("--resume", action="store_true", help="Skip ids already in --out")
    return p.parse_args()


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "overall" in row:
                done.add(row["id"])
    return done


def main() -> None:
    args = parse_args()
    load_dotenv()
    client = OpenAI()  # reads OPENAI_API_KEY from env

    done = load_done_ids(args.out) if args.resume else set()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    labeled = 0
    skipped = 0
    failed = 0

    with args.inp.open(encoding="utf-8") as fin, args.out.open(
        "a" if args.resume else "w", encoding="utf-8"
    ) as fout:
        rows = [json.loads(line) for line in fin if line.strip()]
        if args.limit is not None:
            rows = rows[: args.limit]

        for row in tqdm(rows, desc="labeling"):
            if row["id"] in done:
                skipped += 1
                continue
            try:
                teacher = label_review(client, row["text"], model=args.model)
                out = {**row, **teacher}
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                fout.flush()
                labeled += 1
            except Exception as e:
                failed += 1
                print(f"FAIL {row['id']}: {e}", file=sys.stderr)

    print(f"labeled={labeled}  skipped={skipped}  failed={failed}  -> {args.out}")


if __name__ == "__main__":
    main()