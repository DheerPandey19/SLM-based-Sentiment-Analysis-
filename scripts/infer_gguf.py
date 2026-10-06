"""Run the local Q4_K_M GGUF with the same prompt recipe as training/Colab.

Usage:
  python scripts/infer_gguf.py --review "Great performances all round..."
  python scripts/infer_gguf.py --review-file review.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gguf_infer import DEFAULT_LLAMA_CLI, DEFAULT_TOKENIZER_ID, run_gguf

DEFAULT_MODEL = ROOT / "qwen-aspect-Q4_K_M.gguf"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--review", help="Review text to label")
    g.add_argument("--review-file", type=Path, help="File containing review text")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--llama-cli", type=Path, default=DEFAULT_LLAMA_CLI)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_ID)
    p.add_argument("--n-predict", type=int, default=128)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument(
        "--prompt",
        choices=("full", "weak"),
        default="full",
        help="Prompt style: full (few-shot) or weak (no few-shot)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    review = (
        args.review
        if args.review is not None
        else args.review_file.read_text(encoding="utf-8")
    )
    try:
        cleaned, latency = run_gguf(
            review,
            model_path=args.model,
            llama_cli=args.llama_cli,
            tokenizer_id=args.tokenizer,
            n_predict=args.n_predict,
            temp=args.temp,
            prompt_style=args.prompt,
        )
    except Exception as e:
        raise SystemExit(str(e)) from e

    print(json.dumps(cleaned, ensure_ascii=False, indent=2))
    print(f"# latency_s={latency:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
