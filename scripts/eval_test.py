"""Evaluate models on the held-out test split.

Usage:
  python scripts/eval_test.py --backend gguf --model qwen-aspect-Q4_K_M.gguf --limit 25
  python scripts/eval_test.py --backend gguf --model qwen2.5-1.5b-instruct-q4_k_m.gguf --limit 25
  python scripts/eval_test.py --backend gguf --model qwen-aspect-Q4_K_M.gguf --prompt weak --limit 25
  python scripts/eval_test.py --backend gguf --model qwen-aspect-Q4_K_M.gguf --filter-aspects --limit 25
  python scripts/eval_test.py --backend openai --model gpt-4o-mini --limit 25
  python scripts/eval_test.py --backend gguf --model qwen-aspect-Q4_K_M.gguf --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.aspect_filter import filter_aspects
from src.gguf_infer import DEFAULT_LLAMA_CLI, DEFAULT_TOKENIZER_ID, run_gguf
from src.metrics import aggregate, score_example
from src.schema import validate_label
from src.teacher import label_review


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("gguf", "openai"), required=True)
    p.add_argument(
        "--model",
        required=True,
        help="GGUF path (gguf) or OpenAI model name (openai)",
    )
    p.add_argument("--test", type=Path, default=ROOT / "data" / "splits" / "test.jsonl")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "eval")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--llama-cli", type=Path, default=DEFAULT_LLAMA_CLI)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_ID)
    p.add_argument("--n-predict", type=int, default=128)
    p.add_argument(
        "--prompt",
        choices=("full", "weak"),
        default="full",
        help="Prompt style: full (few-shot) or weak (short system, no few-shot)",
    )
    p.add_argument(
        "--filter-aspects",
        action="store_true",
        help="Drop predicted aspects with no keyword evidence in the review",
    )
    return p.parse_args()


def pred_path(
    out_dir: Path,
    backend: str,
    model: str,
    prompt: str = "full",
    filter_aspects_flag: bool = False,
) -> Path:
    stem = Path(model).stem if backend == "gguf" else model.replace("/", "_")
    parts = [backend, stem]
    if prompt != "full":
        parts.append(prompt)
    if filter_aspects_flag:
        parts.append("filtered")
    return out_dir / ("_".join(parts) + ".jsonl")


def load_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done[row["id"]] = row
    return done


def gold_label(row: dict) -> dict:
    return validate_label({"overall": row["overall"], "aspects": row.get("aspects") or {}})


def main() -> None:
    args = parse_args()
    load_dotenv()

    rows = [
        json.loads(line)
        for line in args.test.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = pred_path(
        args.out_dir,
        args.backend,
        args.model,
        args.prompt,
        args.filter_aspects,
    )
    done = load_done(out) if args.resume else {}

    client: OpenAI | None = None
    if args.backend == "openai":
        client = OpenAI()

    mode = "a" if args.resume and out.exists() else "w"
    score_rows: list[dict] = []
    desc = f"eval:{args.backend}:{args.prompt}"
    if args.filter_aspects:
        desc += ":filtered"

    with out.open(mode, encoding="utf-8") as fout:
        for row in tqdm(rows, desc=desc):
            rid = row["id"]
            gold = gold_label(row)

            if rid in done:
                saved = done[rid]
                pred = saved.get("pred")
                latency = saved.get("latency_s")
                ok = saved.get("valid", pred is not None)
                scored = score_example(gold, pred if ok else None)
                scored["latency_s"] = latency
                score_rows.append(scored)
                continue

            pred = None
            latency = None
            error = None
            try:
                if args.backend == "gguf":
                    pred, latency = run_gguf(
                        row["text"],
                        model_path=Path(args.model),
                        llama_cli=args.llama_cli,
                        tokenizer_id=args.tokenizer,
                        n_predict=args.n_predict,
                        prompt_style=args.prompt,
                        filter_aspects_flag=args.filter_aspects,
                    )
                else:
                    assert client is not None
                    t0 = time.perf_counter()
                    labeled = label_review(
                        client, row["text"], model=args.model, prompt_style=args.prompt
                    )
                    latency = time.perf_counter() - t0
                    pred = {
                        "overall": labeled["overall"],
                        "aspects": labeled["aspects"],
                    }
                    if args.filter_aspects:
                        pred = filter_aspects(pred, row["text"])
            except Exception as e:
                error = str(e)

            scored = score_example(gold, pred)
            scored["latency_s"] = latency
            score_rows.append(scored)

            record = {
                "id": rid,
                "gold": gold,
                "pred": pred,
                "valid": pred is not None,
                "latency_s": latency,
                "error": error,
                "backend": args.backend,
                "model": str(args.model),
                "prompt": args.prompt,
                "filter_aspects": args.filter_aspects,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    summary = aggregate(score_rows)
    summary["backend"] = args.backend
    summary["model"] = str(args.model)
    summary["prompt"] = args.prompt
    summary["filter_aspects"] = args.filter_aspects
    summary["predictions"] = str(out)

    summary_path = args.out_dir / "summary.json"
    summaries: list[dict] = []
    if summary_path.exists():
        try:
            summaries = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(summaries, list):
                summaries = [summaries]
        except json.JSONDecodeError:
            summaries = []

    def _same_run(s: dict) -> bool:
        return (
            s.get("backend") == args.backend
            and s.get("model") == str(args.model)
            and s.get("prompt", "full") == args.prompt
            and bool(s.get("filter_aspects", False)) == args.filter_aspects
        )

    summaries = [s for s in summaries if not _same_run(s)]
    summaries.append(summary)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote predictions -> {out}")
    print(f"wrote summary    -> {summary_path}")


if __name__ == "__main__":
    main()
