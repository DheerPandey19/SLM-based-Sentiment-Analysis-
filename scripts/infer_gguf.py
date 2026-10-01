"""Run the local Q4_K_M GGUF with the same prompt recipe as training/Colab.
Usage:
  python scripts/infer_gguf.py --review "Great performances all round..."
  python scripts/infer_gguf.py --review-file review.txt
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.prompt import build_messages
from src.schema import validate_label

DEFAULT_MODEL = ROOT / "qwen-aspect-Q4_K_M.gguf"
DEFAULT_CLI = Path.home() / "Downloads" / "llama-b11317-bin-win-cpu-x64" / "llama-cli.exe"
TOKENIZER_ID = "Qwen/Qwen2.5-1.5B-Instruct"
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--review", help="Review text to label")
    g.add_argument("--review-file", type=Path, help="File containing review text")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--llama-cli", type=Path, default=DEFAULT_CLI)
    p.add_argument("--tokenizer", default=TOKENIZER_ID)
    p.add_argument("--n-predict", type=int, default=128)
    p.add_argument("--temp", type=float, default=0.0)
    return p.parse_args()
def extract_json(text: str) -> object:
    """Pull a label JSON object out of noisy llama-cli output.

    llama-cli may echo the chat prompt (which contains example `{...}` schema
    text). Prefer the last parseable object that looks like a label.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates: list[object] = []
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[i : j + 1]
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and "overall" in obj and "aspects" in obj:
                        candidates.append(obj)
                    break

    if not candidates:
        raise ValueError(f"no label JSON found in output:\n{text}")
    return candidates[-1]


def main() -> None:
    args = parse_args()
    review = args.review if args.review is not None else args.review_file.read_text(encoding="utf-8")
    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")
    if not args.llama_cli.exists():
        raise SystemExit(f"llama-cli not found: {args.llama_cli}")
    # 1) Same messages as Colab / teacher
    messages = build_messages(review.strip())
    # 2) Format the Qwen chat way (tokenizer only — no big model load)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 3) One-shot call to llama-cli
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_path = f.name
    cmd = [
        str(args.llama_cli),
        "-m", str(args.model),
        "-f", prompt_path,
        "--no-jinja",          # already templated
        "-n", str(args.n_predict),
        "--temp", str(args.temp),
        "-st",                 # single turn, then exit
        "--no-display-prompt",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    finally:
        os.unlink(prompt_path)
    raw = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise SystemExit(f"llama-cli failed ({result.returncode}):\n{raw}")
    # 4) Parse + validate
    try:
        cleaned = validate_label(extract_json(raw))
    except Exception as e:
        print("RAW OUTPUT:\n", raw, file=sys.stderr)
        raise SystemExit(f"could not parse/validate: {e}") from e
    print(json.dumps(cleaned, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()