"""Run a Qwen chat-templated review through llama-cli + a local GGUF."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from transformers import AutoTokenizer

from .prompt import build_messages
from .schema import validate_label

DEFAULT_TOKENIZER_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_LLAMA_CLI = (
    Path.home() / "Downloads" / "llama-b11317-bin-win-cpu-x64" / "llama-cli.exe"
)

# Cache tokenizer across many eval calls in one process.
_tokenizer: AutoTokenizer | None = None
_tokenizer_id: str | None = None


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


def _get_tokenizer(tokenizer_id: str) -> AutoTokenizer:
    global _tokenizer, _tokenizer_id
    if _tokenizer is None or _tokenizer_id != tokenizer_id:
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        _tokenizer_id = tokenizer_id
    return _tokenizer


def run_gguf(
    review: str,
    model_path: Path,
    llama_cli: Path = DEFAULT_LLAMA_CLI,
    tokenizer_id: str = DEFAULT_TOKENIZER_ID,
    n_predict: int = 128,
    temp: float = 0.0,
) -> tuple[dict, float]:
    """Label one review with a local GGUF.

    Returns (validated {overall, aspects}, latency_seconds).
    Raises on llama-cli failure or unparseable output.
    """
    model_path = Path(model_path)
    llama_cli = Path(llama_cli)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not llama_cli.exists():
        raise FileNotFoundError(f"llama-cli not found: {llama_cli}")

    messages = build_messages(review.strip())
    tok = _get_tokenizer(tokenizer_id)
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".txt", delete=False
    ) as f:
        f.write(prompt)
        prompt_path = f.name

    cmd = [
        str(llama_cli),
        "-m",
        str(model_path),
        "-f",
        prompt_path,
        "--no-jinja",
        "-n",
        str(n_predict),
        "--temp",
        str(temp),
        "-st",
        "--no-display-prompt",
    ]
    t0 = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    finally:
        os.unlink(prompt_path)
    latency = time.perf_counter() - t0

    raw = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(f"llama-cli failed ({result.returncode}):\n{raw}")

    cleaned = validate_label(extract_json(raw))
    return cleaned, latency
