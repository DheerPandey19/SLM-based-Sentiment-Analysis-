# Small Language Model (SLM) Aspect Sentiment Tagger

Distill GPT-4o-mini’s judgment into a **1.5B** model that turns a movie review into structured JSON: overall sentiment plus per-aspect tags (plot, acting, visuals, pacing, dialogue, soundtrack, direction) — then run it locally as GGUF on CPU.

**Headline so far:** under a strong few-shot prompt the fine-tuned and base models are close; under a **weak** prompt (no few-shot), the fine-tuned model wins across the board. Distillation buys robustness when scaffolding is removed. The remaining hole is **aspect over-tagging** (precision ~0.3): both locals invent tags the gold label did not use.

All numbers below are on the full held-out test set (**250** reviews).

## Pipeline

1. **Data** — IMDb-style reviews, teacher-labeled with **GPT-4o-mini**, split train / val / test (~2000 / 250 / 250).
2. **Train** — **LoRA** fine-tune of **Qwen2.5-1.5B-Instruct** on Colab (adapter local / Drive, not in git).
3. **Deploy** — Merge → **GGUF Q4_K_M** → **llama.cpp** on CPU.
4. **Eval** — Fine-tuned GGUF vs base GGUF; optional GPT-4o-mini ceiling.

Gold labels are teacher-written, so a full GPT row would be a soft ceiling (teacher agreeing with itself), not independent human truth. GPT on the full 250 is optional; an n=25 pilot sat at ~100% overall / ~0.98 aspect F1.

## Results (n=250)

### Full prompt (system + few-shot)

| System | JSON validity | Overall accuracy | Aspect P | Aspect R | Aspect F1 | Mean latency |
|--------|---------------|------------------|----------|----------|-----------|--------------|
| Fine-tuned Qwen GGUF (ours) | **98.4%** | 87.0% | 0.34 | **0.73** | **0.47** | ~22 s (CPU) |
| Base Qwen2.5-1.5B-Instruct GGUF | 93.2% | **88.4%** | 0.34 | 0.69 | 0.45 | ~22 s (CPU) |

Close call: fine-tuned edges validity and aspect F1; base slightly ahead on overall sentiment.

### Weak prompt (short system, no few-shot) — `--prompt weak`

| System | JSON validity | Overall accuracy | Aspect P | Aspect R | Aspect F1 | Mean latency |
|--------|---------------|------------------|----------|----------|-----------|--------------|
| Fine-tuned Qwen GGUF (ours) | **96.0%** | **86.3%** | **0.31** | **0.79** | **0.44** | ~14 s (CPU) |
| Base Qwen2.5-1.5B-Instruct GGUF | 86.4% | 83.8% | 0.28 | 0.71 | 0.41 | ~11 s (CPU) |

Fine-tuned wins every column. That is the writeup claim: LoRA helps most when you drop the few-shot crutch.

**Overall accuracy** = `overall` match among valid JSON only. **Aspect P/R/F1** = exact `(aspect, sentiment)` pairs. High overall does not mean good aspect tagging.

### What is still wrong

Both models **over-predict aspects**. Gold averages ~2–3 tags; preds often dump most of the allowed list. Precision stays ~0.28–0.34 even when overall looks fine (~86–88%). Fixing that is the next phase — not chasing overall accuracy further.



## Repo layout

| Path | Role |
|------|------|
| `src/schema.py` | Allowed aspects and label validation |
| `src/prompt.py` | `full` / `weak` chat messages |
| `src/teacher.py` | GPT-4o-mini labeling helper |
| `src/gguf_infer.py` | Local GGUF inference via llama.cpp |
| `src/metrics.py` | Eval metrics |
| `scripts/` | Download, label, split, infer, eval |
| `data/splits/` | Train / val / test JSONL |
| `docs/roadmap.md` | Longer project plan |

Large artifacts (**GGUF**, LoRA adapters, `outputs/`) stay local or on Drive — see `.gitignore`.

## Quick start

Requires: `requirements.txt`, `llama-cli` from llama.cpp, and `qwen-aspect-Q4_K_M.gguf` in the project root.

```powershell
pip install -r requirements.txt
python scripts/infer_gguf.py --review "Great acting, but the pacing dragged."
```

Eval:

```powershell
python scripts/eval_test.py --backend gguf --model qwen-aspect-Q4_K_M.gguf --resume
python scripts/eval_test.py --backend gguf --model qwen2.5-1.5b-instruct-q4_k_m.gguf --resume
python scripts/eval_test.py --backend gguf --model qwen-aspect-Q4_K_M.gguf --prompt weak --resume
```

`--limit 25` for a smoke run. `--prompt weak` writes separate `*_weak.jsonl` dumps.

## Status

| Phase | Status |
|-------|--------|
| Data + teacher labels + splits | Done |
| LoRA fine-tune + GGUF export | Done (artifacts local) |
| Full test eval (n=250) FT vs base | Done |
| Weak-prompt ablation (n=250) | Done — FT wins |
| Label cleanup + retrain + demo / writeup | Next |

Goal: a clear, reproducible SLM distillation case study — not a one-shot leaderboard claim.
