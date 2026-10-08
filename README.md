# Small Language Model (SLM) Aspect Sentiment Tagger

Portfolio case study: distill **GPT-4o-mini** into a local **1.5B** model that turns a movie review into structured JSON — overall sentiment plus per-aspect tags (plot, acting, visuals, pacing, dialogue, soundtrack, direction) — then run it on CPU as **GGUF**.

## Summary

| Finding | Result |
|---------|--------|
| Distillation vs base | Under a **weak** prompt (no few-shot), fine-tuned wins every metric |
| Structured output | Fine-tuned JSON validity **96–99%** on the held-out set |
| Over-tagging fix | Inference **evidence filter** raises aspect precision **0.34 → 0.63** and F1 **0.47 → 0.54** without retraining |
| Scope delivered | Movies only; books / NPU left out of the final build |

All tables use the full held-out test set (**250** reviews). Gold labels are teacher-written, so a GPT row would be a soft ceiling, not independent human truth.

## What we built

1. **Data** — IMDb-style reviews, labeled with GPT-4o-mini, split ~2000 / 250 / 250.
2. **Train** — LoRA fine-tune of **Qwen2.5-1.5B-Instruct** on Colab.
3. **Deploy** — Merge → **GGUF Q4_K_M** → **llama.cpp** on CPU.
4. **Eval** — Fine-tuned vs base; weak-prompt ablation; optional aspect evidence filter.

## Results (n=250)

### Full prompt (system + few-shot)

| System | JSON validity | Overall accuracy | Aspect P | Aspect R | Aspect F1 | Mean latency |
|--------|---------------|------------------|----------|----------|-----------|--------------|
| Fine-tuned Qwen GGUF (ours) | **98.4%** | 87.0% | 0.34 | **0.73** | **0.47** | ~22 s (CPU) |
| Base Qwen2.5-1.5B-Instruct GGUF | 93.2% | **88.4%** | 0.34 | 0.69 | 0.45 | ~22 s (CPU) |

### Weak prompt (no few-shot) — `--prompt weak`

| System | JSON validity | Overall accuracy | Aspect P | Aspect R | Aspect F1 | Mean latency |
|--------|---------------|------------------|----------|----------|-----------|--------------|
| Fine-tuned Qwen GGUF (ours) | **96.0%** | **86.3%** | **0.31** | **0.79** | **0.44** | ~14 s (CPU) |
| Base Qwen2.5-1.5B-Instruct GGUF | 86.4% | 83.8% | 0.28 | 0.71 | 0.41 | ~11 s (CPU) |

### Aspect evidence filter — `--filter-aspects`

Drop predicted aspects with no keyword cue in the review (plus a sparse few-shot in the full prompt).

| System | JSON validity | Overall accuracy | Aspect P | Aspect R | Aspect F1 | Mean latency |
|--------|---------------|------------------|----------|----------|-----------|--------------|
| Fine-tuned + filter | **99.2%** | 85.9% | **0.63** | 0.48 | **0.54** | ~20 s (CPU) |
| Fine-tuned (no filter) | 98.4% | 87.0% | 0.34 | **0.73** | 0.47 | ~22 s (CPU) |

**Overall accuracy** = `overall` match among valid JSON. **Aspect P/R/F1** = exact `(aspect, sentiment)` pairs.

## Takeaways

- Distillation helps most when few-shot scaffolding is removed.
- Overall sentiment (~86–88%) was never the hard part; inventing aspects was.
- A cheap lexical filter fixed over-tagging without another LoRA pass — precision nearly doubled, at the cost of some recall.

## Repo layout

| Path | Role |
|------|------|
| `src/schema.py` | Allowed aspects and label validation |
| `src/prompt.py` | `full` / `weak` chat messages |
| `src/aspect_filter.py` | Drop aspects without review evidence |
| `src/teacher.py` | GPT-4o-mini labeling helper |
| `src/gguf_infer.py` | Local GGUF inference via llama.cpp |
| `src/metrics.py` | Eval metrics |
| `scripts/` | Download, label, split, infer, eval |
| `data/splits/` | Train / val / test JSONL |
| `docs/roadmap.md` | Original project plan (historical) |

Large artifacts (**GGUF**, LoRA adapters, `outputs/`) stay local or on Drive — see `.gitignore`.

## Quick start

Requires: `requirements.txt`, `llama-cli` from llama.cpp, and `qwen-aspect-Q4_K_M.gguf` in the project root.

```powershell
pip install -r requirements.txt
python scripts/infer_gguf.py --review "Great acting, but the pacing dragged." --filter-aspects
```

```powershell
python scripts/eval_test.py --backend gguf --model qwen-aspect-Q4_K_M.gguf --filter-aspects --resume
```

`--prompt weak` and `--filter-aspects` write separate dumps (`*_weak.jsonl`, `*_filtered.jsonl`). Use `--limit 25` for a smoke run.
