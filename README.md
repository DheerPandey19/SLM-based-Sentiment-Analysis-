# Small Language Model (SLM) Aspect Sentiment Tagger

Work-in-progress project: turn a movie review into structured **JavaScript Object Notation (JSON)** with overall sentiment and per-aspect sentiments (plot, acting, visuals, pacing, dialogue, soundtrack, direction).

Results below are on the full held-out test set (**250** reviews), not a small pilot slice.

## What it does

1. **Data** — Pull IMDb-style reviews, label them with a teacher model (**Generative Pre-trained Transformer 4o mini (GPT-4o-mini)**), split train / validation / test.
2. **Train** — **Low-Rank Adaptation (LoRA)** fine-tune of **Qwen2.5-1.5B-Instruct** on Google Colab (adapter kept local / Google Drive, not in git).
3. **Deploy locally** — Merge adapter, convert to **GPT-Generated Unified Format (GGUF)**, quantize to **Q4_K_M**, run with **llama.cpp** on CPU.
4. **Eval** — Compare fine-tuned GGUF vs base Qwen GGUF (and optionally GPT-4o-mini) on the held-out test set.

## Full test results (250 reviews)

Same prompt recipe for both local systems (system instructions + few-shot example). Gold labels were written by GPT-4o-mini, so a teacher row would be a soft ceiling (teacher agreeing with itself), not an independent human score. Full-test GPT eval is optional / not run yet; an earlier n=25 pilot had GPT at ~100% overall / ~0.98 aspect F1.

| System | JSON validity | Overall accuracy | Aspect P | Aspect R | Aspect F1 | Mean latency |
|--------|---------------|------------------|----------|----------|-----------|--------------|
| Fine-tuned Qwen GGUF (ours) | **98.4%** | 87.0% | 0.34 | **0.73** | **0.47** | ~22 s (CPU) |
| Base Qwen2.5-1.5B-Instruct GGUF | 93.2% | **88.4%** | 0.34 | 0.69 | 0.45 | ~22 s (CPU) |

**F1** = harmonic mean of precision and recall for aspect–sentiment pairs.

### Honest takeaway so far

- A small n=25 pilot overstated “base clearly wins.” On **n=250**, fine-tuned edges **aspect F1** and **JSON validity**; overall accuracy is roughly tied (base slightly ahead).
- Both local models still **over-predict aspects** (precision ~0.34): they invent tags the gold label did not use. That is the main quality hole.
- Next: weak-prompt ablation (does LoRA help without few-shot scaffolding?), then cleaner / sparser labels and a format-aligned retrain.

## How we plan to improve accuracy

1. ~~**Full test evaluation**~~ — Done (250 reviews, fine-tuned vs base).
2. **Prompt ablation** — Compare base vs fine-tuned with a short / minimal prompt. LoRA often helps when the long few-shot scaffold is removed.
3. **Cleaner aspect labels** — Spot-check teacher labels; add more examples with sparse or empty `aspects` so the model learns when *not* to tag.
4. **Training alignment** — Make sure Colab training uses the same chat template and `build_messages` format as local eval.
5. **Tune LoRA** — Adjust learning rate, rank, and early stopping using aspect F1 (not only loss).
6. **Optional decoding constraints** — JSON / grammar constrained decoding in llama.cpp if validity ever drops.

## Repo layout

| Path | Role |
|------|------|
| `src/schema.py` | Allowed aspects and label validation |
| `src/prompt.py` | Teacher / student chat messages |
| `src/teacher.py` | GPT-4o-mini labeling helper |
| `src/gguf_infer.py` | Local GGUF inference via llama.cpp |
| `src/metrics.py` | Eval metrics |
| `scripts/` | Download, label, split, infer, eval |
| `data/splits/` | Train / val / test JSONL (labels included) |
| `docs/roadmap.md` | Longer project plan |

Large files (**GGUF** weights, LoRA adapters, raw eval dumps under `outputs/`) stay local or on Drive — see `.gitignore`.

## Quick start (local GGUF)

Requires: Python deps from `requirements.txt`, `llama-cli` from llama.cpp, and your `qwen-aspect-Q4_K_M.gguf` in the project root.

```powershell
pip install -r requirements.txt
python scripts/infer_gguf.py --review "Great acting, but the pacing dragged."
```

Full test eval (example):

```powershell
python scripts/eval_test.py --backend gguf --model qwen-aspect-Q4_K_M.gguf --resume
python scripts/eval_test.py --backend gguf --model qwen2.5-1.5b-instruct-q4_k_m.gguf --resume
```

Use `--limit 25` for a quick smoke run.

## Status

| Phase | Status |
|-------|--------|
| Data + teacher labels + splits | Done |
| LoRA fine-tune + GGUF export | Done (artifacts local) |
| Local infer + pilot eval (n=25) | Done |
| Full test eval (n=250) FT vs base | Done |
| Prompt ablation + label cleanup + demo / writeup polish | In progress |

Feedback and iteration welcome — the goal is a clear, reproducible SLM case study, not a one-shot leaderboard claim.
