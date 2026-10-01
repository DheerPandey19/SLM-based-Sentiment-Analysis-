# Small Language Model (SLM) Aspect Sentiment Tagger

Work-in-progress project: turn a movie review into structured **JavaScript Object Notation (JSON)** with overall sentiment and per-aspect sentiments (plot, acting, visuals, pacing, dialogue, soundtrack, direction).

This repo is still improving. Early results are a **pilot** on 25 test reviews, not a final claim.

## What it does

1. **Data** — Pull IMDb-style reviews, label them with a teacher model (**Generative Pre-trained Transformer 4o mini (GPT-4o-mini)**), split train / validation / test.
2. **Train** — **Low-Rank Adaptation (LoRA)** fine-tune of **Qwen2.5-1.5B-Instruct** on Google Colab (adapter kept local / Google Drive, not in git).
3. **Deploy locally** — Merge adapter, convert to **GPT-Generated Unified Format (GGUF)**, quantize to **Q4_K_M**, run with **llama.cpp** on CPU.
4. **Eval** — Compare fine-tuned GGUF vs base Qwen GGUF vs GPT-4o-mini on the held-out test set.

## Pilot results (25 test reviews)

Same prompt recipe for all systems (system instructions + few-shot example). Gold labels were written by GPT-4o-mini, so that row is a soft ceiling (teacher agreeing with itself), not an independent human score.

| System | JSON validity | Overall accuracy | Aspect F1 score | Mean latency |
|--------|---------------|------------------|-----------------|--------------|
| Fine-tuned Qwen GGUF (ours) | 100% | 84% | 0.44 | ~11 s (CPU) |
| Base Qwen2.5-1.5B-Instruct GGUF | 100% | 92% | 0.43 | ~11 s (CPU) |
| GPT-4o-mini (teacher) | 100% | 100% | 0.98 | ~1.4 s |

**F1** = harmonic mean of precision and recall for aspect–sentiment pairs.

### Honest takeaway so far

- Structured JSON output works well (100% valid).
- On this small slice, **base + strong prompting already matches or beats our LoRA model** on overall accuracy; aspect F1 is almost tied.
- Both local models **over-predict aspects** (low precision ~0.33): they invent tags the gold label did not use.
- Fine-tuning may still help under weaker prompts or on the full 250-review test set — that work is next.

## How we plan to improve accuracy

1. **Full test evaluation** — Run all 250 test reviews (not only 25) so the comparison is stable.
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

Pilot eval (example):

```powershell
python scripts/eval_test.py --backend gguf --model qwen-aspect-Q4_K_M.gguf --limit 25
python scripts/eval_test.py --backend openai --model gpt-4o-mini --limit 25
```

## Status

| Phase | Status |
|-------|--------|
| Data + teacher labels + splits | Done |
| LoRA fine-tune + GGUF export | Done (artifacts local) |
| Local infer + pilot eval (n=25) | Done |
| Full test + prompt ablation + demo / writeup polish | In progress |

Feedback and iteration welcome — the goal is a clear, reproducible SLM case study, not a one-shot leaderboard claim.
