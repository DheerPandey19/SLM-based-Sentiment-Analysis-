# Movie/Book Review Sentiment + Aspect Tagger

> Historical plan. The shipped case study is **movies only** (Phases 1–3 + eval + weak-prompt ablation + inference aspect filter). See the root [README](../README.md) for final results. Books, Gradio, and NPU were not built.

## Task definition

Given a review, output structured tags instead of just "positive/negative":

- **Overall sentiment:** positive / negative / mixed
- **Aspects mentioned (movies):** plot, acting, visuals/cinematography, pacing, dialogue, soundtrack, direction
- **Aspects mentioned (books):** plot, characters, writing style, pacing, ending, world-building
- **Per-aspect sentiment:** e.g. "acting: positive, pacing: negative"

### Example output

```json
{
  "overall": "mixed",
  "aspects": {
    "acting": "positive",
    "pacing": "negative",
    "ending": "negative"
  }
}
```

This is genuinely useful (real companies want per-aspect breakdowns, not just star ratings) and forces the model to do structured JSON output — a skill that matters a lot in production LLM/SLM work.

## Datasets

- **IMDb** (50k movie reviews, binary sentiment) — good base but no aspects
- **SST** (Stanford Sentiment Treebank) — fine-grained sentiment
- **Amazon Book Reviews** (via Hugging Face `amazon_reviews_multi` or the book category of Amazon Review Data) — has star ratings + free text

For aspects, you'll likely need to generate synthetic labels: use a larger model (Claude/GPT via API, cheap for a few thousand examples) to auto-label a subset of real reviews with aspect+sentiment pairs, then have your fine-tuned small model learn to replicate that. This "distill a big model's judgment into a small model" pattern is itself a well-known, legitimate industry technique worth mentioning explicitly in your writeup.

## Week-by-week plan

### Phase 1 — Data

- Pull ~2,000–3,000 movie/book reviews (mix IMDb + Amazon books)
- Use Claude/GPT API to auto-label each with overall sentiment + aspect-sentiment JSON (few-shot prompt, cheap at this scale)
- Manually spot-check ~100 labels for quality, fix systematic errors
- Split train/val/test (80/10/10)

### Phase 2 — Fine-tune

- Base model: Qwen2.5-1.5B-Instruct (better instruction-following for structured JSON output than Llama-3.2-1B)
- LoRA fine-tune on Colab/Kaggle free GPU using transformers + peft + trl
- Prompt format: `Review: {text}\nExtract sentiment and aspects as JSON:`
- Train for a few epochs, watch val loss, log with W&B

### Phase 3 — Quantize + deploy locally

- Convert to GGUF (via llama.cpp's conversion scripts), quantize to Q4_K_M
- Run locally via llama.cpp on your Envy x360
- Optionally: convert to OpenVINO IR format and run on the NPU, compare tokens/sec vs CPU

### Phase 4 — Benchmark + writeup

- Compare on held-out test set: your fine-tuned 1.5B model vs. base un-tuned Qwen2.5-1.5B vs. GPT-4o-mini (same prompt)
- Metrics: JSON validity rate (does it even produce parseable output), aspect-detection F1, sentiment accuracy, latency, cost-per-1000-reviews
- Build a small Gradio demo where you paste a review and see the extracted tags
- Write the case study: problem → data → method → results table → cost/latency tradeoff → NPU benchmark → what you'd change for production

## Why this scopes well as a portfolio piece

- Structured output (JSON) is harder and more impressive than plain classification — shows you can constrain a small model to a schema
- The distillation angle (big model labels data → small model learns it) is a real, current industry pattern
- The final comparison table (fine-tuned SLM vs. base vs. GPT-4o-mini, plus CPU vs NPU) gives you multiple concrete numbers to put in a resume bullet, not just "I built a model"
