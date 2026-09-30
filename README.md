# Prefill-Classifier

A working exploration of logit extraction as a way to turn a causal decoder language model into a structured classifier — reading a single prefill forward pass and treating the model's raw logits over a fixed set of choice tokens as calibrated probabilities.

Instead of decoding a response token by token, this approach runs one prefill pass and reads out logits for predefined choices. The repo covers the whole loop: extracting probabilities, calibrating them, and visualizing the result.

## What's inside

| Path | Purpose |
| --- | --- |
| `src/classifier/` | Logit-extraction classifiers built on `google/gemma-4-E2B-it` — single-question, multi-query batch, and a tutorial script |
| `src/calibration/` | Post-hoc probability calibration (temperature scaling, Platt scaling) and reliability-diagram plotting |
| `data/` | Demo datasets (IMDB and AG-News) |
| `notebooks/` | End-to-end demos: classify, calibrate, and plot before/after |
| `docs/` | Research notes and a revealjs slide deck on the paradigm |

## Core idea

1. **Extract** — run a prefill-only forward pass and pull logits for choice tokens, softmax them into probabilities.
2. **Calibrate** — fit temperature or Platt scaling so the reported confidences match observed accuracy.
3. **Inspect** — reliability diagrams show calibration quality per class and overall.

## Getting started

This is a `uv` project targeting Python 3.13. Install dependencies with:

```bash
uv sync
```

The runnable pipeline lives in `src/`. See `notebooks/classify_review_type.qmd` for a full walkthrough — loading a classifier, scoring the review dataset, sweeping calibration, and rendering before/after reliability plots.
