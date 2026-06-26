# Agent Prompt: E5 — Decoupled-λ Sweep (β-WBP), Exploratory

> **Platform:** Lab RTX 6000 (24 GB VRAM, CUDA)
> **Your task:** Extend the E2 pipeline to sweep β ∈ {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0} and produce an accuracy-vs-β curve for each benchmark and overall average.

---

## Context

Read `../SHARED_CONTEXT.md` for full mathematical background, notation, and existing code.
Read `experiments/e2_accuracy/AGENT_PROMPT.md` — E5 reuses the E2 pipeline verbatim, adding only a β loop.# E5 — Decoupled-lambda Sweep (Phase 4)

**E5 Objective:** Generalize the calibration. Take a first, exploratory look at whether decoupling the WBP filter from the exact Pico-match constraint ($\beta=1$) can improve accuracy. Evaluated via Task Arithmetic (we do not need to run TIES here since it's exploratory). This is framed as **exploratory** — no formal tuning protocol, no cross-domain generalization claims.

The β parameter scales the Tikhonov regularization:
```
lambda_beta = beta * (T - 1) / Tr(G)
```
- `beta = 1.0` → exact Pico match (proven mathematically)
- `beta < 1.0` → weaker suppression of shared directions
- `beta > 1.0` → stronger suppression of shared directions

**Hardware note:** This is 7 × E2 evaluations. Expect 8–16 hours total. Book a dedicated lab GPU slot.

---

## What to Implement

### File: `experiments/e5_beta_sweep/run_e5.py`

This script is a direct extension of `run_e2.py`. It:
1. Runs E2's pipeline for each β value.
2. Only tests WBP (no need to re-run no-cal or Pico for each β — run them once, reuse results).
3. Produces a plot of overall avg accuracy vs. β.

#### CLI arguments

```bash
python run_e5.py \
  --adapters_dir /path/to/adapters \
  --base_model unsloth/Meta-Llama-3.1-8B \
  --output_dir /path/to/results/e5 \
  --dtype bfloat16 \
  --device cuda \
  --seed 42 \
  --e2_results_json /path/to/results/e2/results.json  # to overlay Pico and no-cal baselines
```

#### β sweep values (hardcoded, not CLI)
```python
BETA_VALUES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
```

#### Algorithm

1. **Load base model and adapters** — same as E2.
2. **Load E2 baseline results** from `--e2_results_json` — extract no-cal and Pico scores for overlay on plots.
3. **For each β in BETA_VALUES:**
   a. Apply WBP with `beta=β` to compute merged weight deltas per layer.
   b. Evaluate on all 4 benchmarks using `evaluate.py` (reuse from E2 — copy the file to this directory or import from E2).
   c. Compute the arithmetic average across benchmarks.
   d. Record results.
4. **Write results JSON** (see below).
5. **Generate plots** (see below).

**Optimization:** The benchmark datasets and tokenizer can be loaded once and reused across all β values. Only re-merge weights and re-evaluate for each β.

#### Output JSON structure

```json
{
  "experiment": "E5",
  "hardware": "RTX 6000 24GB",
  "base_model": "unsloth/Meta-Llama-3.1-8B",
  "dtype": "bfloat16",
  "T": 4,
  "seed": 42,
  "beta_values": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0],
  "baselines": {
    "no_cal_average": 0.259,
    "pico_average": 0.294
  },
  "results": [
    {
      "beta": 0.25,
      "gsm8k_exact_match": 0.287,
      "humaneval_pass_at_1": 0.098,
      "finqa_exact_match": 0.201,
      "medmcqa_accuracy": 0.448,
      "average": 0.2585
    },
    ...
    {
      "beta": 1.0,
      "average": 0.294   // Should match pico_average to float precision
    },
    ...
  ]
}
```

#### Plots to generate

**Plot 1: Overall average accuracy vs. β**
- X-axis: β values (log scale if desired, or linear)
- Y-axis: Average accuracy across 4 benchmarks
- Horizontal dashed lines for no-cal baseline and Pico (β=1) baseline
- Mark β=1 point with a vertical dashed line labeled "Pico-equivalent"
- Title: "WBP β sweep — Average benchmark accuracy (T=4, Meta-Llama-3.1-8B)"
- Save to `./results/e5/beta_sweep_avg.png` at 300 DPI

**Plot 2: Per-benchmark accuracy vs. β (4 subplots)**
- One subplot per benchmark, same x-axis
- Each line shows accuracy for that benchmark as β varies
- Horizontal lines for Pico and no-cal on each subplot
- Save to `./results/e5/beta_sweep_per_benchmark.png`

---

## Interpretation Guide (include as comments in the script)

The script should log an interpretation string at the end:

```python
peak_beta = beta_values[np.argmax(averages)]
peak_val = max(averages)
pico_val = e2_results["pico"]["average"]

if peak_beta != 1.0 and peak_val > pico_val + 0.005:
    logging.info(f"FINDING: Curve peaks at beta={peak_beta} (avg={peak_val:.3f}), "
                 f"above Pico ({pico_val:.3f}). The family contains better points than beta=1. "
                 f"NOTE: A principled data-free beta selection method is still needed.")
elif max(averages) - min(averages) < 0.01:
    logging.info("FINDING: Flat curve — Pico's beta=1 was already near-optimal in this regime.")
elif averages[0] > averages[-1]:
    logging.info("FINDING: Monotonically decreasing with beta — over-shrinkage beyond beta=1.")
else:
    logging.info("FINDING: Monotonically increasing with beta — under-shrinkage at beta=1.")
```

---

## Important: What NOT to claim from E5

Add these as comments in `run_e5.py` for documentation:
```python
# DO NOT CLAIM from this experiment alone:
# - That WBP-tuned "beats" Pico as a deployable result
# - That the chosen beta generalizes to other models or domains
# - That the data-free property is preserved (beta chosen via benchmark accuracy is NOT data-free)
# These are explicitly acknowledged limitations in the thesis's Future Work section.
```

---

## Dependencies

Same as E2:
```bash
pip install transformers peft datasets safetensors accelerate evaluate torch
```

---

## Deliverables

- `experiments/e5_beta_sweep/run_e5.py`
- Copy `evaluate.py` from E2 into this directory (or import from relative path)
