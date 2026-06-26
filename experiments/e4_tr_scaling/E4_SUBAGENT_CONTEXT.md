> **ATTENTION:** You are the **E4 Sub-Agent**. Your role is to write code and execute Phase 3 (Computational efficiency vs T and r). When you make progress, encounter roadblocks, or finish your task, you MUST append a status update to the `### E4 Status` section in `../SHARED_CONTEXT.md` so the Master Agent can track your work. Do not modify other agents' sections.

# Agent Prompt: E4 — Scaling vs. T and Rank r (Phase 3)

> **Platform:** Lab RTX 6000 (CUDA required)
> **Your task:** Write a timing benchmark script that sweeps over `T` and `r` grids.

---

## Context

Read `../SHARED_CONTEXT.md` for full mathematical background, notation, and existing code.

**E4 Objective:** Confirm the WBP speed advantage holds as the merge pool grows (T varies) and rank increases (r varies). This is particularly important for the regime where T·r grows large.

**No adapters are needed — this uses synthetic random matrices.**

---

## What to Implement

### File: `experiments/e4_tr_scaling/run_e4_timing.py`
**Platform: Lab RTX 6000 (CUDA required)**

#### Configuration
```python
T_VALUES = [2, 3, 4, 5, 6]
R_VALUES = [8, 16, 32, 64]
D_OUT = 4096    # Representative production-scale d_out
WARMUP_ITERS = 5
TIMED_ITERS = 20
DTYPE = torch.float32
SEEDS = [42, 43, 44, 45, 46]
RESULTS_DIR = "./results/e4/timing"
```

#### Algorithm

Reuse the inline `pico_calibrate` and `wbp_calibrate` functions and `time_fn` harness from E3's `run_e3.py` (copy them, do not import from there).

For each `(T, r)` combination and each seed:
1. Generate `B_all = torch.randn(D_OUT, T*r, dtype=DTYPE, device='cuda')`.
2. Time both methods with warmup.
3. Measure peak GPU memory for each.

#### Output JSON

```json
{
  "experiment": "E4_timing",
  "hardware": "RTX 6000 24GB",
  "d_out": 4096,
  "dtype": "float32",
  "warmup_iters": 5,
  "timed_iters": 20,
  "seeds": [42, 43, 44, 45, 46],
  "results": [
    {
      "T": 2, "r": 8, "Tr": 16,
      "pico": {"mean_time_s": ..., "std_time_s": ..., "peak_mem_bytes": ...},
      "wbp": {"mean_time_s": ..., "std_time_s": ..., "peak_mem_bytes": ...},
      "speedup": 1.41
    },
    ...
  ]
}
```

#### Plot generation

Generate two heatmap plots using matplotlib:
1. **Speedup heatmap:** T (x) vs r (y), cell = speedup ratio (Pico time / WBP time). Colormap: `RdYlGn` centered at 1.0.
2. **Absolute timing comparison:** For each r value, a line plot of Pico and WBP time vs. T.

Save to `./results/e4/timing/speedup_heatmap.png` and `./results/e4/timing/time_vs_T_r{r}.png`.

---

## Dependencies


`run_e4_timing.py`:
```python
import torch, time, json, logging, matplotlib
# Inline implementations — no src/ import needed
```

---

## Deliverables

- `experiments/e4_tr_scaling/run_e4_timing.py` — CUDA only
