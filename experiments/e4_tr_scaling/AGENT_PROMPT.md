# Agent Prompt: E4 — Scaling vs. T and Rank r

> **Platform:** Two sub-parts with different hardware requirements:
>   - **Equivalence grid check:** Mac Mini M4 (CPU) — runs on synthetic matrices
>   - **Timing grid:** Lab RTX 6000 (CUDA) — synthetic matrices only, bundle with E3
> **Your task:** Write TWO scripts: one for the equivalence check (Mac compatible), one for the timing benchmark (CUDA required).

---

## Context

Read `../SHARED_CONTEXT.md` for full mathematical background, notation, and existing code.

**E4 Objective:** Confirm equivalence and speed hold as the merge pool grows (T varies) and rank increases (r varies). This is particularly important for the regime where T·r grows large.

**No new adapters are needed — both sub-parts use synthetic random matrices.**

---

## What to Implement

### File 1: `experiments/e4_tr_scaling/run_e4_equivalence.py`
**Platform: Mac Mini M4 (CPU) — NO CUDA**

#### Configuration
```python
T_VALUES = [2, 3, 4, 5, 6]
R_VALUES = [8, 16, 32, 64]
D_OUT = 1024     # Fixed d_out for equivalence check
DTYPE = torch.float64  # High precision for equivalence verification
SEEDS = [42, 43, 44, 45, 46]
RESULTS_DIR = "./results/e4/equivalence"
```

#### Algorithm

For each `(T, r)` combination and each `seed`:
1. Generate `B_list = [torch.randn(D_OUT, r) for _ in range(T)]` and `A_list = [torch.randn(r, D_OUT//2) for _ in range(T)]`.
2. Run both `merge_pico(B_list, A_list)` and `merge_wbp(B_list, A_list, beta=1.0)`.
3. Compute relative Frobenius error on `B_merged`:
   ```python
   rel_err = (B_pico - B_wbp).norm('fro') / (B_pico.norm('fro') + 1e-12)
   ```
4. Record per-(T, r, seed) and aggregate: mean_rel_err, max_rel_err over seeds.

#### Output JSON

```json
{
  "experiment": "E4_equivalence",
  "hardware": "Mac Mini M4 (CPU)",
  "d_out": 1024,
  "dtype": "float64",
  "seeds": [42, 43, 44, 45, 46],
  "results": [
    {"T": 2, "r": 8, "Tr": 16, "mean_rel_error": 1.23e-15, "max_rel_error": 2.1e-15},
    {"T": 2, "r": 16, "Tr": 32, "mean_rel_error": ...},
    ...
  ],
  "all_passed": true
}
```

Pass criterion: `max_rel_error < 1e-5` for all (T, r) pairs.

Also generate a heatmap plot (T on x-axis, r on y-axis, cell color = log10(max_rel_error)) saved to `./results/e4/equivalence_heatmap.png`.

---

### File 2: `experiments/e4_tr_scaling/run_e4_timing.py`
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

`run_e4_equivalence.py`:
```python
import torch, json, logging, matplotlib
# src/ package required — run with PYTHONPATH=.
from src.pico import merge_pico
from src.wbp import merge_wbp
```

`run_e4_timing.py`:
```python
import torch, time, json, logging, matplotlib
# Inline implementations — no src/ import needed
```

---

## Deliverables

- `experiments/e4_tr_scaling/run_e4_equivalence.py` — Mac Mini compatible
- `experiments/e4_tr_scaling/run_e4_timing.py` — CUDA only
