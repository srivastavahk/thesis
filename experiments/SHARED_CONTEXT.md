# Shared Project Context — WBP for LoRA Merging

> This file is included by reference in every experiment's AGENT_PROMPT.md.
> Read it in full before implementing any experiment.

---

## Project Summary

M.Tech thesis. The project compares two mathematically equivalent methods for calibrating LoRA adapters before merging:

- **Pico** (prior work, Tang & Yang): uses SVD over the stacked B-matrix `B_all` to downscale shared directions.
- **WBP** (this thesis): replaces that SVD with a closed-form Woodbury/Tikhonov filter on a much smaller Gram matrix, proven exactly equivalent via the Woodbury matrix identity.

**Both methods return calibrated B matrices for Task Arithmetic merging.** WBP's claim is that it achieves the same result without the GPU-bottlenecked SVD.

---

## Notation (use these exact variable names in all code)

| Symbol | Meaning | Shape |
|---|---|---|
| `W_0` | frozen pretrained weight | `(d_out, d_in)` |
| `B_t`, `A_t` | LoRA factors for task t; ΔW_t = B_t A_t | `(d_out, r)`, `(r, d_in)` |
| `T` | number of tasks / adapters being merged | scalar |
| `r` | LoRA rank | scalar |
| `B_all` | `[B_1 | ... | B_T]` stacked horizontally | `(d_out, T*r)` |
| `G` | Gram matrix = `B_all.T @ B_all` | `(T*r, T*r)` |
| `lambda` | Tikhonov regularization = `(T-1) / Tr(G)` | scalar |
| `beta` | exploratory scale factor on lambda; beta=1 ↔ exact Pico | scalar |
| `K` | kernel inverse = `(1/lambda * I + G)^-1` | `(T*r, T*r)` |
| `S_pico`, `S_wbp` | calibration operators (provably identical at beta=1) | `(d_out, d_out)` — never form explicitly |
| `gamma` | post-merge Frobenius-norm rescaling factor | scalar |

---

## Mathematical Formulas

### Pico (SVD path)
```
B_all = [B_1 | ... | B_T]   # (d_out, T*r)
U, S, Vh = svd(B_all, full_matrices=False)
s_j = sigma_j^2 / sum_k(sigma_k^2)      # energy sharing score
alpha_j = 1 / (1 + (T-1) * s_j)          # calibration coefficient
# Apply without forming S_pico explicitly:
B_tilde_t = B_t + U @ diag(alpha - 1) @ U.T @ B_t
```

### WBP (Woodbury path)
```
B_all = [B_1 | ... | B_T]             # (d_out, T*r)
G = B_all.T @ B_all                    # (T*r, T*r) — small!
lambda = (T - 1) / Tr(G) * beta
K = inv(1/lambda * I_{T*r} + G)        # invert small matrix only
# Apply right-to-left, never form S_wbp:
B_tilde_t = B_t - B_all @ (K @ (B_all.T @ B_t))
```

### Magnitude rescaling (identical for both)
```
# Numerator: average norm before calibration
avg_original_norm = (1/T) * sum_t ||B_t A_t||_F

# Calibrated merged update
delta_W_calib = (1/T) * sum_t (B_tilde_t @ A_t)

# Gamma rescaling factor
gamma = avg_original_norm / ||delta_W_calib||_F

# Final merged weight update
delta_W_final = gamma * delta_W_calib
```

### T=1 edge case
If T == 1: lambda = 0 → undefined. Return B_t unchanged (no calibration needed).

---

## Existing Code (already implemented & tested)

Repository root: `/Users/demid/thesis/`

```
src/
  pico.py     — merge_pico(B_list, A_list) -> (B_merged, A_merged)
  wbp.py      — merge_wbp(B_list, A_list, beta=1.0) -> (B_merged, A_merged)
  utils.py    — frobenius_norm_low_rank(), compute_gamma()
tests/
  test_equivalence.py  — validates Pico==WBP on random float64 matrices
```

Both `merge_pico` and `merge_wbp` return `(B_merged, A_merged)` as separate tensors.
`B_merged` has shape `(d_out, T*r)`, `A_merged` has shape `(T*r, d_in)`.
The final dense update is `B_merged @ A_merged`.

**Verified:** The two methods match to ~1e-15 relative error in float64. You can import and reuse these directly.

---

## Logging Requirements (mandatory for all experiments)

Every result file must include:
- Hardware: GPU model, CPU model for CPU runs
- Precision: fp32 / bf16 / fp64
- Warmup: state whether timing warmup iterations were excluded
- GPU-sync: for GPU timing, confirm `torch.cuda.synchronize()` was called before stopping the clock
- Seeds: random seed for any synthetic-matrix experiment; report mean ± std over ≥5 seeds
- For accuracy tables: benchmark name, metric (exact-match, pass@k, etc.), eval sample count

---

## Base Model & Adapter Configuration

- **Base model:** `meta-llama/Llama-3.1-8B` (base, not instruct variant — plain base for cleaner task separation)
- **LoRA rank:** r=16 (primary), r=8 (secondary check)
- **LoRA target modules:** `q_proj`, `v_proj` in attention layers
- **Tasks (T=4):** math, coding, finance, medical
- **Datasets:**
  - Math: `meta-math/MetaMathQA` (sample 10k)
  - Coding: `theblackcat102/evol-codealpaca-v1` (sample 10k)
  - Finance: `gbharti/finance-alpaca` (sample 10k)
  - Medical: `medalpaca/medical_meadow_medqa` (sample 10k)
- **Evaluation benchmarks:**
  - Math: GSM8K (exact match, full test set ~1319 examples)
  - Coding: HumanEval (pass@1, 164 problems)
  - Finance: FinQA subset or Fin-bench (exact match)
  - Medical: MedMCQA (accuracy, sample 1000 from validation)

---

## Conventions

- Match variable names to notation table above — no renaming.
- Save results to `/Users/demid/thesis/results/<experiment_id>/`.
- Every results file should be a self-contained JSON or CSV with metadata header.
- Do not use `print()` for results — use Python `logging` with timestamps.
