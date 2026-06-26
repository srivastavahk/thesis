# AGENT.md — Project Context: WBP for LoRA Merging

> **Purpose of this file.** This is the single onboarding document for this project — for an AI
> coding agent (Claude Code, Cursor, etc.) starting a new session, or a human collaborator picking
> this up cold. Read this file first. It tells you what's already settled (don't re-derive or
> re-litigate it) and what's still open (where your effort should actually go). Detailed material
> lives in the companion files listed in §6 — this file is the map, not the territory.

---

## 1. Project Summary

M.Tech thesis project: **"Woodbury B-Space Preconditioning: Fast and Generalized B-Space Calibration for Model Merging"**.

This project defends a novel approach to LoRA merging interference through a four-phase narrative:
1. **Verification (E0):** Verifying the claims of subspace overlap in the original "Crowded in B-Space" paper by checking real adapters using their proposed metrics.
2. **Novel Algorithm (E1):** Proposing Woodbury B-Space Preconditioning (WBP), an SVD-free algorithm that suppresses shared directions efficiently using pure GEMM operations on a small $Tr\times Tr$ Gram matrix.
3. **Benchmarking & Equivalence (E2, E3, E4):** Empirically proving the exact mathematical equivalence between WBP and Pico, quantifying the computational advantage of WBP over Pico, and benchmarking downstream merging performance using both **Task Arithmetic** and **TIES** across three regimes (No Preconditioning, Pico, WBP).
4. **Generalization (E5):** Decoupling our method from the strict Pico equivalence constraint via a tunable $\beta$ parameter to study the preliminary effects of over/under-calibrating subspace interference.

---

## 2. Source Material

- `Crowded_in_B-Space.pdf` — the original Pico paper. Ground truth for Pico's algorithm, its
  empirical motivation (Section 3, Appendix A), and all baseline numbers (Table 1, Table 4).
- `thesis-maths.md` — full derivations for both Pico (restated) and WBP (novel). **Already verified
  line-by-line, including a numerical check with random matrices confirming the equivalence theorem
  to machine precision.** Treat every formula in that file as correct unless you find a specific
  counterexample — don't re-derive from scratch.

---

## 3. Mathematical Core (verified — reference, don't re-derive)

### 3.1 Notation
| Symbol | Meaning |
|---|---|
| $W_0$ | frozen pretrained weight, $d_{out}\times d_{in}$ |
| $B_t, A_t$ | LoRA factors for task $t$; $\Delta W_t = B_tA_t$; $B_t\in\mathbb{R}^{d_{out}\times r}$, $A_t\in\mathbb{R}^{r\times d_{in}}$ |
| $T$ | number of tasks/adapters being merged |
| $r$ | LoRA rank |
| $B_{all}$ | $[B_1\|\cdots\|B_T]\in\mathbb{R}^{d_{out}\times Tr}$ |
| $U,\Sigma,V$ | SVD of $B_{all}$ |
| $s_j,\alpha_j$ | Pico's sharing score and calibration coefficient for direction $j$ |
| $C = B_{all}B_{all}^\top$ | uncentered covariance, $d_{out}\times d_{out}$ |
| $G = B_{all}^\top B_{all}$ | Gram matrix, $Tr\times Tr$ — this is what WBP actually computes |
| $\lambda$ | Tikhonov regularization strength, $=(T-1)/\text{Tr}(G)$ |
| $\beta$ | exploratory scale factor on $\lambda$; $\beta=1$ ⟺ exact Pico match |
| $S_{pico}, S_{wbp}$ | the two (provably identical) calibration operators |
| $\gamma$ | post-merge Frobenius-norm rescaling factor |

### 3.2 Pico (prior work)
$$s_j=\frac{\sigma_j^2}{\sum_k\sigma_k^2},\qquad \alpha_j=\frac{1}{1+(T-1)s_j},\qquad S_{pico}=I+U\,\text{diag}(\alpha-1)\,U^\top$$
$$\tilde B_t = S_{pico}B_t,\quad \Delta\tilde W_t=\tilde B_tA_t,\quad \gamma=\frac{\tfrac1T\sum_t\|B_tA_t\|_F}{\|\Delta W_{calib}\|_F}$$

### 3.3 WBP (this thesis)
$$\lambda=\frac{T-1}{\text{Tr}(G)},\qquad K=\left(\tfrac1\lambda I_{Tr}+G\right)^{-1},\qquad S_{wbp}=I-B_{all}KB_{all}^\top$$
$$\tilde B_t = B_t - B_{all}\big[K(B_{all}^\top B_t)\big]\quad\text{(apply right-to-left, never form }S_{wbp}\text{ explicitly)}$$

### 3.4 Equivalence theorem — **proven exactly, confirmed numerically**
$S_{wbp}=S_{pico}$ as operators, for $\lambda=(T-1)/\text{Tr}(G)$. Not an approximation. Verified to
~$10^{-15}$–$10^{-16}$ relative error on random matrices in float64, in both $Tr\ll d_{out}$ and
$Tr>d_{out}$ regimes. **Edge case:** $T=1 \Rightarrow \lambda=0 \Rightarrow$ the $1/\lambda$ term in
$K$ is undefined — guard explicitly (`if T==1: return B_t unchanged`).

### 3.5 The $\beta$-WBP family (exploratory, not yet validated)
$\lambda_\beta=\beta\cdot\lambda$. $\beta=1$ is exact Pico. Sweeping $\beta$ is **in scope** (E5);
a principled, ideally data-free selection rule for $\beta$ is **not** (deferred, E6).

### 3.6 Complexity
Both Pico (SVD) and WBP (Woodbury) are $O(d_{out}\cdot(Tr)^2)$ asymptotically — **same Big-O, do not
claim otherwise.** WBP's real advantages: (a) avoids the naive $O(d_{out}^3)$ direct-inversion
alternative, (b) pure GEMMs parallelize far better on GPU than iterative SVD.

---

## 4. Verified Facts (treat as settled)

- Pico's restated math in `thesis-maths.md` matches the source paper term-for-term.
- The Pico↔WBP equivalence theorem is correctly derived and numerically confirmed — see §3.4.
- The Woodbury substitution (§3.3) is a standard, correctly-applied identity — re-derived and
  checked independently during verification.
- Pico's own empirical premises (asymmetry between $A$/$B$, low effective rank of $B$) are backed
  by the source paper's Table 4 and Figure 3 — not just asserted.
- **Not yet verified:** anything about WBP's real-world wall-clock speed or downstream accuracy —
  those require running E1–E5 (see `experiments.md`), they have not been measured yet.

---

## 5. Current Scope

**In scope (commit effort here):** Experiments E0–E5.
- **Phase 1 (Verification):**
  - E0: Subspace overlap verification on real trained adapters.
- **Phase 2 (Algorithm & Equivalence):**
  - E1: Operator-level equivalence between WBP and Pico on real adapters.
- **Phase 3 (Benchmarking & Computational Scaling):**
  - E2: Downstream accuracy parity evaluated using **both Task Arithmetic and TIES** merging (No Preconditioning vs. Pico vs. WBP).
  - E3: Wall-clock/memory scaling vs. $d_{out}$ (synthetic, no training needed).
  - E4: Scaling vs. $T$ and rank $r$.
- **Phase 4 (Generalization):**
  - E5: Exploratory $\beta$ sweep (no tuning protocol, no full baseline table).

**Explicitly out of scope** (do not silently expand into these without being asked):
- Re-implementing DARE, DELLA, KnOTS, or Core Space as baselines.
- Cross-backbone transfer experiments.
- A formal $\beta$-selection/tuning protocol.
- Conditioning stress tests / formal edge-case test suite beyond the basic $T=1$ guard.

If asked to "also do X" where X is one of the above, flag that it's outside current scope per
`experiments.md` rather than just doing it — the thesis framing depends on this boundary being
explicit and intentional, not creeping.

---

## 6. File Inventory

### Reference Documents (read-only)

| File | Contents | Status |
|---|---|---|
| `docs/Crowded_in_B-Space.pdf` | Source paper (Pico) | Reference, read-only |
| `docs/thesis-maths.md` | Full Pico + WBP derivations | **Verified line-by-line** |
| `docs/experiments.md` | E1–E5 protocols (+ E6–E13 deferred) | Plan |
| `docs/thesis-outline.md` | Chapter-by-chapter thesis structure | Drafted |
| `AGENT.md` (this file) | Onboarding/context summary | Living document |

### Source Code (`src/`) — **implemented and tested**

| File | Function | Status |
|---|---|---|
| `src/__init__.py` | Package marker | Done |
| `src/pico.py` | `merge_pico(B_list, A_list)` | ✅ Implemented |
| `src/wbp.py` | `merge_wbp(B_list, A_list, beta=1.0)` | ✅ Implemented |
| `src/utils.py` | `frobenius_norm_low_rank()`, `compute_gamma()` | ✅ Implemented |

Both `merge_pico` and `merge_wbp` return `(B_merged, A_merged)` as **separate tensors**
(`B_merged`: shape `(d_out, T*r)`, `A_merged`: shape `(T*r, d_in)`).
The final dense update is `B_merged @ A_merged`.

### Tests (`tests/`)

| File | What it checks | Status |
|---|---|---|
| `tests/test_equivalence.py` | Pico == WBP on random float64 matrices (T=4, d_out=1024, r=16) | ✅ Passing, max diff < 1e-12 |

### Experiment Scripts (`experiments/`) — **all scripts implemented**

#### Adapter Training — Lab RTX 6000

| File | Platform | Status |
|---|---|---|
| `experiments/adapter_training/AGENT_PROMPT.md` | — | Reference |
| `experiments/adapter_training/README.md` | — | Reference |
| `experiments/adapter_training/train_adapter.py` | **Lab RTX 6000 (CUDA)** | ✅ Written, not yet run |

`train_adapter.py` key details:
- Trains `unsloth/Meta-Llama-3.1-8B` with plain bf16 LoRA (`r=16`, `lora_alpha=16`, targets `q_proj`/`v_proj`).
- Accepts `--domain {math,coding,finance,medical}`. Run four times sequentially.
- Uses FlashAttention-2 (with graceful fallback), `paged_adamw_8bit`, gradient checkpointing.
- **Pushes trained adapter directly to HuggingFace Hub** after local save.
  - Username: `mml2024003`
  - Repo name: `mml2024003/Meta-Llama-3.1-8B_{domain}` (e.g. `mml2024003/Meta-Llama-3.1-8B_math`)
  - Visibility: private by default (`HF_REPO_PRIVATE = True` at top of file)
  - Requires `HF_TOKEN` env var with **write** permissions.
- Saves `adapters/{domain}/adapter_meta.json` locally and uploads it to the Hub repo.

#### E1 — Mac Mini M4 (CPU)

| File | Platform | Status |
|---|---|---|
| `experiments/e1_equivalence/AGENT_PROMPT.md` | — | Reference |
| `experiments/e1_equivalence/README.md` | — | Reference |
| `experiments/e1_equivalence/run_e1.py` | **Mac Mini M4 (CPU)** | ✅ Written, syntax verified |

`run_e1.py` key details:
- Loads real trained adapter weights directly from disk (`safetensors` or `.bin`) — **does NOT load the base model**.
- Builds `layer_map[layer_key] = {"B": [B_1..B_T], "A": [A_1..A_T]}` from adapter state dicts.
- Runs `merge_pico` and `merge_wbp(beta=1.0)` per layer, computes Frobenius relative error.
- Also verifies the T=1 edge-case guard.
- Pass criterion: `max_rel_error < 1e-4` (flag if `> 1e-4`, fail if `> 1e-2`).
- Writes `results/e1/results.json`.
- **Prerequisite:** Adapters must be in `./adapters/{math,coding,finance,medical}/`.

Run command:
```bash
cd /Users/demid/thesis && source .venv/bin/activate
PYTHONPATH=. python experiments/e1_equivalence/run_e1.py \
  --adapters_dir ./adapters --output_dir ./results/e1 --dtype float64
```

#### E2 — Lab RTX 6000 (CUDA)

| File | Platform | Status |
|---|---|---|
| `experiments/e2_accuracy/AGENT_PROMPT.md` | — | Reference |
| `experiments/e2_accuracy/README.md` | — | Reference |
| `experiments/e2_accuracy/run_e2.py` | **Lab RTX 6000** | ✅ Written, not yet run |
| `experiments/e2_accuracy/evaluate.py` | **Lab RTX 6000** | ✅ Written, not yet run |

`run_e2.py` key details:
- Loads base model **once**; patches/restores weights in-place for each calibration mode.
- Three calibration modes: `no_cal` (plain TA average), `pico`, `wbp_beta1`.
- Evaluates all 4 benchmarks via `evaluate.py`: GSM8K, HumanEval (subprocess sandbox), FinQA, MedMCQA.
- Inference batch size ≤ 4 to stay within the ~8 GB VRAM headroom on 8B model.
- Writes `results/e2/results.json`.
- **Prerequisite:** Adapters downloaded from Hub to `./adapters/`.

Run command:
```bash
PYTHONPATH=. python experiments/e2_accuracy/run_e2.py \
  --adapters_dir ./adapters --base_model unsloth/Meta-Llama-3.1-8B \
  --output_dir ./results/e2 --dtype bfloat16 --device cuda --seed 42
```

#### E3 — Lab RTX 6000 (CUDA)

| File | Platform | Status |
|---|---|---|
| `experiments/e3_scaling/AGENT_PROMPT.md` | — | Reference |
| `experiments/e3_scaling/README.md` | — | Reference |
| `experiments/e3_scaling/run_e3.py` | **Lab RTX 6000** | ✅ Written, not yet run |

`run_e3.py` key details:
- Synthetic only — no adapters or base model needed.
- Sweeps `d_out ∈ {512, 1024, 2048, 4096, 8192}`, T=4, r=16, 5 seeds × 20 timed iterations.
- Inline calibration functions (no `src/` import) to avoid Python overhead in timing measurements.
- CUDA-synchronized before stopping clock. Reports mean ± std over all seeds.
- Generates `results/e3/timing_plot.png` (log-log) and `results/e3/memory_plot.png`.
- **Bundle with E4-timing in the same lab GPU session** (takes < 30 min total).

Run command:
```bash
python experiments/e3_scaling/run_e3.py
```

#### E4 — Mac Mini (equivalence) + Lab RTX 6000 (timing)

| File | Platform | Status |
|---|---|---|
| `experiments/e4_tr_scaling/AGENT_PROMPT.md` | — | Reference |
| `experiments/e4_tr_scaling/README.md` | — | Reference |
| `experiments/e4_tr_scaling/run_e4_equivalence.py` | **Mac Mini M4 (CPU)** | ✅ **EXECUTED — all passed** |
| `experiments/e4_tr_scaling/run_e4_timing.py` | **Lab RTX 6000** | ✅ Written, not yet run |

`run_e4_equivalence.py` — **already run and passed on Mac Mini:**
- Grid: T ∈ {2,3,4,5,6} × r ∈ {8,16,32,64} = 20 combinations, 5 seeds each.
- All 20 cells passed at ~1e-16 relative error (float64, CPU).
- Results: `results/e4/equivalence/results.json` (`all_passed: true`)
- Heatmap: `results/e4/equivalence/equivalence_heatmap.png`

`run_e4_timing.py` — CUDA only, bundle with E3 on the lab GPU.

#### E5 — Lab RTX 6000 (CUDA)

| File | Platform | Status |
|---|---|---|
| `experiments/e5_beta_sweep/AGENT_PROMPT.md` | — | Reference |
| `experiments/e5_beta_sweep/README.md` | — | Reference |
| `experiments/e5_beta_sweep/run_e5.py` | **Lab RTX 6000** | ✅ Written, not yet run |

`run_e5.py` key details:
- β sweep: `[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]` — WBP only, Pico and no-cal overlaid from E2 results.
- Loads model once; patches/restores for each β.
- Auto-interprets curve shape at end (peak / flat / monotone) — see inline comments.
- **Must not claim:** that β≠1 generalises to other models, or that the data-free property is preserved.
- Requires `--e2_results_json ./results/e2/results.json` for baseline overlay.
- Generates `results/e5/beta_sweep_avg.png` and `results/e5/beta_sweep_per_benchmark.png`.

Run command:
```bash
PYTHONPATH=. python experiments/e5_beta_sweep/run_e5.py \
  --adapters_dir ./adapters --output_dir ./results/e5 \
  --e2_results_json ./results/e2/results.json --dtype bfloat16 --device cuda --seed 42
```

### Shared Experiment Infrastructure

| File | Contents |
|---|---|
| `experiments/SHARED_CONTEXT.md` | Math context for sub-agents (do not edit) |
| `experiments/README.md` | Master README: hardware matrix, execution order, logging rules |

### Python Environment

- Venv: `/Users/demid/thesis/.venv` (Python 3.13)
- Installed on Mac Mini: `torch==2.12.1`, `safetensors`, `numpy`, `matplotlib`
- Lab machine additionally needs: `transformers peft trl datasets accelerate huggingface_hub bitsandbytes`

### Results (what exists so far)

| Path | Contents | Status |
|---|---|---|
| `results/e4/equivalence/results.json` | E4 equivalence grid results | ✅ Done — all_passed: true |
| `results/e4/equivalence/equivalence_heatmap.png` | Log-scale heatmap | ✅ Generated |
| `results/e1/` | E1 on real adapters | ✅ Done — passed: true (max_rel_error=2.04e-15) |
| `results/e2/` | E2 accuracy parity | ⏳ Waiting for lab GPU session |
| `results/e3/` | E3 timing | ⏳ Waiting for lab GPU session |
| `results/e5/` | E5 β sweep | ⏳ Waiting for E2 results |

---

## 7. Conventions

- Match the source paper's notation exactly when writing code or docs (`B_all`, `T`, `r`, not
  renamed variables) — makes cross-referencing the thesis text to code painless.
- Report all timing numbers with: hardware, precision (fp32/bf16), warmup-excluded, GPU-synchronized
  before stopping the clock, mean ± std over ≥5 seeds.
- Any accuracy number must specify: benchmark, metric (exact-match/pass@k/etc.), eval sample count.
- When citing the source paper's own baseline numbers (DARE/DELLA/KnOTS/Core Space) anywhere, label
  them explicitly as "reported in [paper], not reproduced here" — never present them in the same
  table row format as self-run numbers without that caveat.

---

## 8. Status / Next Steps

### Completed ✅

- [x] Implement `src/pico.py` — `merge_pico(B_list, A_list)` with T=1 guard.
- [x] Implement `src/wbp.py` — `merge_wbp(B_list, A_list, beta=1.0)` with T=1 guard.
- [x] Implement `src/utils.py` — shared `frobenius_norm_low_rank()` and `compute_gamma()`.
- [x] Write `tests/test_equivalence.py` — passes, max diff < 1e-12 on random float64 matrices.
- [x] Write all experiment scripts: `train_adapter.py`, `run_e1.py`, `run_e2.py`, `evaluate.py`,
      `run_e3.py`, `run_e4_equivalence.py`, `run_e4_timing.py`, `run_e5.py`.
- [x] **Run E4-equivalence on Mac Mini** — all 20 (T, r) cells passed at ~1e-16 relative error.
      Results: `results/e4/equivalence/results.json`

### Pending — requires Lab GPU

- [x] **Lab GPU Session 0 (~16–20h):** Train all 4 adapters sequentially.
  (Completed using Unsloth script on Lab GPU).

- [x] **Copy adapters to Mac Mini** after training:
  (Completed).

- [x] **Run E1** — verify equivalence on real adapter weights:
  Ran successfully on Lab GPU (`max_rel_error=2.04e-15`).

- [ ] **Lab GPU Session 1 (~30–45 min):** Bundle E3 + E4-timing.
  ```bash
  python experiments/e3_scaling/run_e3.py
  python experiments/e4_tr_scaling/run_e4_timing.py
  ```

- [ ] **Lab GPU Session 2 (~2–4h):** Run E2 accuracy parity.
  ```bash
  PYTHONPATH=. python experiments/e2_accuracy/run_e2.py \
    --adapters_dir ./adapters --output_dir ./results/e2 --dtype bfloat16 --device cuda
  ```

- [ ] **Lab GPU Session 3 (~8–16h):** Run E5 β sweep.
  ```bash
  PYTHONPATH=. python experiments/e5_beta_sweep/run_e5.py \
    --adapters_dir ./adapters --output_dir ./results/e5 \
    --e2_results_json ./results/e2/results.json --dtype bfloat16 --device cuda
  ```

- [ ] **Update this file** once each experiment completes — record actual timing numbers,
      hardware, and whether the pass/fail criteria were met.

---

## 9. Instructions if you are an AI agent picking this up

1. Read this file fully before doing anything.
2. Do not re-derive or question the math in §3 — it's verified; if something looks inconsistent,
   say so explicitly and point to the exact line, don't silently "fix" it by changing the formula.
3. Do not implement or run anything from the "out of scope" list in §5 unless the user explicitly
   asks for it in that session.
4. When implementing, use the pseudocode in §3.2/§3.3 as the spec — variable names should trace
   back to the notation table in §3.1.
5. When reporting any result back to the user, follow the logging requirements in §7 — incomplete
   provenance (no seed, no precision, no hardware) is not acceptable for this project's results.
6. If you complete an item in §8's checklist, update it in this file rather than leaving it stale.
7. **All experiment scripts are already written** (see §6). Do not rewrite them unless the user
   explicitly asks for changes. The correct next action is to *run* existing scripts, not regenerate them.
8. **Hardware routing is strict — do not run CUDA scripts on Mac Mini:**
   | Script | Platform |
   |---|---|
   | `run_e1.py`, `run_e4_equivalence.py` | Mac Mini M4 (CPU) |
   | `train_adapter.py`, `run_e2.py`, `run_e3.py`, `run_e4_timing.py`, `run_e5.py` | Lab RTX 6000 (CUDA) |
9. **HuggingFace identity:** username `mml2024003`. Adapters are pushed automatically by
   `train_adapter_unsloth.py` to `mml2024003/Meta-Llama-3.1-8B_{domain}`. To pull them to Mac Mini:
   ```bash
   huggingface-cli download mml2024003/Meta-Llama-3.1-8B_math    --local-dir ./adapters/math
   huggingface-cli download mml2024003/Meta-Llama-3.1-8B_coding  --local-dir ./adapters/coding
   huggingface-cli download mml2024003/Meta-Llama-3.1-8B_finance --local-dir ./adapters/finance
   huggingface-cli download mml2024003/Meta-Llama-3.1-8B_medical --local-dir ./adapters/medical
   ```

---

## 10. Implementation State Snapshot

> Updated at the end of each working session. Any agent can resume from here without
> needing the conversation history.

**Last updated:** 2026-06-25

### Verified results (already on disk)

| Item | Location | Detail |
|---|---|---|
| E4 equivalence grid | `results/e4/equivalence/results.json` | `all_passed: true` — all 20 (T,r) cells at ~1e-16 relative error, float64, CPU |
| E4 heatmap | `results/e4/equivalence/equivalence_heatmap.png` | Log-scale heatmap across T×r grid |
| E1 equivalence on adapters | `results/e1/results.json` | `passed: true` — max_rel_error ~2.04e-15, float64, 64 layers |

### Scripts written but not yet run (waiting on hardware)

| Script | Blocked on |
|---|---|

| `run_e2.py` + `evaluate.py` | Trained adapters + Lab GPU |
| `run_e3.py` | Lab GPU (bundle with E4-timing) |
| `run_e4_timing.py` | Lab GPU (bundle with E3) |
| `run_e5.py` | `results/e2/results.json` + Lab GPU |

### Key design decisions (do not reverse without explicit user instruction)

| Decision | Rationale |
|---|---|
| `merge_pico` / `merge_wbp` return `(B_merged, A_merged)` separately, not dense `delta_W` | Keeps adapters low-rank as long as possible; E2/E5 form `B_merged @ A_merged` themselves |
| `delta_W` computed in float32, cast to bf16 only when writing to model | Prevents precision loss in merging arithmetic |
| E2 and E5 patch/restore model weights in-place, loading the base model once | Avoids reloading 16 GB three times per run |
| E3 and E4-timing use inline calibration functions, not `src/` imports | Excludes Python import overhead from CUDA timing measurements |
| `train_adapter.py` saves locally first, then pushes to Hub | Hub push failure does not lose the trained adapter |
| HumanEval code execution runs in a subprocess, never `exec()` | Security: prevents arbitrary code from running in the main Python process |
| Hub repos are private by default (`HF_REPO_PRIVATE = True`) | Can be changed to `False` at the top of `train_adapter.py` |

