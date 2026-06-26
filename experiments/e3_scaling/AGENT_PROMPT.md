# Agent Prompt:# E3 — Wall-clock & Memory Scaling vs d_out (Phase 3)

> **Platform:** Lab RTX 6000 (24 GB VRAM, CUDA)
> **Your task:** Write a self-contained Python timing benchmark script that sweeps `d_out` and compares wall-clock time and peak GPU memory between Pico (thin SVD) and WBP (Gram + Woodbury) using synthetic matrices.

---

## Context

**E3 Objective:** Quantify the computational advantage of WBP over Pico. Show that avoiding the SVD yields massive wall-clock speedups for large $d_{out}$ and large $T$, while maintaining linear scaling. benchmark script that sweeps `d_out` and compares wall-clock time and peak GPU memory between Pico (thin SVD) and WBP (Gram + Woodbury) using synthetic matrices.

Read `../SHARED_CONTEXT.md` for full mathematical background, notation, and existing code.

**E3 Objective:** Measure actual speedup, not asserted speedup. This generates the timing plot that supports the thesis's efficiency claim.

**This experiment uses ONLY synthetic random matrices — no trained adapters are needed.**

The script must run on CUDA. MPS (Apple Silicon) results are not acceptable as primary thesis numbers.

---

## What to Implement

### File: `experiments/e3_scaling/run_e3.py`

#### Configuration (hardcoded at top of script, not CLI args)

```python
D_OUT_VALUES = [512, 1024, 2048, 4096, 8192]
T = 4
R = 16          # Tr = 64
WARMUP_ITERS = 5
TIMED_ITERS = 20
DTYPE = torch.float32
SEEDS = [42, 43, 44, 45, 46]   # 5 seeds → report mean ± std
RESULTS_DIR = "./results/e3"
```

#### Structure

The script should implement these two functions and a timing harness:

```python
def pico_calibrate(B_all: torch.Tensor, T: int) -> torch.Tensor:
    """Thin SVD path. Returns calibrated B_all (d_out, T*r)."""
    U, S, Vh = torch.linalg.svd(B_all, full_matrices=False)
    S_sq = S ** 2
    s = S_sq / S_sq.sum()
    alpha = 1.0 / (1.0 + (T - 1) * s)
    alpha_minus_1 = (alpha - 1.0).unsqueeze(0)  # broadcast
    return B_all + U @ (alpha_minus_1.T * (U.T @ B_all))

def wbp_calibrate(B_all: torch.Tensor, T: int) -> torch.Tensor:
    """Woodbury path. Returns calibrated B_all (d_out, T*r)."""
    G = B_all.T @ B_all
    lam = (T - 1) / torch.trace(G)
    Tr = B_all.shape[1]
    K = torch.linalg.inv(torch.eye(Tr, device=B_all.device, dtype=B_all.dtype) / lam + G)
    return B_all - B_all @ (K @ (B_all.T @ B_all))
```

#### Timing harness (for each `d_out` and each `seed`)

```python
def time_fn(fn, *args, warmup=WARMUP_ITERS, iters=TIMED_ITERS):
    # Warmup (excluded from timing)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()   # MANDATORY before stopping clock
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    return times  # list of per-iteration wall-clock times in seconds
```

#### Memory measurement

Use `torch.cuda.max_memory_allocated()` reset before each call:
```python
torch.cuda.reset_peak_memory_stats()
fn(*args)
torch.cuda.synchronize()
peak_mem_bytes = torch.cuda.max_memory_allocated()
```

#### For each `d_out` and `seed`:
1. Generate `B_all = torch.randn(d_out, T*R, dtype=DTYPE, device='cuda')` with `torch.manual_seed(seed)`.
2. Time `pico_calibrate` and `wbp_calibrate`.
3. Measure peak memory for each.

#### Output JSON structure:

```json
{
  "experiment": "E3",
  "hardware": "RTX 6000 24GB",
  "dtype": "float32",
  "T": 4, "r": 16, "Tr": 64,
  "warmup_iters": 5,
  "timed_iters": 20,
  "seeds": [42, 43, 44, 45, 46],
  "results": [
    {
      "d_out": 512,
      "pico": {
        "mean_time_s": 0.000123,
        "std_time_s": 0.000005,
        "peak_mem_bytes": 1234567
      },
      "wbp": {
        "mean_time_s": 0.000089,
        "std_time_s": 0.000003,
        "peak_mem_bytes": 987654
      },
      "speedup": 1.38
    },
    ...
  ]
}
```

#### Plot generation

After writing JSON, generate a log-log plot using `matplotlib`:
- X axis: `d_out` (log scale)
- Y axis: wall-clock time in milliseconds (log scale)
- Two lines: Pico (blue, circles) and WBP (orange, squares)
- Error bars: ± 1 std
- Title: "Calibration Wall-Clock Time vs. d_out (T=4, r=16, RTX 6000)"
- Save to `./results/e3/timing_plot.png` at 300 DPI

Also generate a memory plot: peak GPU memory (MB) vs. `d_out` on linear axes.

---

## Dependencies

```bash
pip install torch matplotlib
```

The `src/` package is NOT needed — the calibration functions are reimplemented inline in this script for standalone timing (importing from src would add Python overhead).

---

## Deliverables

- `experiments/e3_scaling/run_e3.py` — the standalone timing + plotting script
- No README needed — see `experiments/e3_scaling/README.md` (already provided)
