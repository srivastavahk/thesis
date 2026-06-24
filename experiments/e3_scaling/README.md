# README: E3 — Wall-Clock & Memory Scaling vs. d_out

> **Platform:** Lab RTX 6000 (24 GB VRAM, CUDA required)
> **Prerequisite:** None — uses synthetic matrices only

---

## What This Experiment Does

Sweeps `d_out ∈ {512, 1024, 2048, 4096, 8192}` with T=4, r=16 and times both Pico's SVD and WBP's Woodbury path on the GPU. Produces timing plots with error bars over 5 seeds.

**Expected result:** Both scale roughly O(d_out) for fixed Tr=64. The win shows up as a constant-factor gap between the two curves, not a different slope. WBP should be faster due to GEMM vs. iterative SVD.

---

## Step-by-Step Instructions

### Step 1: Transfer the script to the lab machine

From your Mac Mini:
```bash
scp /Users/demid/thesis/experiments/e3_scaling/run_e3.py <user>@<lab-machine-ip>:~/thesis/experiments/e3_scaling/
```

Or use any file transfer method available (lab NFS, rsync, USB).

### Step 2: On the lab machine — verify CUDA environment

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"
# Expected: something like 'RTX 6000' or 'NVIDIA RTX 6000 Ada'
```

If torch is not installed:
```bash
pip install torch matplotlib
```

### Step 3: Create the results directory

```bash
mkdir -p ~/thesis/results/e3
```

### Step 4: Run the script

```bash
cd ~/thesis
python experiments/e3_scaling/run_e3.py
```

**Expected runtime:** 5–15 minutes total (all d_out values, 5 seeds, warmup included).

Watch for the log output confirming each `d_out` value is being processed:
```
INFO: d_out=512, seed=42 — Pico: 0.234ms, WBP: 0.178ms
INFO: d_out=1024, seed=42 — Pico: 0.456ms, WBP: 0.312ms
...
```

### Step 5: Copy results back to Mac Mini

```bash
# On lab machine
scp ~/thesis/results/e3/results.json <user>@<mac-ip>:/Users/demid/thesis/results/e3/
scp ~/thesis/results/e3/timing_plot.png <user>@<mac-ip>:/Users/demid/thesis/results/e3/
scp ~/thesis/results/e3/memory_plot.png <user>@<mac-ip>:/Users/demid/thesis/results/e3/
```

### Step 6: Interpret the results

Open `results/e3/results.json` and check the `speedup` field for each `d_out`:

| Speedup value | Interpretation |
|---|---|
| `> 1.5x` consistently | ✅ Strong hardware efficiency argument for WBP |
| `1.1x – 1.5x` | ✅ Modest but real improvement — state it as "constant-factor speedup due to GEMM utilization" |
| `< 1.0x` (WBP slower) | ⚠️ Unexpected — check if SVD is using accelerated CUDA path; may happen at very small d_out |

**Important thesis framing:** Both methods have the same asymptotic complexity O(d_out · (Tr)²). The gap is about hardware utilization (GEMM vs. iterative SVD), NOT a better big-O. State this explicitly in your thesis chapter.

### Step 7: Inspect the plots

- `timing_plot.png` — log-log timing plot; both curves should be roughly linear (≈ same slope); constant vertical offset shows WBP advantage
- `memory_plot.png` — WBP should use less peak memory at large d_out since it never forms the d_out × d_out covariance C explicitly

---

## Recording Requirements (from AGENT.md §7)

When reporting E3 numbers in your thesis, always state:
- GPU model: RTX 6000 Ada (24 GB)
- Precision: float32
- Warmup iterations excluded: yes (5 warmup, 20 timed)
- GPU-synchronized: yes (`torch.cuda.synchronize()` before stopping clock)
- Seeds: 5 (42–46), reporting mean ± std

---

## What to Do Next

- Bundle E4 timing sub-part into the same GPU session (it's the same script extended over T and r values).
- Then proceed to E2 (accuracy parity) in a separate, dedicated GPU booking.
