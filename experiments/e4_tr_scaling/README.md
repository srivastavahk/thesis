# README: E4 — Scaling vs. T and Rank r

> **Platform:** Two sub-parts:
> - **Equivalence check:** Mac Mini M4 (CPU) — run locally
> - **Timing:** Lab RTX 6000 — bundle with E3 in same GPU session

---

## What This Experiment Does

Sweeps T ∈ {2,3,4,5,6} and r ∈ {8,16,32,64} to confirm:
1. **Equivalence sub-part:** Pico and WBP produce identical results at every (T, r) grid point.
2. **Timing sub-part:** How the Pico–WBP speedup ratio changes as the merge pool grows.

---

## Sub-Part A: Equivalence Check (Mac Mini)

### Step 1: Run equivalence script locally

```bash
cd /Users/demid/thesis
source .venv/bin/activate
pip install matplotlib

PYTHONPATH=. python experiments/e4_tr_scaling/run_e4_equivalence.py
```

**Expected runtime:** 5–15 minutes on Mac Mini M4 CPU.

### Step 2: Inspect results

```bash
cat results/e4/equivalence/results.json | python -m json.tool | grep -E '"all_passed"|"max_rel_error"'
```

Expected: `"all_passed": true` and all `max_rel_error` values < 1e-5.

### Step 3: View the heatmap

Open `results/e4/equivalence/equivalence_heatmap.png`.

All cells should be dark blue (very low log10 error ≈ -14 to -15). Any bright yellow/red cell at a specific (T, r) indicates a conditioning problem worth investigating.

---

## Sub-Part B: Timing (Lab RTX 6000)

### Step 1: Transfer the script (bundle with E3)

```bash
# From Mac Mini, alongside E3 files
scp /Users/demid/thesis/experiments/e4_tr_scaling/run_e4_timing.py <user>@<lab-ip>:~/thesis/experiments/e4_tr_scaling/
mkdir -p ~/thesis/results/e4/timing  # on lab machine
```

### Step 2: Run on lab machine

```bash
cd ~/thesis
python experiments/e4_tr_scaling/run_e4_timing.py
```

**Expected runtime:** 15–30 minutes (5 T-values × 4 r-values × 5 seeds × warmup).

### Step 3: Copy results back

```bash
scp -r <user>@<lab-ip>:~/thesis/results/e4/ /Users/demid/thesis/results/e4/
```

### Step 4: Interpret the speedup heatmap

Open `results/e4/timing/speedup_heatmap.png`.

| Speedup pattern | Interpretation |
|---|---|
| Speedup increases with higher T·r | ✅ As SVD resolves more singular values iteratively, WBP's fixed GEMM advantage grows |
| Speedup relatively flat | ✅ Acceptable; means the constant-factor claim holds uniformly |
| WBP slower at small Tr | ⚠️ Normal — at Tr=16 (T=2, r=8), SVD overhead is tiny; WBP's advantage shows at larger Tr |

---

## What to Do Next

After E4: proceed to E5 (β sweep). Book a separate GPU session.
