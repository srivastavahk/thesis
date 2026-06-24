# README: E1 — Operator-level Equivalence on Real Adapters

> **Platform:** Mac Mini M4 (CPU)
> **Prerequisite:** 4 trained adapters in `/Users/demid/thesis/adapters/` (from adapter_training step)

---

## What This Experiment Does

Loads real trained LoRA adapter weights and verifies that Pico and WBP produce numerically identical calibrated B matrices on every layer of the model. This is the validation gate — if it passes, we trust the WBP implementation is correct on real weights.

---

## Step-by-Step Instructions

### Step 1: Install dependencies

```bash
cd /Users/demid/thesis
source .venv/bin/activate
pip install safetensors
```

### Step 2: Verify adapter files are present

```bash
ls /Users/demid/thesis/adapters/
# Should show: math/  coding/  finance/  medical/

ls /Users/demid/thesis/adapters/math/
# Should show: adapter_config.json  adapter_model.safetensors  adapter_meta.json
```

### Step 3: Run the equivalence check

```bash
cd /Users/demid/thesis
source .venv/bin/activate

PYTHONPATH=. python experiments/e1_equivalence/run_e1.py \
  --adapters_dir ./adapters \
  --base_model meta-llama/Llama-3.1-8B \
  --output_dir ./results/e1 \
  --dtype float64 \
  --seed 42
```

**Expected runtime:** 1–5 minutes on Mac Mini M4 (no model loading, just adapter weight math).

### Step 4: Inspect results

```bash
cat /Users/demid/thesis/results/e1/results.json
```

Look for:
- `"passed": true`
- `"max_rel_error"` should be in the range `1e-6` to `1e-5` (float32 adapter weights create slightly more error than the float64 synthetic test)

### Step 5: Interpret the output

| `max_rel_error` | Interpretation |
|---|---|
| `< 1e-5` | ✅ Excellent — implementation is correct |
| `1e-5` to `1e-4` | ✅ Acceptable — within float32 precision, note it |
| `> 1e-4` | ⚠️ Potential conditioning issue — check the `per_layer_results` to find which layers are problematic; look for near-singular G matrices |
| `> 1e-2` | ❌ Bug — likely an implementation error; do not proceed to E2 |

### Step 6: Check the T=1 guard

The script automatically tests the T=1 edge case. Look for:
```
INFO: T=1 guard: OK
```
in the log output.

---

## Understanding the per_layer_results

The JSON contains per-layer error metrics. High error on specific layers usually means:
- That layer's `G = B_all.T @ B_all` is poorly conditioned (large condition number)
- The Woodbury inversion accumulates more floating point error there
- This is expected for layers where all adapters learned near-identical features

---

## What to Do Next

- If E1 passes → proceed to E3 (synthetic timing, no adapters needed) and book lab GPU for E2.
- If E1 fails → debug before running anything else. The math is proven correct; any failure is an implementation or numerical precision issue.
