# README: E2 — Downstream Accuracy Parity

> **Platform:** Lab RTX 6000 (24 GB VRAM, CUDA required)
> **Prerequisite:** 4 trained adapters (from adapter_training step)

---

## What This Experiment Does

Merges 4 domain adapters using 3 modes (no calibration, Pico, WBP) via Task Arithmetic, then evaluates on 4 benchmarks: GSM8K (math), HumanEval (coding), FinQA (finance), MedMCQA (medical).

**Expected result:** Pico and WBP rows should be statistically indistinguishable. Both should beat no-calibration.

---

## Step-by-Step Instructions

### Step 1: Transfer scripts and adapters to lab machine

```bash
# From Mac Mini
rsync -avz /Users/demid/thesis/experiments/e2_accuracy/ <user>@<lab-ip>:~/thesis/experiments/e2_accuracy/
rsync -avz /Users/demid/thesis/adapters/ <user>@<lab-ip>:~/thesis/adapters/
rsync -avz /Users/demid/thesis/src/ <user>@<lab-ip>:~/thesis/src/
```

### Step 2: On lab machine — verify environment

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory // 1e9, 'GB')"

pip install transformers peft datasets safetensors accelerate evaluate
```

### Step 3: Create results directory

```bash
mkdir -p ~/thesis/results/e2
```

### Step 4: Run the evaluation pipeline

```bash
cd ~/thesis
PYTHONPATH=. python experiments/e2_accuracy/run_e2.py \
  --adapters_dir ./adapters \
  --base_model meta-llama/Llama-3.1-8B \
  --output_dir ./results/e2 \
  --dtype bfloat16 \
  --device cuda \
  --seed 42
```

**Expected runtime:** 2–4 hours (base model load + 3 merge modes × 4 benchmarks).

Monitor progress from the log output — each benchmark eval should print progress every ~100 examples.

### Step 5: Copy results back

```bash
scp ~/thesis/results/e2/results.json <user>@<mac-ip>:/Users/demid/thesis/results/e2/
```

### Step 6: Interpret the results

Open `results/e2/results.json`. Build a comparison table:

| Method | GSM8K | HumanEval | FinQA | MedMCQA | Avg |
|---|---|---|---|---|---|
| No Calibration | - | - | - | - | - |
| Pico | - | - | - | - | - |
| WBP (β=1) | - | - | - | - | - |

Fill in the values from `results.json`.

#### What to expect and claim:
- **Pico ≈ WBP:** Differences should be within ±0.5% on each benchmark. If larger, recheck that the same `gamma` rescaling was applied and the same random seed was used.
- **Pico > No-Cal:** Expect a gain of 2–8% average, consistent with Tang & Yang's reported improvement for Task Arithmetic.
- **Do NOT** put DARE/DELLA/KnOTS in this table as self-run rows. If you want to mention them, cite the source paper's numbers in a footnote labeled "reported in [Tang & Yang], not reproduced here."

#### If Pico ≠ WBP (difference > 1%):
1. Check that E1 passed with `max_rel_error < 1e-4`. If E1 failed, this is expected.
2. Check that `gamma` computation is identical in both paths (it should be, since both use `src/utils.py`).
3. Check that the same weight patching logic was applied (a common bug: forgetting to add delta_W to base weights, or adding it twice).

---

## Recording Requirements

When reporting E2 in the thesis:
- Benchmark: GSM8K (exact match, 1319 test examples), HumanEval (pass@1, 164 problems), FinQA (exact match, 500 test examples), MedMCQA (accuracy, 1000 val examples)
- Hardware: RTX 6000 24 GB
- Precision: bfloat16
- Seed: 42

---

## What to Do Next

After E2: proceed to E5 (β sweep, same pipeline, same GPU). Book a separate session.
