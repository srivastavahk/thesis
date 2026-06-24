# README: E5 — Decoupled-λ Sweep (β-WBP), Exploratory

> **Platform:** Lab RTX 6000 (24 GB VRAM, CUDA required)
> **Prerequisite:** 4 trained adapters + E2 results JSON

---

## What This Experiment Does

Sweeps β ∈ {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0} in the WBP formula `λ_β = β·(T-1)/Tr(G)` and plots overall benchmark accuracy vs. β. The β=1 point is the exact Pico match — this experiment explores whether other points in the family can do better.

---

## Step-by-Step Instructions

### Step 1: Ensure E2 is complete first

E5 overlays its results on E2's baselines. You need `results/e2/results.json` to exist.

### Step 2: Transfer scripts to lab machine

```bash
rsync -avz /Users/demid/thesis/experiments/e5_beta_sweep/ <user>@<lab-ip>:~/thesis/experiments/e5_beta_sweep/
# Also copy evaluate.py from E2 if not already there
cp ~/thesis/experiments/e2_accuracy/evaluate.py ~/thesis/experiments/e5_beta_sweep/
```

Also transfer adapters and src/ if not already on the lab machine from E2.

### Step 3: Book a dedicated GPU slot

E5 runs 7 × E2 evaluations. Expected runtime: **8–16 hours**. Book accordingly. Do not run this in a shared screen session without a tmux or nohup wrapper.

### Step 4: Run the sweep

```bash
cd ~/thesis
PYTHONPATH=. nohup python experiments/e5_beta_sweep/run_e5.py \
  --adapters_dir ./adapters \
  --base_model meta-llama/Llama-3.1-8B \
  --output_dir ./results/e5 \
  --dtype bfloat16 \
  --device cuda \
  --seed 42 \
  --e2_results_json ./results/e2/results.json \
  > ./results/e5/run_log.txt 2>&1 &
echo $!  # Save the PID
```

Using `nohup` protects against SSH disconnects.

Monitor progress:
```bash
tail -f ~/thesis/results/e5/run_log.txt
```

### Step 5: Copy results back

```bash
scp -r <user>@<lab-ip>:~/thesis/results/e5/ /Users/demid/thesis/results/e5/
```

### Step 6: Interpret the results

Open `results/e5/beta_sweep_avg.png`. Read the log summary printed at the end of the run:

```bash
grep "FINDING:" ~/thesis/results/e5/run_log.txt
```

#### Interpretation guide

| Curve shape | What it means | How to write it in the thesis |
|---|---|---|
| Peak at β ≠ 1, above Pico | The family contains better calibrations than exact-Pico | "WBP-β={x} outperforms Pico by y% avg; however, this β was chosen by benchmark accuracy, not a data-free rule. A principled selection method is the natural next step (Future Work §7.x)." |
| Flat curve (< 1% variation) | Pico's β=1 was already near-optimal | "The calibration surface is flat in this regime, suggesting Pico's λ is a robust choice. Exploring β adds no practical value here." |
| Monotonically decreasing with β | Over-shrinkage beyond β=1 | "Stronger suppression degrades accuracy, confirming Pico's choice of λ is not over-conservative." |
| Monotonically increasing with β | Under-shrinkage at β=1 | "Weaker suppression progressively improves accuracy, suggesting Pico's λ over-suppresses shared directions at this scale." |

**All four outcomes are valid, publishable findings. Do not try to force a particular result.**

---

## Recording Requirements

When reporting E5 in thesis:
- β values tested: {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0}
- Hardware: RTX 6000 24 GB
- Precision: bfloat16
- Seed: 42
- Note explicitly: "β=1.0 coincides with exact Pico equivalent (mathematically proven)"
- Note explicitly: "β chosen by validation accuracy is not data-free; see Future Work"
