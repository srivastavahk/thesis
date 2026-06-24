# README: Adapter Training (Lab RTX 6000)

> **Platform:** Lab RTX 6000 (24 GB VRAM, CUDA)
> **Purpose:** Fine-tune 4 domain-specific LoRA adapters on `meta-llama/Llama-3.1-8B` (plain bf16 LoRA). These are the prerequisite for E1, E2, E4, E5.

---

## Prerequisites

- SSH access to the lab machine with RTX 6000
- HuggingFace account with access to `meta-llama/Llama-3.1-8B` (request access at [huggingface.co/meta-llama](https://huggingface.co/meta-llama/Llama-3.1-8B) if not already granted — approval is usually fast)
- HuggingFace User Access Token ([hf.co/settings/tokens](https://huggingface.co/settings/tokens))

---

## Step-by-Step Instructions

### Step 1: Transfer the training script to the lab machine

```bash
# From Mac Mini
scp /Users/demid/thesis/experiments/adapter_training/train_adapter.py \
    <user>@<lab-ip>:~/thesis/experiments/adapter_training/
```

### Step 2: On the lab machine — set up environment

```bash
# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Install dependencies (if not already present)
pip install transformers peft trl datasets accelerate safetensors huggingface_hub

# Set your HuggingFace token (do this once per session)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### Step 3: Create output directories

```bash
mkdir -p ~/thesis/adapters ~/thesis/checkpoints
```

### Step 4: Run training for each domain

Run all four domains sequentially. Each takes ~3–5 hours on the RTX 6000. Use `nohup` or `tmux` so SSH disconnects don't kill the process.

```bash
cd ~/thesis

# Option A: tmux (recommended — lets you detach and reconnect)
tmux new -s training
# Inside tmux, run one domain at a time:

PYTHONPATH=. python experiments/adapter_training/train_adapter.py --domain math
PYTHONPATH=. python experiments/adapter_training/train_adapter.py --domain coding
PYTHONPATH=. python experiments/adapter_training/train_adapter.py --domain finance
PYTHONPATH=. python experiments/adapter_training/train_adapter.py --domain medical

# Detach from tmux with Ctrl+B, D — reattach later with: tmux attach -t training

# Option B: nohup (fire and forget, one at a time)
nohup python experiments/adapter_training/train_adapter.py --domain math \
    > ~/thesis/logs/train_math.log 2>&1 &
```

**Important:** All four domains can be run in sequence in a single lab session (~16–20 hours total). Book the GPU accordingly.

### Step 5: Monitor training progress

```bash
# Watch the log in real time (if using nohup)
tail -f ~/thesis/logs/train_math.log

# Check GPU utilization
watch -n 5 nvidia-smi
```

Expected training loss should be decreasing and reach ~1.0–1.8 by step 3000.

### Step 6: Verify adapter output

After each domain completes, check:
```bash
ls ~/thesis/adapters/math/
# Expected: adapter_config.json  adapter_model.safetensors  adapter_meta.json

cat ~/thesis/adapters/math/adapter_meta.json
# Check: final_train_loss < 2.0
```

### Step 7: Copy adapters back to Mac Mini

```bash
# From Mac Mini
rsync -avz <user>@<lab-ip>:~/thesis/adapters/ /Users/demid/thesis/adapters/
```

Verify the local structure:
```bash
ls /Users/demid/thesis/adapters/
# Should show: math/  coding/  finance/  medical/
```

---

## Estimated Time per Domain (RTX 6000)

| Domain | Dataset Download | Training (3000 steps, bf16) | Total |
|---|---|---|---|
| Math | ~5 min | ~3–4 h | ~4 h |
| Coding | ~5 min | ~3–4 h | ~4 h |
| Finance | ~3 min | ~3–4 h | ~4 h |
| Medical | ~3 min | ~3–4 h | ~4 h |
| **All 4 (sequential)** | | | **~16–18 h** |

Book a single uninterrupted lab GPU slot of ~18–20 hours, or spread across 2 days (2 domains per session).

---

## VRAM Budget

| Component | VRAM |
|---|---|
| Llama-3.1-8B weights (bf16) | ~16 GB |
| LoRA adapter params (r=16) | ~0.3 GB |
| Optimizer states (AdamW) | ~1–2 GB |
| Activations (batch=4, seq=512) | ~1–2 GB |
| **Total** | **~18–20 GB** |

Fits within 24 GB. If you hit OOM: reduce `per_device_train_batch_size` from 4 → 2 first.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `CUDA out of memory` | Reduce `per_device_train_batch_size` to 2 |
| `401 Unauthorized` on model download | Check `HF_TOKEN` is set; verify Llama-3.1-8B access is approved on HuggingFace |
| Dataset not found | Check dataset name on HuggingFace Hub; some splits may vary |
| Training loss is NaN | Reduce `learning_rate` to `1e-4`; check for degenerate data samples |
| SSH drops mid-training | Use `tmux` — reconnect with `tmux attach -t training` |

---

## What to Do Next

Once all 4 adapters are in `/Users/demid/thesis/adapters/`, proceed to:
- **E1:** `experiments/e1_equivalence/README.md` — runs on Mac Mini, no GPU needed
