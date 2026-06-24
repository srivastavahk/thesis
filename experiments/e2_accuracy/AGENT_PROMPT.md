# Agent Prompt: E2 — Downstream Accuracy Parity

> **Platform:** Lab RTX 6000 (24 GB VRAM, CUDA)
> **Your task:** Write a complete merge-and-evaluate pipeline that applies Task Arithmetic with three calibration modes (no-cal, Pico, WBP) and evaluates on four domain benchmarks.

---

## Context

Read `../SHARED_CONTEXT.md` for full mathematical background, notation, and existing code.

**E2 Objective:** Show that WBP-merged models match Pico-merged models on downstream benchmark accuracy end-to-end — not just at the operator level. Both should also beat "no calibration" (plain Task Arithmetic).

**Hardware note:** This runs on the Lab RTX 6000 (24 GB VRAM). `meta-llama/Llama-3.1-8B` at bf16 uses ~16 GB, leaving ~8 GB headroom for activations. Keep inference batch size ≤ 4 to stay within VRAM.

---

## What to Implement

### Files to create:

1. `experiments/e2_accuracy/run_e2.py` — main pipeline
2. `experiments/e2_accuracy/evaluate.py` — benchmark evaluation functions

---

### `run_e2.py`

#### CLI arguments

```bash
python run_e2.py \
  --adapters_dir /path/to/adapters \
  --base_model meta-llama/Llama-3.1-8B \
  --output_dir /path/to/results/e2 \
  --dtype bfloat16 \
  --device cuda \
  --seed 42
```

#### Algorithm

**Step 1: Load base model weights**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(base_model)
```

**Step 2: Load all 4 adapter weight dicts from disk**
Load `adapter_model.safetensors` for each of {math, coding, finance, medical} using `safetensors.torch.load_file()`. Do NOT use `PeftModel.from_pretrained()` — load raw state dicts so we can apply custom calibration.

**Step 3: Extract B and A tensors per layer**
Parse adapter keys like:
```
base_model.model.model.layers.{i}.self_attn.{q_proj|v_proj}.lora_{B|A}.weight
```
Build a per-layer dict:
```python
layer_adapters = {
  "layers.0.self_attn.q_proj": {
    "B": [B_math, B_coding, B_finance, B_medical],
    "A": [A_math, A_coding, A_finance, A_medical]
  },
  ...
}
```

**Step 4: For each calibration mode — compute merged weight deltas**

Implement three modes:

```python
def apply_no_cal(B_list, A_list):
    """Plain Task Arithmetic: simple average."""
    T = len(B_list)
    delta_W = sum(B @ A for B, A in zip(B_list, A_list)) / T
    return delta_W  # (d_out, d_in)

def apply_pico(B_list, A_list):
    """Pico calibration then average."""
    from src.pico import merge_pico
    B_merged, A_merged = merge_pico(B_list, A_list)
    return B_merged @ A_merged  # (d_out, d_in)

def apply_wbp(B_list, A_list, beta=1.0):
    """WBP calibration then average."""
    from src.wbp import merge_wbp
    B_merged, A_merged = merge_wbp(B_list, A_list, beta=beta)
    return B_merged @ A_merged  # (d_out, d_in)
```

**Step 5: Build merged model**

For each calibration mode:
1. Start from base model state dict.
2. For each layer key, compute `delta_W` using the mode's function.
3. Add `delta_W` to the corresponding frozen weight in the base model:
   `merged_weight = base_weight + delta_W`
4. Load into a copy of the model using `load_state_dict()`.

**Step 6: Evaluate on all 4 benchmarks**

Call `evaluate.py` functions. See `evaluate.py` spec below.

**Step 7: Write results JSON**
```json
{
  "experiment": "E2",
  "hardware": "RTX 6000 24GB",
  "base_model": "meta-llama/Llama-3.1-8B",
  "dtype": "bfloat16",
  "T": 4,
  "seed": 42,
  "results": {
    "no_cal": {
      "gsm8k_exact_match": 0.312,
      "humaneval_pass_at_1": 0.104,
      "finqa_exact_match": 0.198,
      "medmcqa_accuracy": 0.421,
      "average": 0.259
    },
    "pico": { ... },
    "wbp_beta1": { ... }
  }
}
```

---

### `evaluate.py`

Implement one evaluation function per benchmark. Each function takes `(model, tokenizer, device)` and returns a float score.

#### `eval_gsm8k(model, tokenizer, device, n_samples=None) -> float`
- Load `gsm8k` dataset, `main` split, `test` set (~1319 examples, use all).
- Prompt format: `"Question: {question}\nAnswer:"` — greedy decode, extract the final number.
- Metric: exact match on the numeric answer (strip commas, dollar signs; compare as floats).
- Return fraction correct.

#### `eval_humaneval(model, tokenizer, device) -> float`
- Load `openai_humaneval` dataset, `test` split (164 problems).
- Use pass@1 with greedy decoding.
- Execute the generated code in a sandboxed subprocess with a timeout of 10 seconds.
- Return fraction of problems where all provided test cases pass.
- **Safety note:** Only run the auto-generated code in a subprocess, never with `exec()` in the main process.

#### `eval_finqa(model, tokenizer, device, n_samples=500) -> float`
- Load `dreamerdeo/finqa` dataset, `test` split. Sample `n_samples` examples (seed-controlled).
- Prompt: `"Context: {context}\nQuestion: {question}\nAnswer:"` — greedy decode.
- Metric: exact match on the answer string.
- Return fraction correct.

#### `eval_medmcqa(model, tokenizer, device, n_samples=1000) -> float`
- Load `openlifescienceai/medmcqa` dataset, `validation` split. Sample `n_samples` examples.
- Format as multiple choice: `"Question: {question}\nA. {opa}\nB. {opb}\nC. {opc}\nD. {opd}\nAnswer:"`.
- Greedy decode, check if model outputs the correct letter (A/B/C/D) or maps to `cop` field (0/1/2/3 → A/B/C/D).
- Return accuracy.

---

## Important Implementation Notes

1. **Do NOT load the base model 3 times.** Keep one base model instance and patch weights for each eval mode, then restore.
2. **Disable gradient computation:** `with torch.no_grad():` for all inference.
3. **Max new tokens:** Use 256 for GSM8K/FinQA/MedMCQA; 512 for HumanEval.
4. **Batch inference:** Use batch size **4** (not 8) for inference — the 8B model at bf16 leaves ~8 GB headroom, which is sufficient for small batches. Pad left (causal LM).
5. **Numeric stability:** Compute delta_W in float32 even if the model runs in bf16; cast only when writing to model weights.

---

## Dependencies

```bash
pip install transformers peft datasets safetensors accelerate torch evaluate
```

---

## Deliverables

- `experiments/e2_accuracy/run_e2.py`
- `experiments/e2_accuracy/evaluate.py`
