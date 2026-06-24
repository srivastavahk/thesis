# Agent Prompt: Adapter Training (Lab GPU)

> **Platform:** Lab RTX 6000 (24 GB VRAM, CUDA)
> **Your task:** Write a self-contained Python script to fine-tune `meta-llama/Llama-3.1-8B` with plain LoRA (bf16) for ONE domain. The script is a standard CLI Python script, not a notebook. Four separate sequential runs of this script will train adapters for four domains.

---

## Context

Read `../SHARED_CONTEXT.md` for full mathematical background and conventions.

We need T=4 LoRA adapters (one per domain: math, coding, finance, medical) trained on a shared base model `meta-llama/Llama-3.1-8B`. These adapters are the prerequisite for experiments E1, E2, E4, and E5.

The RTX 6000 has 24 GB VRAM. `Llama-3.1-8B` at bf16 uses ~16 GB for weights, leaving ~8 GB for optimizer states and activations — sufficient for LoRA training with moderate batch size.

---

## What to Implement

Write a single Python script: `train_adapter.py`

The script must accept a `--domain` argument: one of `{math, coding, finance, medical}`.

### Required behavior

1. **Load base model:** `meta-llama/Llama-3.1-8B` in bf16.
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       "meta-llama/Llama-3.1-8B",
       torch_dtype=torch.bfloat16,
       device_map="cuda",
   )
   ```
2. **Apply LoRA** via `peft`:
   - `r=16`, `lora_alpha=32`, `lora_dropout=0.05`
   - Target modules: `q_proj`, `v_proj`
   - `bias="none"`, `task_type="CAUSAL_LM"`
3. **Load dataset** based on `--domain`:
   - `math` → `meta-math/MetaMathQA`, field `output` as target, `query` as input. Sample 10,000 rows.
   - `coding` → `theblackcat102/evol-codealpaca-v1`, field `output` as target, `instruction` as input. Sample 10,000 rows.
   - `finance` → `gbharti/finance-alpaca`, field `output` as target, `instruction` as input. Sample 10,000 rows.
   - `medical` → `medalpaca/medical_meadow_medqa`, field `output` as target, `input` + `instruction` concatenated as input. Sample 10,000 rows.
4. **Format** each example as:
   ```
   ### Instruction:\n{input}\n\n### Response:\n{output}
   ```
5. **Tokenize** with max_length=512, truncation=True, padding=False. Use `DataCollatorForSeq2Seq` with `pad_to_multiple_of=8`.
6. **Train** with `trl.SFTTrainer` or `transformers.Trainer`:
   - `max_steps=3000`, `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`
   - `learning_rate=2e-4`, `lr_scheduler_type="cosine"`, `warmup_steps=100`
   - `optim="adamw_torch"`, `bf16=True`
   - Save checkpoint every 1000 steps to `./checkpoints/{domain}/`
7. **Save adapter** to `./adapters/{domain}/` using `model.save_pretrained()`.
8. **Save a metadata JSON** to `./adapters/{domain}/adapter_meta.json`:
   ```json
   {
     "domain": "<domain>",
     "base_model": "meta-llama/Llama-3.1-8B",
     "lora_r": 16,
     "lora_alpha": 32,
     "target_modules": ["q_proj", "v_proj"],
     "training_precision": "bf16",
     "max_steps": 3000,
     "dataset": "<dataset name>",
     "dataset_samples": 10000,
     "final_train_loss": <float>
   }
   ```

### Required imports
```python
import torch, json, logging, argparse, os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
```

### HuggingFace authentication
The model is gated. At the top of the script, read the token from an environment variable:
```python
from huggingface_hub import login
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
```

---

## Output Structure

```
./adapters/
  math/
    adapter_config.json
    adapter_model.safetensors
    adapter_meta.json
  coding/
    ...
  finance/
    ...
  medical/
    ...
```

Adapters are saved directly on the lab machine's filesystem. They will be copied to the Mac Mini via `scp` or `rsync` after all four are trained.

---

## Constraints

- Plain bf16 LoRA (no quantization). The RTX 6000's 24 GB VRAM accommodates the 8B model (~16 GB) with sufficient headroom for LoRA optimizer states.
- If VRAM is tight, reduce `per_device_train_batch_size` to 2 before any other change.
- Set `torch.manual_seed(42)` and `PYTHONHASHSEED=42` at the top for reproducibility.
- Use `logging` (not `print`) for all progress output, so `nohup` logs are clean.
- Include a comment block at the top of the script showing how to run it for each domain.

---

## Deliverables

- `experiments/adapter_training/train_adapter.py` — the CLI training script
