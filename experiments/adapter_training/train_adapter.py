"""
Adapter Training — Lab RTX 6000 (24 GB VRAM, CUDA)
====================================================
Trains a domain-specific LoRA adapter on meta-llama/Llama-3.1-8B (bf16).
Run ONE domain at a time; all four domains are trained sequentially.

Usage (run from project root ~/thesis/):
-----------------------------------------
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

    PYTHONPATH=. python experiments/adapter_training/train_adapter.py --domain math
    PYTHONPATH=. python experiments/adapter_training/train_adapter.py --domain coding
    PYTHONPATH=. python experiments/adapter_training/train_adapter.py --domain finance
    PYTHONPATH=. python experiments/adapter_training/train_adapter.py --domain medical

    # Or in one shot with tmux (recommended):
    # tmux new -s training
    # Then paste all four lines sequentially.

Output per domain
-----------------
    adapters/{domain}/adapter_config.json
    adapters/{domain}/adapter_model.safetensors
    adapters/{domain}/adapter_meta.json        ← includes hub_repo_id
    checkpoints/{domain}/                       ← intermediate checkpoints every 1000 steps

HuggingFace Hub
---------------
    Repo name : mml2024003/Llama-3.1-8B_{domain}
    Visibility: private by default (change HF_REPO_PRIVATE below to False for public)
    The same HF_TOKEN used to download Llama-3.1-8B is used to push.
"""

import argparse
import json
import logging
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Reproducibility — set before any torch / HF operations
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ---------------------------------------------------------------------------
# HuggingFace Hub push configuration
# ---------------------------------------------------------------------------
HF_USERNAME     = "mml2024003"
HF_REPO_PRIVATE = True   # set to False to make the repo public

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace authentication (Llama is gated)
# ---------------------------------------------------------------------------
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    log.info("Logged in to HuggingFace Hub.")
else:
    log.warning(
        "HF_TOKEN not set. If meta-llama/Llama-3.1-8B download fails with 401, "
        "export HF_TOKEN=<your token> and retry."
    )

# ---------------------------------------------------------------------------
# Domain configuration
# ---------------------------------------------------------------------------
DOMAIN_CONFIG = {
    "math": {
        "dataset": "meta-math/MetaMathQA",
        "split": "train",
        "input_field": "query",
        "output_field": "output",
        "n_samples": 10_000,
    },
    "coding": {
        "dataset": "theblackcat102/evol-codealpaca-v1",
        "split": "train",
        "input_field": "instruction",
        "output_field": "output",
        "n_samples": 10_000,
    },
    "finance": {
        "dataset": "gbharti/finance-alpaca",
        "split": "train",
        "input_field": "instruction",
        "output_field": "output",
        "n_samples": 10_000,
    },
    "medical": {
        "dataset": "medalpaca/medical_meadow_medqa",
        "split": "train",
        "input_field": None,          # special: concat input + instruction
        "output_field": "output",
        "n_samples": 10_000,
    },
}

BASE_MODEL = "meta-llama/Llama-3.1-8B"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
MAX_SEQ_LENGTH = 512
MAX_STEPS = 3000
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4   # effective batch = 16
WARMUP_STEPS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_example(row: dict, domain: str) -> str:
    """Convert a raw dataset row into the instruction/response template."""
    if domain == "medical":
        # Concatenate input + instruction for medical domain
        inp = (row.get("input", "") + " " + row.get("instruction", "")).strip()
    else:
        cfg = DOMAIN_CONFIG[domain]
        inp = row.get(cfg["input_field"], "")
    out = row.get(DOMAIN_CONFIG[domain]["output_field"], "")
    return f"### Instruction:\n{inp}\n\n### Response:\n{out}"


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(domain: str):
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        TrainingArguments,
        Trainer,
    )

    cfg = DOMAIN_CONFIG[domain]
    log.info("=" * 60)
    log.info("Training adapter: domain=%s", domain)
    log.info("  base_model  : %s", BASE_MODEL)
    log.info("  dataset     : %s", cfg["dataset"])
    log.info("  n_samples   : %d", cfg["n_samples"])
    log.info("  lora_r      : %d   lora_alpha: %d", LORA_R, LORA_ALPHA)
    log.info("  max_steps   : %d", MAX_STEPS)
    log.info("=" * 60)

    output_adapter_dir = f"./adapters/{domain}"
    checkpoint_dir = f"./checkpoints/{domain}"
    os.makedirs(output_adapter_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------------
    log.info("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 2. Load base model in bf16 with FlashAttention-2 if available
    # ------------------------------------------------------------------
    log.info("Loading base model (this may take a few minutes) ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        )
        log.info("FlashAttention-2 enabled.")
    except (ValueError, ImportError) as e:
        log.warning("FlashAttention-2 not available (%s). Falling back to standard attention.", e)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

    model.config.use_cache = False  # required when using gradient checkpointing

    # ------------------------------------------------------------------
    # 3. Apply LoRA
    # ------------------------------------------------------------------
    log.info("Applying LoRA ...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 4. Load and prepare dataset
    # ------------------------------------------------------------------
    log.info("Loading dataset: %s ...", cfg["dataset"])
    raw_dataset = load_dataset(cfg["dataset"], split=cfg["split"])

    # Sample n_samples rows deterministically
    n = min(cfg["n_samples"], len(raw_dataset))
    raw_dataset = raw_dataset.shuffle(seed=SEED).select(range(n))
    log.info("Dataset size after sampling: %d", len(raw_dataset))

    # Format and tokenize
    def tokenize(batch):
        texts = [format_example(row, domain) for row in
                 [{k: batch[k][i] for k in batch} for i in range(len(batch[list(batch.keys())[0]]))]]
        tokenized = tokenizer(
            texts,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    log.info("Tokenizing dataset ...")
    tokenized_dataset = raw_dataset.map(
        tokenize,
        batched=True,
        batch_size=256,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    # ------------------------------------------------------------------
    # 5. Training arguments
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        optim="paged_adamw_8bit",          # saves ~2 GB VRAM vs. adamw_torch
        bf16=True,
        gradient_checkpointing=True,        # saves activation VRAM
        logging_steps=50,
        save_steps=1000,
        save_total_limit=3,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
        seed=SEED,
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    log.info("Starting training ...")
    t0 = time.perf_counter()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    elapsed = time.perf_counter() - t0
    log.info("Training complete in %.1f minutes.", elapsed / 60)

    final_loss = train_result.training_loss
    log.info("Final training loss: %.4f", final_loss)

    # ------------------------------------------------------------------
    # 7. Save adapter locally first (safe fallback)
    # ------------------------------------------------------------------
    log.info("Saving adapter locally to %s ...", output_adapter_dir)
    model.save_pretrained(output_adapter_dir)
    tokenizer.save_pretrained(output_adapter_dir)
    log.info("Local save complete.")

    # ------------------------------------------------------------------
    # 8. Push adapter to HuggingFace Hub
    #
    #    Repo name format: {HF_USERNAME}/{base_model_name}_{domain}
    #    e.g. mml2024003/Llama-3.1-8B_math
    #
    #    The base model name is the last component of BASE_MODEL
    #    (strips the org prefix, e.g. "meta-llama/Llama-3.1-8B" → "Llama-3.1-8B").
    # ------------------------------------------------------------------
    base_model_name = BASE_MODEL.split("/")[-1]          # "Llama-3.1-8B"
    hub_repo_id     = f"{HF_USERNAME}/{base_model_name}_{domain}"  # "mml2024003/Llama-3.1-8B_math"

    log.info("Pushing adapter to HuggingFace Hub: %s ...", hub_repo_id)
    try:
        model.push_to_hub(
            hub_repo_id,
            private=HF_REPO_PRIVATE,
            commit_message=f"Add {domain} LoRA adapter ({MAX_STEPS} steps, bf16, r={LORA_R})",
        )
        tokenizer.push_to_hub(
            hub_repo_id,
            private=HF_REPO_PRIVATE,
            commit_message="Add tokenizer",
        )
        hub_url = f"https://huggingface.co/{hub_repo_id}"
        log.info("Hub push successful: %s", hub_url)
    except Exception as exc:
        log.error(
            "Hub push FAILED: %s\n"
            "The adapter is still saved locally at %s — you can push manually with:\n"
            "  huggingface-cli upload %s %s .",
            exc, output_adapter_dir, hub_repo_id, output_adapter_dir,
        )
        hub_url = None

    # ------------------------------------------------------------------
    # 9. Save metadata JSON (local + upload to Hub)
    # ------------------------------------------------------------------
    meta = {
        "domain": domain,
        "base_model": BASE_MODEL,
        "hub_repo_id": hub_repo_id,
        "hub_url": hub_url,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "target_modules": LORA_TARGET_MODULES,
        "training_precision": "bf16",
        "max_steps": MAX_STEPS,
        "dataset": cfg["dataset"],
        "dataset_samples": n,
        "final_train_loss": round(final_loss, 6),
        "training_time_minutes": round(elapsed / 60, 1),
        "seed": SEED,
    }
    meta_path = os.path.join(output_adapter_dir, "adapter_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata saved to %s", meta_path)

    # Also upload the metadata file to the Hub repo
    if hub_url is not None:
        try:
            from huggingface_hub import upload_file
            upload_file(
                path_or_fileobj=meta_path,
                path_in_repo="adapter_meta.json",
                repo_id=hub_repo_id,
                commit_message="Add adapter metadata JSON",
            )
            log.info("adapter_meta.json uploaded to Hub.")
        except Exception as exc:
            log.warning("Could not upload adapter_meta.json to Hub: %s", exc)

    log.info("Done. Adapter for domain='%s' available at: %s", domain, hub_url or output_adapter_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a domain-specific LoRA adapter on Llama-3.1-8B."
    )
    parser.add_argument(
        "--domain",
        required=True,
        choices=list(DOMAIN_CONFIG.keys()),
        help="Domain to train: math | coding | finance | medical",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not torch.cuda.is_available():
        log.error(
            "CUDA is not available. This script must run on the Lab RTX 6000. "
            "Do NOT run on Mac Mini."
        )
        sys.exit(1)

    log.info("GPU: %s  VRAM: %.1f GB",
             torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / 1e9)

    train(args.domain)
