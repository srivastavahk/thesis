"""
Adapter Training with Unsloth — Lab RTX 6000 (24 GB VRAM, CUDA)
================================================================
Trains a domain-specific LoRA adapter on meta-llama/Llama-3.1-8B (bf16) using Unsloth.
Run ONE domain at a time; all four domains are trained sequentially.

Usage (run from project root ~/thesis/):
-----------------------------------------
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

    PYTHONPATH=. python experiments/adapter_training/train_adapter_unsloth.py --domain math
    PYTHONPATH=. python experiments/adapter_training/train_adapter_unsloth.py --domain coding
    PYTHONPATH=. python experiments/adapter_training/train_adapter_unsloth.py --domain finance
    PYTHONPATH=. python experiments/adapter_training/train_adapter_unsloth.py --domain medical

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
# Fix CUDA allocator fragmentation — prevents OOM when reserved-but-unallocated
# memory blocks are large but none is contiguous enough for a new allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# HuggingFace Hub push configuration
# ---------------------------------------------------------------------------
HF_USERNAME     = "mml2024003"
HF_REPO_PRIVATE = False   # set to False to make the repo public

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

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B"
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0      # Unsloth is optimized for dropout = 0
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
MAX_SEQ_LENGTH = 512
MAX_STEPS = 3000
LEARNING_RATE = 2e-4
# Memory budget on RTX 6000 (24 GB):
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8   # effective batch = 16
WARMUP_STEPS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_messages(row: dict, domain: str) -> dict:
    """Convert a raw dataset row into a chat messages list."""
    if domain == "medical":
        # Concatenate input + instruction for medical domain
        inp = (row.get("input", "") + " " + row.get("instruction", "")).strip()
    else:
        cfg = DOMAIN_CONFIG[domain]
        inp = row.get(cfg["input_field"], "")
    
    out = row.get(DOMAIN_CONFIG[domain]["output_field"], "")
    return {
        "messages": [
            {"role": "user", "content": inp},
            {"role": "assistant", "content": out}
        ]
    }


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(domain: str):
    import gc
    from datasets import load_dataset
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only

    # Explicitly clear CUDA memory before starting
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = DOMAIN_CONFIG[domain]
    log.info("=" * 60)
    log.info("Training adapter (Unsloth): domain=%s", domain)
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
    # 1. Load base model and tokenizer via Unsloth
    # ------------------------------------------------------------------
    log.info("Loading base model (Unsloth) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = BASE_MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = torch.bfloat16,
        load_in_4bit = False,  # we are doing bf16 LoRA
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply official Llama-3.1 chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    
    # ------------------------------------------------------------------
    # 2. Apply LoRA via Unsloth
    # ------------------------------------------------------------------
    log.info("Applying LoRA (Unsloth) ...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = LORA_R,
        target_modules = LORA_TARGET_MODULES,
        lora_alpha = LORA_ALPHA,
        lora_dropout = LORA_DROPOUT,
        bias = "none",
        use_gradient_checkpointing = "unsloth", # Use Unsloth's optimized VRAM efficient checkpointing
        random_state = SEED,
        use_rslora = False,
        loftq_config = None,
    )
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 3. Load and prepare dataset
    # ------------------------------------------------------------------
    log.info("Loading dataset: %s ...", cfg["dataset"])
    raw_dataset = load_dataset(cfg["dataset"], split=cfg["split"])

    # Sample n_samples rows deterministically
    n = min(cfg["n_samples"], len(raw_dataset))
    raw_dataset = raw_dataset.shuffle(seed=SEED).select(range(n))
    log.info("Dataset size after sampling: %d", len(raw_dataset))

    # Format to chat messages and apply template
    def format_chat(batch):
        # Convert batched dict of columns to list of row dicts
        rows = [{k: batch[k][i] for k in batch} for i in range(len(batch[list(batch.keys())[0]]))]
        messages_list = [format_messages(row, domain)["messages"] for row in rows]
        
        # Apply chat template
        texts = tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": texts}

    log.info("Applying chat templates ...")
    tokenized_dataset = raw_dataset.map(
        format_chat,
        batched=True,
        batch_size=256,
        remove_columns=raw_dataset.column_names,
        desc="Formatting chat",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    # ------------------------------------------------------------------
    # 4. Training arguments (SFTTrainer)
    # ------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=checkpoint_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        optim="paged_adamw_8bit",          # saves ~2 GB VRAM vs. adamw_torch
        bf16=True,
        logging_steps=50,
        save_steps=1000,
        save_total_limit=3,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
        seed=SEED,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    # ------------------------------------------------------------------
    # 5. Train (auto-resumes from latest checkpoint if one exists)
    # ------------------------------------------------------------------
    import glob
    existing_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")))
    resume_from_checkpoint = bool(existing_checkpoints)
    if resume_from_checkpoint:
        latest_ckpt = existing_checkpoints[-1]
        log.info("Resuming training from checkpoint: %s", latest_ckpt)
    else:
        log.info("No checkpoint found — starting training from step 0.")

    t0 = time.perf_counter()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        args=training_args,
    )

    # Optimize learning: mask the instruction so the loss is only calculated on the assistant's response.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    elapsed = time.perf_counter() - t0
    log.info("Training complete in %.1f minutes.", elapsed / 60)

    final_loss = train_result.training_loss
    log.info("Final training loss: %.4f", final_loss)

    # ------------------------------------------------------------------
    # 6. Save adapter locally first (safe fallback)
    # ------------------------------------------------------------------
    log.info("Saving adapter locally to %s ...", output_adapter_dir)
    model.save_pretrained(output_adapter_dir)
    tokenizer.save_pretrained(output_adapter_dir)
    log.info("Local save complete.")

    # ------------------------------------------------------------------
    # 7. Push adapter to HuggingFace Hub
    # ------------------------------------------------------------------
    base_model_name = BASE_MODEL.split("/")[-1]          # "Llama-3.1-8B"
    hub_repo_id     = f"{HF_USERNAME}/{base_model_name}_{domain}"  # "mml2024003/Llama-3.1-8B_math"

    log.info("Pushing adapter to HuggingFace Hub: %s ...", hub_repo_id)
    try:
        model.push_to_hub(
            hub_repo_id,
            private=HF_REPO_PRIVATE,
            token=os.environ.get("HF_TOKEN"),
            commit_message=f"Add {domain} LoRA adapter (Unsloth, {MAX_STEPS} steps, bf16, r={LORA_R})",
        )
        tokenizer.push_to_hub(
            hub_repo_id,
            private=HF_REPO_PRIVATE,
            token=os.environ.get("HF_TOKEN"),
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
    # 8. Save metadata JSON (local + upload to Hub)
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
        "unsloth": True,
    }
    meta_path = os.path.join(output_adapter_dir, "adapter_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata saved to %s", meta_path)

    if hub_url is not None:
        try:
            from huggingface_hub import upload_file
            upload_file(
                path_or_fileobj=meta_path,
                path_in_repo="adapter_meta.json",
                repo_id=hub_repo_id,
                commit_message="Add adapter metadata JSON",
                token=os.environ.get("HF_TOKEN"),
            )
            log.info("adapter_meta.json uploaded to Hub.")
        except Exception as exc:
            log.warning("Could not upload adapter_meta.json to Hub: %s", exc)

    log.info("Done. Adapter for domain='%s' available at: %s", domain, hub_url or output_adapter_dir)

    # Clean up memory after training finishes
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a domain-specific LoRA adapter on Llama-3.1-8B using Unsloth."
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
