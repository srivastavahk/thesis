"""
E2 — Downstream Accuracy Parity (No-Cal / Pico / WBP)
=======================================================
Platform : Lab RTX 6000 (24 GB VRAM, CUDA)
Purpose  : Merge T=4 LoRA adapters with three calibration strategies and
           evaluate on GSM8K, HumanEval, FinQA, MedMCQA.

Usage
-----
    PYTHONPATH=/path/to/thesis python experiments/e2_accuracy/run_e2.py \
        --adapters_dir ./adapters \
        --base_model   unsloth/Meta-Llama-3.1-8B \
        --output_dir   ./results/e2 \
        --dtype        float16 \
        --device       cuda \
        --seed         42

IMPORTANT IMPLEMENTATION NOTES
--------------------------------
 - The base model is loaded ONCE. Weights are patched for each mode, evaluated,
   then restored — no need to reload 16 GB three times.
 - delta_W is computed in float32 even if the model runs in bf16 for
   numerical stability, then cast to bf16 when writing into the model.
 - All inference uses torch.no_grad(). Batch size is kept at ≤4 to fit
   within the ~8 GB headroom left after the 16 GB model weights.
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

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
# HuggingFace auth
# ---------------------------------------------------------------------------
from huggingface_hub import login as hf_login

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    hf_login(token=_hf_token)


# ---------------------------------------------------------------------------
# Adapter loading (same logic as run_e1.py)
# ---------------------------------------------------------------------------

def _load_adapter_state_dict(adapter_dir: Path) -> Dict[str, torch.Tensor]:
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file
        return load_file(str(safetensors_path))
    if bin_path.is_file():
        return torch.load(str(bin_path), map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def load_all_adapters(
    adapters_dir: Path,
) -> tuple[Dict[str, Dict[str, List[torch.Tensor]]], List[str]]:
    """
    Returns:
        layer_adapters : {layer_key: {"B": [B_t ...], "A": [A_t ...]}}
        domain_names   : ordered list of domain names
    """
    subdirs = sorted([p for p in adapters_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not subdirs:
        raise RuntimeError(f"No subdirs in {adapters_dir}")
    domain_names = [p.name for p in subdirs]
    log.info("Loading adapters for domains: %s", domain_names)

    per_domain: List[Dict[str, torch.Tensor]] = []
    for d in subdirs:
        sd = _load_adapter_state_dict(d)
        # Keep as float32 for stable delta_W computation
        sd = {k: v.float() for k, v in sd.items()}
        per_domain.append(sd)

    first = per_domain[0]
    b_keys = sorted(k for k in first if k.endswith("lora_B.weight"))

    layer_adapters: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    for bk in b_keys:
        ak = bk.replace("lora_B.weight", "lora_A.weight")
        layer_key = bk.replace(".lora_B.weight", "")
        if not all(bk in sd and ak in sd for sd in per_domain):
            log.warning("Skipping layer %s — missing in some adapters.", layer_key)
            continue
        layer_adapters[layer_key] = {
            "B": [sd[bk] for sd in per_domain],
            "A": [sd[ak] for sd in per_domain],
        }

    log.info("Built layer_adapters with %d layers, T=%d", len(layer_adapters), len(subdirs))
    return layer_adapters, domain_names


# ---------------------------------------------------------------------------
# Calibration modes — return delta_W per layer in float32
# ---------------------------------------------------------------------------

def apply_no_cal(B_list: List[torch.Tensor], A_list: List[torch.Tensor]) -> torch.Tensor:
    """Plain Task Arithmetic: simple average of ΔW_t."""
    T = len(B_list)
    return sum(B @ A for B, A in zip(B_list, A_list)) / T


def apply_pico(B_list: List[torch.Tensor], A_list: List[torch.Tensor]) -> torch.Tensor:
    """Pico calibration then dense delta."""
    from src.pico import merge_pico
    B_merged, A_merged = merge_pico(B_list, A_list)
    return B_merged @ A_merged


def apply_wbp(
    B_list: List[torch.Tensor],
    A_list: List[torch.Tensor],
    beta: float = 1.0,
) -> torch.Tensor:
    """WBP calibration then dense delta."""
    from src.wbp import merge_wbp
    B_merged, A_merged = merge_wbp(B_list, A_list, beta=beta)
    return B_merged @ A_merged


# ---------------------------------------------------------------------------
# Model weight patching / restoration
# ---------------------------------------------------------------------------

def _lora_key_to_base_key(layer_key: str) -> Optional[str]:
    """
    Convert a PEFT adapter key like:
        base_model.model.model.layers.0.self_attn.q_proj
    to the corresponding base model weight key:
        model.layers.0.self_attn.q_proj.weight
    Returns None if unable to map.
    """
    # Strip the leading "base_model.model." prefix added by PEFT
    key = layer_key
    for prefix in ("base_model.model.", "base_model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    return key + ".weight"


def patch_model_weights(
    model,
    layer_adapters: Dict[str, Dict[str, List[torch.Tensor]]],
    calibration_fn,
    target_dtype: torch.dtype,
    **fn_kwargs,
) -> Dict[str, torch.Tensor]:
    """
    For each LoRA layer:
      1. Compute delta_W using calibration_fn (in float32).
      2. Add delta_W to the base model's frozen weight.
      3. Stash the original weight so we can restore later.

    Returns a dict of {param_name: original_weight_tensor} for restoration.
    """
    original_weights: Dict[str, torch.Tensor] = {}
    state_dict = model.state_dict()

    with torch.no_grad():
        for layer_key, tensors in layer_adapters.items():
            base_key = _lora_key_to_base_key(layer_key)
            if base_key is None or base_key not in state_dict:
                log.debug("Skipping layer %s — base key %s not found.", layer_key, base_key)
                continue

            # Compute delta_W in float32 for numerical stability
            delta_W = calibration_fn(tensors["B"], tensors["A"], **fn_kwargs)  # (d_out, d_in) f32

            # Locate the parameter
            param = None
            for name, p in model.named_parameters():
                if name == base_key:
                    param = p
                    break
            if param is None:
                # Try via state_dict path (handles quantized / device_map scenarios)
                continue

            original_weights[base_key] = param.data.clone()
            param.data.add_(delta_W.to(param.dtype).to(param.device))

    return original_weights


def restore_model_weights(model, original_weights: Dict[str, torch.Tensor]):
    """Undo the weight patching done by patch_model_weights."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in original_weights:
                p.data.copy_(original_weights[name].to(p.device))


# ---------------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------------

def run_eval_suite(model, tokenizer, seed: int, scores: dict, save_cb) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate import eval_gsm8k, eval_humaneval, eval_finance, eval_medmcqa

    benchmarks = [
        ("gsm8k_exact_match", eval_gsm8k),
        ("finance_acc_norm", eval_finance),
        ("medmcqa_accuracy", eval_medmcqa),
        ("humaneval_pass_at_1", eval_humaneval),
    ]

    for b_name, b_func in benchmarks:
        if b_name not in scores:
            log.info("  Running %s ...", b_name)
            scores[b_name] = b_func(model, tokenizer, device=None, seed=seed)
            save_cb()
        else:
            log.info("  Skipping %s (already completed)", b_name)

    if all(b_name in scores for b_name, _ in benchmarks):
        scores["average"] = sum(scores[b_name] for b_name, _ in benchmarks) / len(benchmarks)
        save_cb()

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="E2 — Downstream accuracy parity (Pico vs WBP).")
    parser.add_argument("--adapters_dir", type=Path, default=Path("./adapters"))
    parser.add_argument("--base_model",   type=str,  default="unsloth/Meta-Llama-3.1-8B")
    parser.add_argument("--output_dir",   type=Path, default=Path("./results/e2"))
    parser.add_argument("--dtype",        type=str,  default="float16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device",       type=str,  default="cuda")
    parser.add_argument("--seed",         type=int,  default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            log.error("CUDA not available. This script must run on the Lab RTX 6000.")
            sys.exit(1)
        
        # Explicitly clear CUDA memory before starting
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model_dtype = dtype_map[args.dtype]

    log.info("=" * 60)
    log.info("E2 — Downstream accuracy parity")
    log.info("  base_model   : %s", args.base_model)
    log.info("  adapters_dir : %s", args.adapters_dir)
    log.info("  output_dir   : %s", args.output_dir)
    log.info("  dtype        : %s", args.dtype)
    log.info("  device       : %s", args.device)
    log.info("  seed         : %d", args.seed)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load adapters
    # ------------------------------------------------------------------
    layer_adapters, domain_names = load_all_adapters(args.adapters_dir)
    T = len(domain_names)

    # ------------------------------------------------------------------
    # 2. Load base model ONCE
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading base model %s (this takes ~3 min) ...", args.base_model)
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=model_dtype,
                device_map=args.device,
                attn_implementation=attn_impl,
            )
            log.info("Attention implementation: %s", attn_impl)
            break
        except (ValueError, ImportError, RuntimeError) as e:
            log.warning("attn_implementation='%s' unavailable (%s), trying next ...", attn_impl, e)
    else:
        raise RuntimeError("Could not load model with any attention implementation.")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info("Base model loaded.")

    # ------------------------------------------------------------------
    # 3. Evaluate each calibration mode
    # ------------------------------------------------------------------
    calibration_modes = {
        "no_cal":   (apply_no_cal,  {}),
        "pico":     (apply_pico,    {}),
        "wbp_beta1":(apply_wbp,     {"beta": 1.0}),
    }

    out_path = args.output_dir / "results.json"
    if out_path.exists():
        log.info("Found existing results at %s. Resuming...", out_path)
        with open(out_path, "r") as f:
            existing_data = json.load(f)
            all_results = existing_data.get("results", {})
    else:
        all_results = {}

    def save_progress():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "experiment":  "E2",
            "hardware":    "RTX 6000 24GB",
            "base_model":  args.base_model,
            "dtype":       args.dtype,
            "T":           T,
            "domain_names": domain_names,
            "seed":        args.seed,
            "results":     all_results,
        }
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    for mode_name, (fn, fn_kwargs) in calibration_modes.items():
        if mode_name not in all_results:
            all_results[mode_name] = {}

        expected_benchmarks = ["gsm8k_exact_match", "finance_acc_norm", "medmcqa_accuracy", "humaneval_pass_at_1"]
        if all(b in all_results[mode_name] for b in expected_benchmarks):
            log.info("Mode %s already fully completed. Skipping.", mode_name)
            continue

        log.info("-" * 50)
        log.info("Mode: %s", mode_name)
        t0 = time.perf_counter()

        # Patch weights
        originals = patch_model_weights(
            model, layer_adapters, fn, target_dtype=model_dtype, **fn_kwargs
        )
        log.info("  Weight patching complete. Running benchmarks ...")

        # Evaluate
        with torch.no_grad():
            scores = run_eval_suite(model, tokenizer, args.seed, all_results[mode_name], save_progress)

        # Restore weights
        restore_model_weights(model, originals)
        del originals
        torch.cuda.empty_cache()

        elapsed = time.perf_counter() - t0
        log.info("Mode %s complete in %.1f min. Average: %.4f",
                 mode_name, elapsed / 60, scores.get("average", 0.0))

    # ------------------------------------------------------------------
    # 4. Write results JSON
    # ------------------------------------------------------------------
    save_progress()
    log.info("Results successfully written to %s", out_path)

    # Summary table
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("%-12s  %-8s  %-8s  %-8s  %-8s  %-8s",
             "Mode", "GSM8K", "HumanEv", "Finance", "MedMCQA", "Avg")
    for mode, s in all_results.items():
        log.info("%-12s  %-8.4f  %-8.4f  %-8.4f  %-8.4f  %-8.4f",
                 mode,
                 s.get("gsm8k_exact_match", 0.0),
                 s.get("humaneval_pass_at_1", 0.0),
                 s.get("finance_acc_norm", 0.0),
                 s.get("medmcqa_accuracy", 0.0),
                 s.get("average", 0.0))


if __name__ == "__main__":
    main()
