"""
E5 — Decoupled-λ Sweep (β-WBP), Exploratory
============================================
Platform : Lab RTX 6000 (24 GB VRAM, CUDA)
Purpose  : Sweep β ∈ {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0} and measure
           downstream benchmark accuracy for each β, using the same adapter
           weights and base model as E2.

Usage
-----
    PYTHONPATH=/path/to/thesis python experiments/e5_beta_sweep/run_e5.py \
        --adapters_dir  ./adapters \
        --base_model    unsloth/Meta-Llama-3.1-8B \
        --output_dir    ./results/e5 \
        --dtype         bfloat16 \
        --device        cuda \
        --seed          42 \
        --e2_results_json ./results/e2/results.json

IMPORTANT — what NOT to claim from this experiment:
-----------------------------------------------------
# DO NOT CLAIM from E5 alone:
# - That WBP-tuned "beats" Pico as a deployable result
# - That the chosen beta generalizes to other models or domains
# - That the data-free property is preserved
#   (beta chosen via benchmark accuracy is NOT data-free)
# These are explicitly acknowledged limitations in the thesis's Future Work section.

This is framed as EXPLORATORY — see AGENT.md §5.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np

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
# β values (hardcoded per AGENT_PROMPT spec)
# ---------------------------------------------------------------------------
BETA_VALUES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]


# ---------------------------------------------------------------------------
# Reuse adapter loading and weight patching from E2
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parents[2]))           # project root
sys.path.insert(0, str(Path(__file__).parents[1] / "e2_accuracy"))  # evaluate.py

from experiments.e2_accuracy.run_e2 import (
    load_all_adapters,
    patch_model_weights,
    restore_model_weights,
)
from src.wbp import merge_wbp


def apply_wbp_beta(
    B_list: List[torch.Tensor],
    A_list: List[torch.Tensor],
    beta: float = 1.0,
) -> torch.Tensor:
    """WBP calibration at a given β, returns dense delta_W."""
    B_merged, A_merged = merge_wbp(B_list, A_list, beta=beta)
    return B_merged @ A_merged


# ---------------------------------------------------------------------------
# Benchmark evaluation (import from E2's evaluate.py)
# ---------------------------------------------------------------------------

def run_eval_suite(model, tokenizer, seed: int) -> dict:
    from evaluate import eval_gsm8k, eval_humaneval, eval_finance, eval_medmcqa
    scores = {}
    scores["gsm8k_exact_match"]     = eval_gsm8k(model, tokenizer, device=None, seed=seed)
    scores["humaneval_pass_at_1"]   = eval_humaneval(model, tokenizer, device=None, seed=seed)
    scores["finance_acc_norm"]      = eval_finance(model, tokenizer, device=None, seed=seed)
    scores["medmcqa_accuracy"]      = eval_medmcqa(model, tokenizer, device=None, seed=seed)
    scores["average"] = sum(scores.values()) / len(scores)
    return scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="E5 — β sweep for WBP.")
    parser.add_argument("--adapters_dir",     type=Path, default=Path("./adapters"))
    parser.add_argument("--base_model",       type=str,  default="unsloth/Meta-Llama-3.1-8B")
    parser.add_argument("--output_dir",       type=Path, default=Path("./results/e5"))
    parser.add_argument("--dtype",            type=str,  default="float16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device",           type=str,  default="cuda")
    parser.add_argument("--seed",             type=int,  default=42)
    parser.add_argument("--e2_results_json",  type=Path, default=None,
                        help="Path to E2 results.json for baseline overlay on plots.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            log.error("CUDA not available. This script must run on the Lab RTX 6000.")
            sys.exit(1)
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model_dtype = dtype_map[args.dtype]

    log.info("=" * 60)
    log.info("E5 — β sweep (WBP exploratory)")
    log.info("  β values     : %s", BETA_VALUES)
    log.info("  base_model   : %s", args.base_model)
    log.info("  adapters_dir : %s", args.adapters_dir)
    log.info("  output_dir   : %s", args.output_dir)
    log.info("  dtype        : %s", args.dtype)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load baseline results from E2 (for plot overlays)
    # ------------------------------------------------------------------
    e2_baselines = {"no_cal_average": None, "pico_average": None}
    if args.e2_results_json and args.e2_results_json.is_file():
        with open(args.e2_results_json) as f:
            e2_data = json.load(f)
        e2_baselines["no_cal_average"] = e2_data["results"].get("no_cal", {}).get("average")
        e2_baselines["pico_average"]   = e2_data["results"].get("pico",   {}).get("average")
        log.info("E2 baselines: no_cal=%.4f  pico=%.4f",
                 e2_baselines["no_cal_average"] or 0.0,
                 e2_baselines["pico_average"]   or 0.0)
    else:
        log.warning("--e2_results_json not provided or not found. Baseline lines will be omitted from plots.")

    # ------------------------------------------------------------------
    # 2. Load adapters
    # ------------------------------------------------------------------
    layer_adapters, domain_names = load_all_adapters(args.adapters_dir)
    T = len(domain_names)

    # ------------------------------------------------------------------
    # 3. Load base model ONCE
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading base model ...")
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
    # 4. β sweep
    # ------------------------------------------------------------------
    sweep_results = []

    for beta in BETA_VALUES:
        log.info("-" * 50)
        log.info("β = %.2f", beta)
        t0 = time.perf_counter()

        # Patch weights with WBP at this β
        originals = patch_model_weights(
            model, layer_adapters,
            calibration_fn=apply_wbp_beta,
            target_dtype=model_dtype,
            beta=beta,
        )

        # Evaluate
        scores = run_eval_suite(model, tokenizer, args.seed)

        # Restore
        restore_model_weights(model, originals)
        del originals
        torch.cuda.empty_cache()

        elapsed = time.perf_counter() - t0
        log.info("  β=%.2f  avg=%.4f  elapsed=%.1f min", beta, scores["average"], elapsed / 60)

        sweep_results.append({"beta": beta, **scores})

    # ------------------------------------------------------------------
    # 5. Automated interpretation (per AGENT_PROMPT spec)
    # ------------------------------------------------------------------
    averages    = [r["average"] for r in sweep_results]
    beta_values = [r["beta"]    for r in sweep_results]

    peak_idx   = int(np.argmax(averages))
    peak_beta  = beta_values[peak_idx]
    peak_val   = averages[peak_idx]
    pico_val   = e2_baselines.get("pico_average") or averages[beta_values.index(1.0)]

    if peak_beta != 1.0 and peak_val > pico_val + 0.005:
        log.info(
            "FINDING: Curve peaks at beta=%.2f (avg=%.3f), above Pico (%.3f). "
            "The family contains better points than beta=1. "
            "NOTE: A principled data-free beta selection method is still needed.",
            peak_beta, peak_val, pico_val,
        )
    elif max(averages) - min(averages) < 0.01:
        log.info("FINDING: Flat curve — Pico's beta=1 was already near-optimal in this regime.")
    elif averages[0] > averages[-1]:
        log.info("FINDING: Monotonically decreasing with beta — over-shrinkage beyond beta=1.")
    else:
        log.info("FINDING: Monotonically increasing with beta — under-shrinkage at beta=1.")

    # ------------------------------------------------------------------
    # 6. Write results JSON
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment":   "E5",
        "hardware":     torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "base_model":   args.base_model,
        "dtype":        args.dtype,
        "T":            T,
        "domain_names": domain_names,
        "seed":         args.seed,
        "beta_values":  BETA_VALUES,
        "baselines":    e2_baselines,
        "results":      sweep_results,
    }
    out_path = args.output_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results written to %s", out_path)

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    try:
        _make_plots(sweep_results, e2_baselines, args.output_dir)
    except Exception as e:
        log.warning("Plot generation failed: %s  (results JSON still valid)", e)

    log.info("E5 complete.")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _make_plots(sweep_results: list, baselines: dict, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    betas    = [r["beta"]   for r in sweep_results]
    averages = [r["average"] for r in sweep_results]

    no_cal_avg = baselines.get("no_cal_average")
    pico_avg   = baselines.get("pico_average")

    # ---- Plot 1: Average accuracy vs. β ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(betas, averages, color="darkorange", marker="o", linewidth=2.5, label="WBP(β)")
    if no_cal_avg is not None:
        ax.axhline(no_cal_avg, color="gray",      linestyle="--", linewidth=1.5, label="No-Cal baseline")
    if pico_avg is not None:
        ax.axhline(pico_avg,   color="steelblue", linestyle="--", linewidth=1.5, label="Pico (β=1.0)")
    ax.axvline(1.0, color="steelblue", linestyle=":", linewidth=1.2, alpha=0.7, label="Pico-equivalent (β=1)")
    ax.set_xlabel("β", fontsize=12)
    ax.set_ylabel("Average accuracy (4 benchmarks)", fontsize=12)
    ax.set_title("WBP β sweep — Average benchmark accuracy\n(T=4, Meta-Llama-3.1-8B, RTX 6000)")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    p1 = output_dir / "beta_sweep_avg.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    log.info("Average accuracy plot saved to %s", p1)

    # ---- Plot 2: Per-benchmark accuracy vs. β (4 subplots) ----
    benchmarks = [
        ("gsm8k_exact_match",   "GSM8K (exact match)"),
        ("humaneval_pass_at_1", "HumanEval (pass@1)"),
        ("finance_acc_norm",    "Finance/MMLU Macro (acc_norm)"),
        ("medmcqa_accuracy",    "MedMCQA (accuracy)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (key, title) in zip(axes.flat, benchmarks):
        scores = [r[key] for r in sweep_results]
        ax.plot(betas, scores, color="darkorange", marker="o", linewidth=2, label="WBP(β)")
        if no_cal_avg is not None:
            # Use E2 per-benchmark if available (we only have average here, skip)
            pass
        if pico_avg is not None:
            ax.axvline(1.0, color="steelblue", linestyle=":", linewidth=1.2, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_xlabel("β", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
    plt.suptitle("WBP β sweep — Per-benchmark accuracy (T=4, Meta-Llama-3.1-8B)", fontsize=12, y=1.01)
    plt.tight_layout()
    p2 = output_dir / "beta_sweep_per_benchmark.png"
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Per-benchmark plot saved to %s", p2)


if __name__ == "__main__":
    main()
