"""
E1 — Operator-level Equivalence on Real LoRA Adapters
======================================================
Platform : Mac Mini M4 (CPU) or Lab PC (CUDA)
Purpose  : Load T real trained LoRA adapters from disk, run Pico and WBP on
           every layer, and verify that B_tilde_pico == B_tilde_wbp to
           machine precision.

Usage
-----
    PYTHONPATH=/Users/demid/thesis python experiments/e1_equivalence/run_e1.py \
        --adapters_dir ./adapters \
        --output_dir   ./results/e1 \
        --dtype        float64 \
        --seed         42

All progress is reported through Python logging (not print).
Results are written to <output_dir>/results.json.
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

# ---------------------------------------------------------------------------
# Logging setup — must happen before any other imports that log
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attempt to import safetensors; fall back to torch.load for .bin files
# ---------------------------------------------------------------------------
try:
    from safetensors.torch import load_file as _safetensors_load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    log.warning(
        "safetensors not installed — will fall back to torch.load for .bin files. "
        "Install with: pip install safetensors"
    )


# ---------------------------------------------------------------------------
# Import project source modules
# ---------------------------------------------------------------------------
try:
    from src.pico import merge_pico
    from src.wbp import merge_wbp
except ModuleNotFoundError as exc:
    log.error(
        "Could not import src.pico / src.wbp. "
        "Run with PYTHONPATH pointing to the project root. "
        "Original error: %s",
        exc,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Adapter loading helpers
# ---------------------------------------------------------------------------

def _load_state_dict(adapter_dir: Path) -> Dict[str, torch.Tensor]:
    """
    Load the raw state dict from an adapter directory.
    Prefers .safetensors; falls back to adapter_model.bin.
    """
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"

    if safetensors_path.is_file():
        if not HAS_SAFETENSORS:
            raise RuntimeError(
                f"Found {safetensors_path} but safetensors is not installed. "
                "Run: pip install safetensors"
            )
        log.info("  Loading safetensors from %s", safetensors_path)
        return _safetensors_load_file(str(safetensors_path))

    if bin_path.is_file():
        log.info("  Loading bin from %s", bin_path)
        return torch.load(str(bin_path), map_location="cpu", weights_only=True)

    raise FileNotFoundError(
        f"No adapter weight file found in {adapter_dir}. "
        "Expected adapter_model.safetensors or adapter_model.bin."
    )


def load_adapters(
    adapters_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    """
    Discover and load all adapter subdirectories under `adapters_dir`.

    Returns
    -------
    layer_map : dict
        layer_map[layer_key] = {"B": [B_1, ..., B_T], "A": [A_1, ..., A_T]}
        where each B_t has shape (d_out, r) and A_t has shape (r, d_in).
    domain_names : list[str]
        Ordered list of domain names (one per adapter).
    """
    # Discover adapter subdirectories in sorted order for reproducibility
    subdirs = sorted(
        [p for p in adapters_dir.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )
    if not subdirs:
        raise RuntimeError(f"No subdirectories found in {adapters_dir}.")

    domain_names = [p.name for p in subdirs]
    log.info("Found %d adapter domains: %s", len(subdirs), domain_names)

    # per_domain_weights[domain_idx][key] = tensor
    per_domain_weights: List[Dict[str, torch.Tensor]] = []
    for subdir in subdirs:
        log.info("Loading adapter: %s", subdir.name)
        sd = _load_state_dict(subdir)
        # Cast to requested dtype and move to device
        sd = {k: v.to(device=device, dtype=dtype) for k, v in sd.items()}
        per_domain_weights.append(sd)

    # Build layer_map.
    # Key format (from PEFT): base_model.model.model.layers.N.self_attn.q_proj.lora_B.weight
    # We strip the lora_B/lora_A suffix to get the canonical layer key.
    #
    # Strategy: collect all keys that end with "lora_B.weight" from the first
    # domain, derive the layer key, then gather B and A tensors across all domains.
    first_sd = per_domain_weights[0]
    lora_b_keys = [k for k in first_sd if k.endswith("lora_B.weight")]
    log.info("Found %d LoRA B matrices in first adapter.", len(lora_b_keys))

    if not lora_b_keys:
        raise RuntimeError(
            "No 'lora_B.weight' keys found in the first adapter's state dict. "
            "Keys found:\n" + "\n".join(list(first_sd.keys())[:10])
        )

    layer_map: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    skipped = 0
    for b_key in sorted(lora_b_keys):
        # Derive the companion A key
        a_key = b_key.replace("lora_B.weight", "lora_A.weight")

        # Canonical layer name = strip the lora_X.weight suffix
        # e.g. "base_model.model.model.layers.0.self_attn.q_proj"
        layer_key = b_key.replace(".lora_B.weight", "")

        B_tensors: List[torch.Tensor] = []
        A_tensors: List[torch.Tensor] = []
        missing = False
        for domain_idx, sd in enumerate(per_domain_weights):
            if b_key not in sd or a_key not in sd:
                log.warning(
                    "Layer %s missing from adapter %s — skipping layer.",
                    layer_key,
                    domain_names[domain_idx],
                )
                missing = True
                break
            B_tensors.append(sd[b_key])   # shape (d_out, r)
            A_tensors.append(sd[a_key])   # shape (r,  d_in)

        if missing:
            skipped += 1
            continue

        layer_map[layer_key] = {"B": B_tensors, "A": A_tensors}

    if skipped:
        log.warning("Skipped %d layers due to missing keys in some adapters.", skipped)

    log.info(
        "Built layer_map with %d layers, T=%d domains.", len(layer_map), len(subdirs)
    )
    return layer_map, domain_names


# ---------------------------------------------------------------------------
# Per-layer equivalence check
# ---------------------------------------------------------------------------

def check_layer(
    layer_key: str,
    B_list: List[torch.Tensor],
    A_list: List[torch.Tensor],
) -> Dict:
    """
    Run Pico and WBP on a single layer's B/A lists and compute error metrics.

    Returns a dict with:
        layer, max_rel_error, mean_rel_error, max_abs_error, mean_abs_error
    """
    B_pico, A_pico = merge_pico(B_list, A_list)
    B_wbp,  A_wbp  = merge_wbp(B_list, A_list, beta=1.0)

    # Element-wise absolute error on B_merged (shape d_out × T*r)
    abs_err = (B_pico - B_wbp).abs()
    ref_norm = B_pico.norm(p="fro") + 1e-30  # guard against zero denominator

    max_abs_error  = abs_err.max().item()
    mean_abs_error = abs_err.mean().item()

    # Relative error: Frobenius ||B_pico - B_wbp||_F / ||B_pico||_F
    frob_diff = abs_err.norm(p="fro")
    rel_error = (frob_diff / ref_norm).item()

    # Also check that A_merged is identical (it should be — both paths just cat)
    A_max_diff = (A_pico - A_wbp).abs().max().item()
    if A_max_diff > 1e-12:
        log.warning(
            "Layer %s: A_merged differs by %.2e (unexpected).", layer_key, A_max_diff
        )

    return {
        "layer":          layer_key,
        "max_abs_error":  max_abs_error,
        "mean_abs_error": mean_abs_error,
        "rel_error":      rel_error,       # Frobenius relative error
        "max_rel_error":  rel_error,       # kept for compatibility with README spec
    }


# ---------------------------------------------------------------------------
# T=1 edge-case guard
# ---------------------------------------------------------------------------

def verify_t1_guard(layer_map: Dict) -> None:
    """
    Verifies that when T=1 both merge_pico and merge_wbp return the original
    B and A tensors unchanged.  Uses the first available layer.
    """
    if not layer_map:
        log.warning("layer_map is empty — cannot run T=1 guard check.")
        return

    first_key = next(iter(layer_map))
    B_list = layer_map[first_key]["B"]
    A_list = layer_map[first_key]["A"]

    B_single_p, A_single_p = merge_pico([B_list[0]], [A_list[0]])
    assert torch.allclose(B_single_p, B_list[0], atol=0, rtol=0), (
        "T=1 guard FAILED for merge_pico: B was modified."
    )
    assert torch.allclose(A_single_p, A_list[0], atol=0, rtol=0), (
        "T=1 guard FAILED for merge_pico: A was modified."
    )

    B_single_w, A_single_w = merge_wbp([B_list[0]], [A_list[0]])
    assert torch.allclose(B_single_w, B_list[0], atol=0, rtol=0), (
        "T=1 guard FAILED for merge_wbp: B was modified."
    )
    assert torch.allclose(A_single_w, A_list[0], atol=0, rtol=0), (
        "T=1 guard FAILED for merge_wbp: A was modified."
    )

    log.info("T=1 guard: OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="E1 — Verify Pico == WBP on real trained LoRA adapters."
    )
    parser.add_argument(
        "--adapters_dir",
        type=Path,
        default=Path("/Users/demid/thesis/adapters"),
        help="Directory containing adapter subdirectories (one per domain).",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="unsloth/Meta-Llama-3.1-8B",
        help="HuggingFace model name (informational only — model is NOT loaded).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/Users/demid/thesis/results/e1"),
        help="Directory to write results.json into.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Precision to cast adapter weights into before comparison.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the equivalence check on (e.g., 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (for reproducibility; no stochastic ops in this script).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    log.info("=" * 60)
    log.info("E1 — Pico vs WBP equivalence on real adapters")
    log.info("  adapters_dir : %s", args.adapters_dir)
    log.info("  base_model   : %s (NOT loaded — adapter weights only)", args.base_model)
    log.info("  output_dir   : %s", args.output_dir)
    log.info("  dtype        : %s", args.dtype)
    log.info("  device       : %s", device)
    log.info("  seed         : %d", args.seed)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load adapters
    # ------------------------------------------------------------------
    if not args.adapters_dir.is_dir():
        log.error("adapters_dir does not exist: %s", args.adapters_dir)
        sys.exit(1)

    t0 = time.perf_counter()
    layer_map, domain_names = load_adapters(args.adapters_dir, dtype, device)
    T = len(domain_names)
    load_time = time.perf_counter() - t0
    log.info("Adapter loading complete in %.1f s", load_time)

    # ------------------------------------------------------------------
    # 2. T=1 guard verification
    # ------------------------------------------------------------------
    verify_t1_guard(layer_map)

    # ------------------------------------------------------------------
    # 3. Per-layer equivalence check
    # ------------------------------------------------------------------
    log.info("Running per-layer equivalence check across %d layers ...", len(layer_map))
    per_layer_results = []

    t_check_start = time.perf_counter()
    for i, (layer_key, tensors) in enumerate(sorted(layer_map.items())):
        result = check_layer(layer_key, tensors["B"], tensors["A"])
        per_layer_results.append(result)

        # Progress log every 8 layers to avoid flooding the console
        if (i + 1) % 8 == 0 or (i + 1) == len(layer_map):
            log.info(
                "  [%3d/%d]  %-70s  rel_err=%.2e",
                i + 1,
                len(layer_map),
                layer_key[-70:],
                result["rel_error"],
            )

    check_time = time.perf_counter() - t_check_start
    log.info("Equivalence check complete in %.1f s", check_time)

    # ------------------------------------------------------------------
    # 4. Aggregate metrics
    # ------------------------------------------------------------------
    max_rel_error  = max(r["rel_error"]      for r in per_layer_results)
    mean_rel_error = sum(r["rel_error"]      for r in per_layer_results) / len(per_layer_results)
    max_abs_error  = max(r["max_abs_error"]  for r in per_layer_results)
    mean_abs_error = sum(r["mean_abs_error"] for r in per_layer_results) / len(per_layer_results)

    # Pass/fail thresholds from AGENT_PROMPT.md
    if max_rel_error < 1e-5:
        passed = True
        verdict = "PASS — max_rel_error < 1e-5 (excellent)"
        log.info("RESULT: %s  (max_rel_error=%.2e)", verdict, max_rel_error)
    elif max_rel_error < 1e-4:
        passed = True
        verdict = "PASS — max_rel_error < 1e-4 (acceptable, within float32 precision)"
        log.info("RESULT: %s  (max_rel_error=%.2e)", verdict, max_rel_error)
    else:
        passed = False
        verdict = "FAIL — max_rel_error >= 1e-4 (potential conditioning issue)"
        log.warning("RESULT: %s  (max_rel_error=%.2e)", verdict, max_rel_error)

        # Identify the worst offending layers for diagnostics
        worst = sorted(per_layer_results, key=lambda r: r["rel_error"], reverse=True)[:5]
        log.warning("Top-5 worst layers by rel_error:")
        for w in worst:
            log.warning("  %s  rel_error=%.2e", w["layer"], w["rel_error"])

    # ------------------------------------------------------------------
    # 5. Write results JSON
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment":        "E1",
        "hardware":          f"Device: {device}",
        "dtype":             args.dtype,
        "base_model":        args.base_model,
        "T":                 T,
        "domain_names":      domain_names,
        "num_layers_checked": len(per_layer_results),
        "max_rel_error":     max_rel_error,
        "mean_rel_error":    mean_rel_error,
        "max_abs_error":     max_abs_error,
        "mean_abs_error":    mean_abs_error,
        "passed":            passed,
        "verdict":           verdict,
        "seed":              args.seed,
        "load_time_s":       round(load_time,  2),
        "check_time_s":      round(check_time, 2),
        "per_layer_results": per_layer_results,
    }

    out_path = args.output_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info("Results written to %s", out_path)
    log.info("Summary: passed=%s  max_rel_error=%.2e  layers=%d  T=%d",
             passed, max_rel_error, len(per_layer_results), T)

    # Exit with non-zero code on failure so CI / shell scripts can detect it
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
