"""
E4 — Equivalence Verification vs T and r
========================================
Platform : Mac Mini M4 (CPU) — NO CUDA
Purpose  : Verify Pico and WBP numerical equivalence under varying number of
           adapters (T) and rank (r).
"""

import json
import logging
import os
import statistics
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Require running from the repository root (PYTHONPATH=.)
from src.pico import merge_pico
from src.wbp import merge_wbp

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
# Configuration
# ---------------------------------------------------------------------------
T_VALUES = [2, 3, 4, 5, 6]
R_VALUES = [8, 16, 32, 64]
D_OUT = 1024
DTYPE = torch.float64
SEEDS = [42, 43, 44, 45, 46]
RESULTS_DIR = Path("./results/e4/equivalence")

def run_equivalence_sweep():
    results = []
    all_passed = True
    
    # For heatmap: dict to hold max error for each (T, r)
    heatmap_data = {}

    for t in T_VALUES:
        for r in R_VALUES:
            log.info(f"Testing T={t}, r={r}")
            rel_errors = []
            
            for seed in SEEDS:
                torch.manual_seed(seed)
                
                # Generate random factors
                B_list = [torch.randn(D_OUT, r, dtype=DTYPE) for _ in range(t)]
                A_list = [torch.randn(r, D_OUT // 2, dtype=DTYPE) for _ in range(t)]
                
                # Merge with Pico
                B_pico, A_pico = merge_pico(B_list, A_list)
                
                # Merge with WBP
                B_wbp, A_wbp = merge_wbp(B_list, A_list, beta=1.0)
                
                # Compute relative Frobenius error on B_merged
                rel_err = (B_pico - B_wbp).norm('fro') / (B_pico.norm('fro') + 1e-12)
                rel_errors.append(rel_err.item())
            
            mean_rel_error = statistics.mean(rel_errors)
            max_rel_error = max(rel_errors)
            
            log.info(f"  Mean Rel Error: {mean_rel_error:.3e} | Max Rel Error: {max_rel_error:.3e}")
            
            if max_rel_error >= 1e-5:
                all_passed = False
                log.error(f"  FAILED: max_rel_error {max_rel_error:.3e} >= 1e-5")
                
            results.append({
                "T": t,
                "r": r,
                "Tr": t * r,
                "mean_rel_error": mean_rel_error,
                "max_rel_error": max_rel_error
            })
            
            heatmap_data[(t, r)] = max_rel_error

    return results, all_passed, heatmap_data

def generate_heatmap(heatmap_data):
    import numpy as np
    
    # Create 2D array for the heatmap
    Z = np.zeros((len(R_VALUES), len(T_VALUES)))
    for i, r in enumerate(R_VALUES):
        for j, t in enumerate(T_VALUES):
            Z[i, j] = np.log10(heatmap_data[(t, r)] + 1e-16) # Add epsilon for log10
            
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(Z, origin="lower", cmap="viridis", aspect="auto")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(T_VALUES)))
    ax.set_yticks(np.arange(len(R_VALUES)))
    ax.set_xticklabels(T_VALUES)
    ax.set_yticklabels(R_VALUES)
    
    ax.set_xlabel("Number of Tasks (T)", fontsize=12)
    ax.set_ylabel("Rank (r)", fontsize=12)
    ax.set_title("Max Relative Error (log10)\nPico vs WBP Calibration", fontsize=14)
    
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("log10(max relative error)")
    
    # Annotate cells
    for i in range(len(R_VALUES)):
        for j in range(len(T_VALUES)):
            val = Z[i, j]
            text_color = "white" if val < Z.max() - (Z.max() - Z.min())/2 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_color)
            
    plt.tight_layout()
    plot_path = RESULTS_DIR / "equivalence_heatmap.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    log.info(f"Heatmap saved to {plot_path}")

def main():
    log.info("Starting E4 Equivalence Sweep...")
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results, all_passed, heatmap_data = run_equivalence_sweep()
    
    output = {
        "experiment": "E4_equivalence",
        "hardware": "Mac Mini M4 (CPU)",
        "d_out": D_OUT,
        "dtype": "float64",
        "seeds": SEEDS,
        "results": results,
        "all_passed": all_passed
    }
    
    json_path = RESULTS_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Results written to {json_path}")
    
    try:
        generate_heatmap(heatmap_data)
    except Exception as e:
        log.warning(f"Plot generation failed: {e}")
        
    log.info("E4 Equivalence complete.")

if __name__ == "__main__":
    main()
