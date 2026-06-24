"""
E4 — Equivalence Check across (T, r) Grid
==========================================
Platform : Mac Mini M4 (CPU) — no CUDA required
Purpose  : Verify that Pico and WBP agree to machine precision for all
           combinations of T ∈ {2,3,4,5,6} and r ∈ {8,16,32,64},
           using synthetic float64 matrices.

Usage
-----
    PYTHONPATH=/Users/demid/thesis python experiments/e4_tr_scaling/run_e4_equivalence.py

Results land in ./results/e4/equivalence/:
    results.json
    equivalence_heatmap.png
"""

import json
import logging
import statistics
from pathlib import Path

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
# Configuration
# ---------------------------------------------------------------------------
T_VALUES    = [2, 3, 4, 5, 6]
R_VALUES    = [8, 16, 32, 64]
D_OUT       = 1024          # fixed d_out for equivalence check
DTYPE       = torch.float64 # high precision for equivalence verification
SEEDS       = [42, 43, 44, 45, 46]
RESULTS_DIR = Path("./results/e4/equivalence")

PASS_THRESHOLD = 1e-5       # max_rel_error must be below this to pass


# ---------------------------------------------------------------------------
# Imports from src/
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))  # project root
from src.pico import merge_pico
from src.wbp import merge_wbp


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep() -> list:
    results = []

    for T in T_VALUES:
        for r in R_VALUES:
            Tr = T * r
            rel_errors = []

            for seed in SEEDS:
                torch.manual_seed(seed)
                B_list = [torch.randn(D_OUT, r, dtype=DTYPE) for _ in range(T)]
                A_list = [torch.randn(r, D_OUT // 2, dtype=DTYPE) for _ in range(T)]

                B_pico, A_pico = merge_pico(B_list, A_list)
                B_wbp,  A_wbp  = merge_wbp(B_list, A_list, beta=1.0)

                rel_err = (
                    (B_pico - B_wbp).norm(p="fro") /
                    (B_pico.norm(p="fro") + 1e-30)
                ).item()
                rel_errors.append(rel_err)

            mean_rel_err = statistics.mean(rel_errors)
            max_rel_err  = max(rel_errors)
            passed       = max_rel_err < PASS_THRESHOLD

            status = "✓" if passed else "✗"
            log.info(
                "  %s T=%d  r=%2d  Tr=%3d  "
                "mean_rel_err=%.2e  max_rel_err=%.2e",
                status, T, r, Tr, mean_rel_err, max_rel_err,
            )

            results.append({
                "T":             T,
                "r":             r,
                "Tr":            Tr,
                "mean_rel_error": mean_rel_err,
                "max_rel_error":  max_rel_err,
                "passed":         passed,
            })

    return results


# ---------------------------------------------------------------------------
# Plot — heatmap of log10(max_rel_error)
# ---------------------------------------------------------------------------

def make_heatmap(results: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Build matrix: rows = r values, columns = T values
    grid = {}
    for row in results:
        grid[(row["T"], row["r"])] = row["max_rel_error"]

    matrix = []
    for r in R_VALUES:
        row_vals = []
        for T in T_VALUES:
            val = grid.get((T, r), float("nan"))
            row_vals.append(val if val > 0 else 1e-20)
        matrix.append(row_vals)

    matrix_log = [[v if v != v else max(-20.0, __import__("math").log10(v))
                   for v in row] for row in matrix]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix_log, aspect="auto", cmap="viridis_r",
                   vmin=-18, vmax=-4)
    ax.set_xticks(range(len(T_VALUES)))
    ax.set_xticklabels([str(t) for t in T_VALUES])
    ax.set_yticks(range(len(R_VALUES)))
    ax.set_yticklabels([str(r) for r in R_VALUES])
    ax.set_xlabel("T (number of tasks)", fontsize=12)
    ax.set_ylabel("r (LoRA rank)", fontsize=12)
    ax.set_title(
        f"Pico vs WBP: log₁₀(max_rel_error)\n"
        f"d_out={D_OUT}, float64, 5 seeds  (Mac Mini M4 CPU)"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log₁₀(max_rel_error)", fontsize=10)

    # Annotate each cell with the exponent
    for i, r in enumerate(R_VALUES):
        for j, T in enumerate(T_VALUES):
            val = matrix_log[i][j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color="white" if val < -10 else "black", fontsize=9)

    plt.tight_layout()
    heatmap_path = RESULTS_DIR / "equivalence_heatmap.png"
    fig.savefig(heatmap_path, dpi=300)
    plt.close(fig)
    log.info("Heatmap saved to %s", heatmap_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("E4 — Equivalence grid  T=%s  r=%s  d_out=%d  dtype=float64",
             T_VALUES, R_VALUES, D_OUT)
    log.info("Seeds: %s  pass_threshold: %.0e", SEEDS, PASS_THRESHOLD)
    log.info("=" * 60)

    results = run_sweep()

    all_passed = all(r["passed"] for r in results)
    log.info("=" * 60)
    log.info("all_passed = %s", all_passed)
    if not all_passed:
        failing = [r for r in results if not r["passed"]]
        for r in failing:
            log.warning("FAILED: T=%d r=%d max_rel_err=%.2e", r["T"], r["r"], r["max_rel_error"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment":      "E4_equivalence",
        "hardware":        "Mac Mini M4 (CPU)",
        "d_out":           D_OUT,
        "dtype":           "float64",
        "seeds":           SEEDS,
        "pass_threshold":  PASS_THRESHOLD,
        "all_passed":      all_passed,
        "results":         results,
    }
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results written to %s", out_path)

    try:
        make_heatmap(results)
    except Exception as e:
        log.warning("Heatmap generation failed: %s  (results JSON still valid)", e)

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
