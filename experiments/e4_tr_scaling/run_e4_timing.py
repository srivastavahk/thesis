"""
E4 — Timing Grid across (T, r) — CUDA Required
===============================================
Platform : Lab RTX 6000 (24 GB VRAM, CUDA)
Purpose  : Measure wall-clock time and peak GPU memory for Pico (SVD) and
           WBP (Woodbury) across all combinations of T ∈ {2,3,4,5,6}
           and r ∈ {8,16,32,64} using synthetic float32 matrices.

Usage
-----
    python experiments/e4_tr_scaling/run_e4_timing.py

Results land in ./results/e4/timing/:
    results.json
    speedup_heatmap.png
    time_vs_T_r{r}.png   (one per r value)

Bundle this with E3 in the same lab GPU session — it takes < 30 min.
"""

import json
import logging
import statistics
import time
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
T_VALUES      = [2, 3, 4, 5, 6]
R_VALUES      = [8, 16, 32, 64]
D_OUT         = 4096        # representative production-scale d_out
WARMUP_ITERS  = 5
TIMED_ITERS   = 20
DTYPE         = torch.float32
SEEDS         = [42, 43, 44, 45, 46]
RESULTS_DIR   = Path("./results/e4/timing")


# ---------------------------------------------------------------------------
# Inline calibration functions (copied from E3 — no src/ import for clean timing)
# ---------------------------------------------------------------------------

def pico_calibrate(B_all: torch.Tensor, T: int) -> torch.Tensor:
    """Pico thin-SVD path. Returns calibrated B_all (d_out, T*r)."""
    U, S, Vh = torch.linalg.svd(B_all, full_matrices=False)
    S_sq = S ** 2
    s = S_sq / S_sq.sum()
    alpha = 1.0 / (1.0 + (T - 1) * s)
    alpha_minus_1 = (alpha - 1.0).unsqueeze(0)
    return B_all + U @ (alpha_minus_1.T * (U.T @ B_all))


def wbp_calibrate(B_all: torch.Tensor, T: int) -> torch.Tensor:
    """WBP Woodbury path. Returns calibrated B_all (d_out, T*r)."""
    G = B_all.T @ B_all
    lam = (T - 1) / torch.trace(G)
    Tr = B_all.shape[1]
    K = torch.linalg.inv(
        torch.eye(Tr, device=B_all.device, dtype=B_all.dtype) / lam + G
    )
    return B_all - B_all @ (K @ (B_all.T @ B_all))


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def time_fn(fn, *args, warmup: int = WARMUP_ITERS, iters: int = TIMED_ITERS):
    """Returns list of per-iteration wall-clock times (seconds), GPU-synchronized."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()      # MANDATORY before stopping clock
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def measure_peak_mem(fn, *args) -> int:
    """Returns peak GPU memory allocated (bytes)."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep() -> list:
    results = []

    for T in T_VALUES:
        for r in R_VALUES:
            Tr = T * r
            log.info("T=%d  r=%2d  Tr=%3d  d_out=%d", T, r, Tr, D_OUT)

            pico_times_all = []
            wbp_times_all  = []
            pico_mem_all   = []
            wbp_mem_all    = []

            for seed in SEEDS:
                torch.manual_seed(seed)
                B_all = torch.randn(D_OUT, Tr, dtype=DTYPE, device="cuda")

                pico_iter_times = time_fn(pico_calibrate, B_all, T)
                wbp_iter_times  = time_fn(wbp_calibrate,  B_all, T)
                pico_times_all.extend(pico_iter_times)
                wbp_times_all.extend(wbp_iter_times)

                pico_mem_all.append(measure_peak_mem(pico_calibrate, B_all, T))
                wbp_mem_all.append(measure_peak_mem(wbp_calibrate,  B_all, T))

            pico_mean = statistics.mean(pico_times_all)
            pico_std  = statistics.stdev(pico_times_all)
            wbp_mean  = statistics.mean(wbp_times_all)
            wbp_std   = statistics.stdev(wbp_times_all)
            speedup   = pico_mean / wbp_mean if wbp_mean > 0 else float("nan")

            log.info(
                "  Pico: %.3f ms ± %.3f  |  WBP: %.3f ms ± %.3f  |  Speedup: %.2fx",
                pico_mean * 1e3, pico_std * 1e3,
                wbp_mean  * 1e3, wbp_std  * 1e3,
                speedup,
            )

            results.append({
                "T":   T,
                "r":   r,
                "Tr":  Tr,
                "pico": {
                    "mean_time_s":    pico_mean,
                    "std_time_s":     pico_std,
                    "peak_mem_bytes": int(statistics.mean(pico_mem_all)),
                },
                "wbp": {
                    "mean_time_s":    wbp_mean,
                    "std_time_s":     wbp_std,
                    "peak_mem_bytes": int(statistics.mean(wbp_mem_all)),
                },
                "speedup": round(speedup, 4),
            })

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(results: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    # Build speedup matrix: rows = r values, cols = T values
    speedup_grid = {}
    for row in results:
        speedup_grid[(row["T"], row["r"])] = row["speedup"]

    matrix = []
    for r in R_VALUES:
        matrix.append([speedup_grid.get((T, r), float("nan")) for T in T_VALUES])

    # ---- Speedup heatmap ----
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.RdYlGn
    norm = mcolors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=2.5)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(T_VALUES)))
    ax.set_xticklabels([str(t) for t in T_VALUES])
    ax.set_yticks(range(len(R_VALUES)))
    ax.set_yticklabels([str(r) for r in R_VALUES])
    ax.set_xlabel("T (number of tasks)", fontsize=12)
    ax.set_ylabel("r (LoRA rank)", fontsize=12)
    ax.set_title(f"WBP Speedup over Pico (Pico time / WBP time)\nd_out={D_OUT}, RTX 6000, float32")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Speedup (>1 = WBP faster)", fontsize=10)

    for i, r in enumerate(R_VALUES):
        for j, T in enumerate(T_VALUES):
            val = matrix[i][j]
            ax.text(j, i, f"{val:.2f}×", ha="center", va="center",
                    fontsize=9,
                    color="white" if val < 0.8 or val > 2.0 else "black")

    plt.tight_layout()
    heatmap_path = RESULTS_DIR / "speedup_heatmap.png"
    fig.savefig(heatmap_path, dpi=300)
    plt.close(fig)
    log.info("Speedup heatmap saved to %s", heatmap_path)

    # ---- Time vs T line plots (one per r) ----
    for r in R_VALUES:
        r_results = [row for row in results if row["r"] == r]
        r_results.sort(key=lambda x: x["T"])
        Ts          = [row["T"] for row in r_results]
        pico_ms     = [row["pico"]["mean_time_s"] * 1e3 for row in r_results]
        pico_std    = [row["pico"]["std_time_s"]  * 1e3 for row in r_results]
        wbp_ms      = [row["wbp"]["mean_time_s"]  * 1e3 for row in r_results]
        wbp_std     = [row["wbp"]["std_time_s"]   * 1e3 for row in r_results]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(Ts, pico_ms, yerr=pico_std, label="Pico (SVD)",
                    color="steelblue",  marker="o", capsize=4, linewidth=2)
        ax.errorbar(Ts, wbp_ms,  yerr=wbp_std,  label="WBP (Woodbury)",
                    color="darkorange", marker="s", capsize=4, linewidth=2)
        ax.set_xlabel("T (number of tasks)", fontsize=11)
        ax.set_ylabel("Wall-clock time (ms)", fontsize=11)
        ax.set_title(f"Time vs. T  (r={r}, d_out={D_OUT}, RTX 6000)")
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(Ts)
        plt.tight_layout()
        line_path = RESULTS_DIR / f"time_vs_T_r{r}.png"
        fig.savefig(line_path, dpi=300)
        plt.close(fig)
        log.info("Line plot saved to %s", line_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        log.error(
            "CUDA is not available. run_e4_timing.py requires the Lab RTX 6000. "
            "For the equivalence (CPU) sub-part, run run_e4_equivalence.py instead."
        )
        raise SystemExit(1)

    device_name = torch.cuda.get_device_name(0)
    log.info("GPU: %s", device_name)
    log.info("E4 timing sweep: T=%s  r=%s  d_out=%d  seeds=%s",
             T_VALUES, R_VALUES, D_OUT, SEEDS)

    results = run_sweep()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment":   "E4_timing",
        "hardware":     device_name,
        "d_out":        D_OUT,
        "dtype":        "float32",
        "warmup_iters": WARMUP_ITERS,
        "timed_iters":  TIMED_ITERS,
        "seeds":        SEEDS,
        "results":      results,
    }
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results written to %s", out_path)

    try:
        make_plots(results)
    except Exception as e:
        log.warning("Plot generation failed: %s  (results JSON still valid)", e)

    log.info("E4 timing complete.")


if __name__ == "__main__":
    main()
