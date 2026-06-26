"""
E4 — Computational Efficiency vs. T and r
==========================================
Platform : Lab RTX 6000 (24 GB VRAM, CUDA)
Purpose  : Measure wall-clock time and peak GPU memory for Pico (SVD)
           and WBP (Woodbury) as the number of tasks (T) and rank (r) grow.

Usage
-----
    python experiments/e4_tr_scaling/run_e4_timing.py

Results land in ./results/e4/timing/:
    results.json
    speedup_heatmap.png
    time_vs_T_r*.png
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
T_VALUES = [2, 3, 4, 5, 6]
R_VALUES = [8, 16, 32, 64]
D_OUT = 4096    # Representative production-scale d_out
WARMUP_ITERS = 5
TIMED_ITERS = 20
DTYPE = torch.float32
SEEDS = [42, 43, 44, 45, 46]
RESULTS_DIR = Path("./results/e4/timing")

# ---------------------------------------------------------------------------
# Inline calibration functions (no src/ import — avoids Python overhead in timing)
# ---------------------------------------------------------------------------

def pico_calibrate(B_all: torch.Tensor, T: int) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(B_all, full_matrices=False)
    S_sq = S ** 2
    s = S_sq / S_sq.sum()
    alpha = 1.0 / (1.0 + (T - 1) * s)
    alpha_minus_1 = (alpha - 1.0).unsqueeze(0)  # shape (1, Tr) for broadcasting
    return B_all + U @ (alpha_minus_1.T * (U.T @ B_all))

def wbp_calibrate(B_all: torch.Tensor, T: int) -> torch.Tensor:
    G = B_all.T @ B_all                                    # (Tr, Tr)
    lam = (T - 1) / torch.trace(G)
    Tr = B_all.shape[1]
    K = torch.linalg.inv(
        torch.eye(Tr, device=B_all.device, dtype=B_all.dtype) / lam + G
    )                                                       # (Tr, Tr)
    return B_all - B_all @ (K @ (B_all.T @ B_all))

# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def time_fn(fn, *args, warmup: int = WARMUP_ITERS, iters: int = TIMED_ITERS):
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times

def measure_peak_mem(fn, *args) -> int:
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
            log.info("T = %d, r = %d, Tr = %d", T, r, Tr)

            pico_times_all = []
            wbp_times_all = []
            pico_mem_all = []
            wbp_mem_all = []

            for seed in SEEDS:
                torch.manual_seed(seed)
                B_all = torch.randn(D_OUT, Tr, dtype=DTYPE, device="cuda")

                pico_iter_times = time_fn(pico_calibrate, B_all, T)
                pico_times_all.extend(pico_iter_times)

                wbp_iter_times = time_fn(wbp_calibrate, B_all, T)
                wbp_times_all.extend(wbp_iter_times)

                pico_mem_all.append(measure_peak_mem(pico_calibrate, B_all, T))
                wbp_mem_all.append(measure_peak_mem(wbp_calibrate, B_all, T))

            pico_mean = statistics.mean(pico_times_all)
            pico_std = statistics.stdev(pico_times_all)
            wbp_mean = statistics.mean(wbp_times_all)
            wbp_std = statistics.stdev(wbp_times_all)
            speedup = pico_mean / wbp_mean if wbp_mean > 0 else float("nan")

            log.info(
                "  Pico: %.3f ms | WBP: %.3f ms | Speedup: %.2fx",
                pico_mean * 1e3, wbp_mean * 1e3, speedup
            )

            results.append({
                "T": T,
                "r": r,
                "Tr": Tr,
                "pico": {
                    "mean_time_s": pico_mean,
                    "std_time_s": pico_std,
                    "peak_mem_bytes": int(statistics.mean(pico_mem_all)),
                },
                "wbp": {
                    "mean_time_s": wbp_mean,
                    "std_time_s": wbp_std,
                    "peak_mem_bytes": int(statistics.mean(wbp_mem_all)),
                },
                "speedup": round(speedup, 4),
            })

    return results

# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def make_plots(results: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Heatmap: T (x) vs r (y), cell = speedup
    speedup_matrix = np.zeros((len(R_VALUES), len(T_VALUES)))
    for res in results:
        t_idx = T_VALUES.index(res["T"])
        r_idx = R_VALUES.index(res["r"])
        speedup_matrix[r_idx, t_idx] = res["speedup"]

    fig, ax = plt.subplots(figsize=(8, 6))
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=1.0, vmin=speedup_matrix.min()*0.9, vmax=speedup_matrix.max()*1.1)
    cax = ax.imshow(speedup_matrix, cmap="RdYlGn", norm=norm, origin="lower")

    ax.set_xticks(np.arange(len(T_VALUES)))
    ax.set_yticks(np.arange(len(R_VALUES)))
    ax.set_xticklabels(T_VALUES)
    ax.set_yticklabels(R_VALUES)
    ax.set_xlabel("Number of Tasks (T)", fontsize=12)
    ax.set_ylabel("LoRA Rank (r)", fontsize=12)
    ax.set_title(f"WBP Speedup vs Pico (d_out={D_OUT}, float32)\nSpeedup > 1 means WBP is faster", fontsize=12)

    for i in range(len(R_VALUES)):
        for j in range(len(T_VALUES)):
            val = speedup_matrix[i, j]
            color = "white" if val < 0.8 or val > 1.2 else "black"
            ax.text(j, i, f"{val:.2f}x", ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Speedup Ratio")
    plt.tight_layout()
    heatmap_path = RESULTS_DIR / "speedup_heatmap.png"
    fig.savefig(heatmap_path, dpi=300)
    plt.close(fig)
    log.info("Speedup heatmap saved to %s", heatmap_path)

    # 2. Line plots: time vs T for each r
    for r in R_VALUES:
        r_results = [res for res in results if res["r"] == r]
        t_vals = [res["T"] for res in r_results]
        pico_ms = [res["pico"]["mean_time_s"] * 1e3 for res in r_results]
        pico_std = [res["pico"]["std_time_s"] * 1e3 for res in r_results]
        wbp_ms = [res["wbp"]["mean_time_s"] * 1e3 for res in r_results]
        wbp_std = [res["wbp"]["std_time_s"] * 1e3 for res in r_results]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(t_vals, pico_ms, yerr=pico_std, label="Pico (SVD)", color="steelblue", marker="o", capsize=4, linewidth=2)
        ax.errorbar(t_vals, wbp_ms, yerr=wbp_std, label="WBP (Woodbury)", color="darkorange", marker="s", capsize=4, linewidth=2)
        
        ax.set_xlabel("Number of Tasks (T)", fontsize=12)
        ax.set_ylabel("Wall-clock time (ms)", fontsize=12)
        ax.set_title(f"Calibration Time vs T (r={r}, d_out={D_OUT}, RTX 6000)")
        ax.set_xticks(t_vals)
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        lineplot_path = RESULTS_DIR / f"time_vs_T_r{r}.png"
        fig.savefig(lineplot_path, dpi=300)
        plt.close(fig)
        log.info("Line plot saved to %s", lineplot_path)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        log.error(
            "CUDA is not available. E4 requires the Lab RTX 6000. "
            "Exiting."
        )
        raise SystemExit(1)

    device_name = torch.cuda.get_device_name(0)
    log.info("GPU: %s", device_name)
    log.info("E4 sweep: T=%s, r=%s, d_out=%d, seeds=%s, warmup=%d, iters=%d",
             T_VALUES, R_VALUES, D_OUT, SEEDS, WARMUP_ITERS, TIMED_ITERS)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_sweep()

    output = {
        "experiment": "E4_timing",
        "hardware": device_name,
        "d_out": D_OUT,
        "dtype": "float32",
        "warmup_iters": WARMUP_ITERS,
        "timed_iters": TIMED_ITERS,
        "seeds": SEEDS,
        "results": results,
    }
    
    out_path = RESULTS_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results written to %s", out_path)

    try:
        make_plots(results)
    except Exception as e:
        log.warning("Plot generation failed: %s", e)

    log.info("E4 timing complete.")

if __name__ == "__main__":
    main()
