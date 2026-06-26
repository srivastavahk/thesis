"""
E4 — Timing Benchmarks vs T and r
=================================
Platform : Lab RTX 6000 (CUDA required)
Purpose  : Measure wall-clock time and peak GPU memory for Pico and WBP
           under varying number of adapters (T) and rank (r).
"""

import json
import logging
import statistics
import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Inline calibration functions (from E3)
# ---------------------------------------------------------------------------
def pico_calibrate(B_all: torch.Tensor, T: int) -> torch.Tensor:
    """
    Pico thin-SVD path.
    Returns calibrated B_all of shape (d_out, T*r).
    """
    U, S, Vh = torch.linalg.svd(B_all, full_matrices=False)
    S_sq = S ** 2
    s = S_sq / S_sq.sum()
    alpha = 1.0 / (1.0 + (T - 1) * s)
    alpha_minus_1 = (alpha - 1.0).unsqueeze(0)  # shape (1, Tr) for broadcasting
    return B_all + U @ (alpha_minus_1.T * (U.T @ B_all))

def wbp_calibrate(B_all: torch.Tensor, T: int) -> torch.Tensor:
    """
    WBP Woodbury path.
    Returns calibrated B_all of shape (d_out, T*r).
    """
    G = B_all.T @ B_all                                    # (Tr, Tr)
    lam = (T - 1) / torch.trace(G)
    Tr = B_all.shape[1]
    K = torch.linalg.inv(
        torch.eye(Tr, device=B_all.device, dtype=B_all.dtype) / lam + G
    )                                                       # (Tr, Tr)
    return B_all - B_all @ (K @ (B_all.T @ B_all))

# ---------------------------------------------------------------------------
# Timing harness (from E3)
# ---------------------------------------------------------------------------
def time_fn(fn, *args, warmup: int = WARMUP_ITERS, iters: int = TIMED_ITERS):
    """
    Returns a list of per-iteration wall-clock times (seconds).
    GPU-synchronized before stopping the clock.
    """
    # Warmup — not timed
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()      # MANDATORY — ensures kernel completion
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times

def measure_peak_mem(fn, *args) -> int:
    """Returns peak GPU memory allocated (bytes) during fn(*args)."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_timing_sweep():
    results = []

    for t in T_VALUES:
        for r in R_VALUES:
            Tr = t * r
            log.info(f"Testing T={t}, r={r}, Tr={Tr}")
            
            pico_times_all = []
            wbp_times_all  = []
            pico_mem_all   = []
            wbp_mem_all    = []
            
            for seed in SEEDS:
                torch.manual_seed(seed)
                B_all = torch.randn(D_OUT, Tr, dtype=DTYPE, device="cuda")

                # ----- Pico timing -----
                pico_iter_times = time_fn(pico_calibrate, B_all, t)
                pico_times_all.extend(pico_iter_times)

                # ----- WBP timing -----
                wbp_iter_times = time_fn(wbp_calibrate, B_all, t)
                wbp_times_all.extend(wbp_iter_times)

                # ----- Peak memory -----
                pico_mem_all.append(measure_peak_mem(pico_calibrate, B_all, t))
                wbp_mem_all.append(measure_peak_mem(wbp_calibrate, B_all, t))
            
            pico_mean = statistics.mean(pico_times_all)
            pico_std  = statistics.stdev(pico_times_all)
            wbp_mean  = statistics.mean(wbp_times_all)
            wbp_std   = statistics.stdev(wbp_times_all)
            speedup   = pico_mean / wbp_mean if wbp_mean > 0 else float("nan")
            
            log.info(
                f"  Pico: {pico_mean*1e3:.3f} ms \u00b1 {pico_std*1e3:.3f} ms | "
                f"WBP: {wbp_mean*1e3:.3f} ms \u00b1 {wbp_std*1e3:.3f} ms | Speedup: {speedup:.2f}x"
            )
            
            results.append({
                "T": t,
                "r": r,
                "Tr": Tr,
                "pico": {
                    "mean_time_s": pico_mean,
                    "std_time_s": pico_std,
                    "peak_mem_bytes": int(statistics.mean(pico_mem_all))
                },
                "wbp": {
                    "mean_time_s": wbp_mean,
                    "std_time_s": wbp_std,
                    "peak_mem_bytes": int(statistics.mean(wbp_mem_all))
                },
                "speedup": speedup
            })

    return results

def generate_plots(results):
    import numpy as np
    
    # 1. Speedup Heatmap
    Z = np.zeros((len(R_VALUES), len(T_VALUES)))
    for i, r in enumerate(R_VALUES):
        for j, t in enumerate(T_VALUES):
            # Find matching result
            res = next(item for item in results if item["T"] == t and item["r"] == r)
            Z[i, j] = res["speedup"]
            
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use RdYlGn colormap centered at 1.0 (or symmetrically around 1.0)
    import matplotlib.colors as mcolors
    # Center map around 1.0
    vmin, vmax = Z.min(), Z.max()
    divnorm = mcolors.TwoSlopeNorm(vmin=min(vmin, 0.9), vcenter=1.0, vmax=max(vmax, 1.1))
    
    cax = ax.imshow(Z, origin="lower", cmap="RdYlGn", aspect="auto", norm=divnorm)
    
    ax.set_xticks(np.arange(len(T_VALUES)))
    ax.set_yticks(np.arange(len(R_VALUES)))
    ax.set_xticklabels(T_VALUES)
    ax.set_yticklabels(R_VALUES)
    ax.set_xlabel("Number of Tasks (T)", fontsize=12)
    ax.set_ylabel("Rank (r)", fontsize=12)
    ax.set_title("Speedup Ratio (Pico Time / WBP Time)", fontsize=14)
    
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Speedup (x)")
    
    for i in range(len(R_VALUES)):
        for j in range(len(T_VALUES)):
            val = Z[i, j]
            ax.text(j, i, f"{val:.2f}x", ha="center", va="center", color="black")
            
    plt.tight_layout()
    heatmap_path = RESULTS_DIR / "speedup_heatmap.png"
    fig.savefig(heatmap_path, dpi=300)
    plt.close(fig)
    log.info(f"Speedup heatmap saved to {heatmap_path}")
    
    # 2. Absolute timing comparison (Line plots per r)
    for r in R_VALUES:
        r_results = [item for item in results if item["r"] == r]
        # Sort by T
        r_results.sort(key=lambda x: x["T"])
        
        t_vals = [res["T"] for res in r_results]
        pico_ms = [res["pico"]["mean_time_s"] * 1e3 for res in r_results]
        pico_std = [res["pico"]["std_time_s"] * 1e3 for res in r_results]
        wbp_ms = [res["wbp"]["mean_time_s"] * 1e3 for res in r_results]
        wbp_std = [res["wbp"]["std_time_s"] * 1e3 for res in r_results]
        
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(t_vals, pico_ms, yerr=pico_std, label="Pico (SVD)",
                    color="steelblue", marker="o", capsize=4, linewidth=2)
        ax.errorbar(t_vals, wbp_ms, yerr=wbp_std, label="WBP (Woodbury)",
                    color="darkorange", marker="s", capsize=4, linewidth=2)
                    
        ax.set_xlabel("Number of Tasks (T)", fontsize=12)
        ax.set_ylabel("Wall-clock time (ms)", fontsize=12)
        ax.set_title(f"Wall-Clock Time vs. T (r={r}, D_OUT={D_OUT}, float32)")
        ax.legend(fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(t_vals)
        
        plt.tight_layout()
        plot_path = RESULTS_DIR / f"time_vs_T_r{r}.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        log.info(f"Line plot for r={r} saved to {plot_path}")

def main():
    if not torch.cuda.is_available():
        log.error("CUDA is not available. E4 Timing requires the Lab RTX 6000.")
        raise SystemExit(1)
        
    device_name = torch.cuda.get_device_name(0)
    log.info(f"GPU: {device_name}")
    log.info("Starting E4 Timing Sweep...")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = run_timing_sweep()
    
    output = {
        "experiment": "E4_timing",
        "hardware": device_name,
        "d_out": D_OUT,
        "dtype": "float32",
        "warmup_iters": WARMUP_ITERS,
        "timed_iters": TIMED_ITERS,
        "seeds": SEEDS,
        "results": results
    }
    
    json_path = RESULTS_DIR / "results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Results written to {json_path}")
    
    try:
        generate_plots(results)
    except Exception as e:
        log.warning(f"Plot generation failed: {e}")
        
    log.info("E4 Timing complete.")

if __name__ == "__main__":
    main()
