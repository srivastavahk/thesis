"""
E4 — Computational Efficiency vs. T and r
==========================================
Platform : Lab RTX 6000 (24 GB VRAM, CUDA)
Purpose  : Measure wall-clock time and peak GPU memory for Pico (SVD)
           and WBP (Woodbury) as the number of tasks (T) and rank (r) grow.

Usage
-----
    python experiments/e4_tr_scaling/run_e4_timing.py
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
# Inline calibration functions
# ---------------------------------------------------------------------------

def pico_calibrate(B_all: torch.Tensor, T: int) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(B_all, full_matrices=False)
    S_sq = S ** 2
    s = S_sq / S_sq.sum()
    alpha = 1.0 / (1.0 + (T - 1) * s)
    alpha_minus_1 = (alpha - 1.0).unsqueeze(0)
    return B_all + U @ (alpha_minus_1.T * (U.T @ B_all))

def wbp_calibrate(B_all: torch.Tensor, T: int) -> torch.Tensor:
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
                    "raw_times_ms": [t * 1e3 for t in pico_times_all], # Added for Boxplots
                },
                "wbp": {
                    "mean_time_s": wbp_mean,
                    "std_time_s": wbp_std,
                    "peak_mem_bytes": int(statistics.mean(wbp_mem_all)),
                    "raw_times_ms": [t * 1e3 for t in wbp_times_all], # Added for Boxplots
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

    # Pre-compute data structures for the plots
    speedup_matrix = np.zeros((len(R_VALUES), len(T_VALUES)))
    for res in results:
        t_idx = T_VALUES.index(res["T"])
        r_idx = R_VALUES.index(res["r"])
        speedup_matrix[r_idx, t_idx] = res["speedup"]

    # Guarantee colormap constraints (vmin < vcenter < vmax)
    vmin_val = min(speedup_matrix.min() * 0.9, 0.95)
    vmax_val = max(speedup_matrix.max() * 1.1, 1.05)
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=1.0, vmin=vmin_val, vmax=vmax_val)

    # -----------------------------------------------------------------------
    # 1. Heatmap (Original)
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(speedup_matrix, cmap="RdYlGn", norm=norm, origin="lower")

    ax.set_xticks(np.arange(len(T_VALUES)))
    ax.set_yticks(np.arange(len(R_VALUES)))
    ax.set_xticklabels(T_VALUES)
    ax.set_yticklabels(R_VALUES)
    ax.set_xlabel("Number of Tasks (T)", fontsize=12)
    ax.set_ylabel("LoRA Rank (r)", fontsize=12)
    ax.set_title(f"WBP Speedup vs Pico (d_out={D_OUT})", fontsize=12)

    for i in range(len(R_VALUES)):
        for j in range(len(T_VALUES)):
            val = speedup_matrix[i, j]
            color = "white" if val < 0.8 or val > 1.2 else "black"
            ax.text(j, i, f"{val:.2f}x", ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Speedup Ratio")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "01_speedup_heatmap.png", dpi=300)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 2. Line Plots (Original)
    # -----------------------------------------------------------------------
    for r in R_VALUES:
        r_results = [res for res in results if res["r"] == r]
        t_vals = [res["T"] for res in r_results]
        pico_ms = [res["pico"]["mean_time_s"] * 1e3 for res in r_results]
        pico_std = [res["pico"]["std_time_s"] * 1e3 for res in r_results]
        wbp_ms = [res["wbp"]["mean_time_s"] * 1e3 for res in r_results]
        wbp_std = [res["wbp"]["std_time_s"] * 1e3 for res in r_results]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(t_vals, pico_ms, yerr=pico_std, label="Pico", color="steelblue", marker="o", capsize=4)
        ax.errorbar(t_vals, wbp_ms, yerr=wbp_std, label="WBP", color="darkorange", marker="s", capsize=4)
        
        ax.set_xlabel("Number of Tasks (T)")
        ax.set_ylabel("Wall-clock time (ms)")
        ax.set_title(f"Calibration Time vs T (r={r})")
        ax.set_xticks(t_vals)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / f"02_time_vs_T_r{r}.png", dpi=300)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # 3. 3D Surface Plot
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    T_grid, R_grid = np.meshgrid(T_VALUES, R_VALUES)
    
    surf = ax.plot_surface(T_grid, R_grid, speedup_matrix, cmap="RdYlGn", norm=norm, edgecolor="k", alpha=0.9)
    ax.set_xlabel('Tasks (T)')
    ax.set_ylabel('Rank (r)')
    ax.set_zlabel('Speedup (x)')
    ax.set_title("3D Surface: Speedup Scaling")
    ax.set_xticks(T_VALUES)
    ax.set_yticks(R_VALUES)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Speedup Ratio")
    
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "03_speedup_surface_3d.png", dpi=300)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 4. Grouped Bar Chart
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(T_VALUES))
    width = 0.8 / len(R_VALUES) # Dynamic bar width based on number of r values
    
    for i, r in enumerate(R_VALUES):
        r_speedups = speedup_matrix[i, :]
        offset = x + (i - len(R_VALUES)/2 + 0.5) * width
        ax.bar(offset, r_speedups, width, label=f'r={r}')

    ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Break-even (1.0x)')
    ax.set_xlabel('Number of Tasks (T)', fontsize=12)
    ax.set_ylabel('Speedup Ratio', fontsize=12)
    ax.set_title('Grouped Bar Chart: Speedups across T and r', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(T_VALUES)
    ax.legend(title="LoRA Rank", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "04_grouped_bar_speedup.png", dpi=300)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 5. Box Plots (Distributions for a fixed r=32)
    # -----------------------------------------------------------------------
    target_r = 32
    if target_r in R_VALUES:
        fig, ax = plt.subplots(figsize=(9, 6))
        
        pico_data = [res["pico"]["raw_times_ms"] for res in results if res["r"] == target_r]
        wbp_data = [res["wbp"]["raw_times_ms"] for res in results if res["r"] == target_r]
        
        positions_pico = np.arange(len(T_VALUES)) * 2.0 - 0.4
        positions_wbp = np.arange(len(T_VALUES)) * 2.0 + 0.4
        
        box_pico = ax.boxplot(pico_data, positions=positions_pico, widths=0.6, patch_artist=True)
        box_wbp = ax.boxplot(wbp_data, positions=positions_wbp, widths=0.6, patch_artist=True)

        for patch in box_pico['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
        for patch in box_wbp['boxes']:
            patch.set_facecolor('darkorange')
            patch.set_alpha(0.7)

        ax.set_xticks(np.arange(len(T_VALUES)) * 2.0)
        ax.set_xticklabels(T_VALUES)
        ax.set_xlabel("Number of Tasks (T)", fontsize=12)
        ax.set_ylabel("Raw Execution Time (ms)", fontsize=12)
        ax.set_title(f"Time Distribution across Seeds (r={target_r})", fontsize=12)
        
        # Custom legend for Boxplot
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='steelblue', alpha=0.7, label='Pico'),
                           Patch(facecolor='darkorange', alpha=0.7, label='WBP')]
        ax.legend(handles=legend_elements)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / f"05_boxplot_distributions_r{target_r}.png", dpi=300)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # 6. Bubble Chart
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Flatten grid for scatter
    t_flat = []
    r_flat = []
    s_flat = []
    
    for i, r in enumerate(R_VALUES):
        for j, T in enumerate(T_VALUES):
            t_flat.append(T)
            r_flat.append(r)
            s_flat.append(speedup_matrix[i, j])
            
    scatter = ax.scatter(t_flat, r_flat, 
                         s=[s * 150 for s in s_flat], # Scale bubble size
                         c=s_flat, 
                         cmap="RdYlGn", norm=norm,
                         edgecolors="black", alpha=0.8)

    ax.set_xticks(T_VALUES)
    ax.set_yticks(R_VALUES)
    ax.set_xlabel("Number of Tasks (T)", fontsize=12)
    ax.set_ylabel("LoRA Rank (r)", fontsize=12)
    ax.set_title("Bubble Chart: Speedup Magnitude", fontsize=12)
    
    fig.colorbar(scatter, ax=ax, label="Speedup Ratio")
    ax.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "06_bubble_chart.png", dpi=300)
    plt.close(fig)

    log.info("Successfully generated all 6 plots in %s", RESULTS_DIR)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        log.error("CUDA is not available. E4 requires the Lab RTX 6000. Exiting.")
        raise SystemExit(1)

    device_name = torch.cuda.get_device_name(0)
    log.info("GPU: %s", device_name)
    log.info("E4 sweep: T=%s, r=%s, d_out=%d, seeds=%s", T_VALUES, R_VALUES, D_OUT, SEEDS)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_sweep()

    output = {
        "experiment": "E4_timing",
        "hardware": device_name,
        "d_out": D_OUT,
        "dtype": "float32",
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