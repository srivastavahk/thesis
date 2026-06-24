"""
E3 — Wall-Clock & Memory Scaling vs. d_out
===========================================
Platform : Lab RTX 6000 (24 GB VRAM, CUDA) — CUDA required for CUDA-synchronized timing
Purpose  : Measure actual wall-clock time and peak GPU memory for Pico (thin SVD)
           and WBP (Gram + Woodbury) as d_out grows, using synthetic matrices.

Usage
-----
    python experiments/e3_scaling/run_e3.py

Results land in ./results/e3/:
    results.json
    timing_plot.png
    memory_plot.png

No adapter files or base model needed — all synthetic.
"""

import json
import logging
import os
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
# Configuration (hardcoded per AGENT_PROMPT spec)
# ---------------------------------------------------------------------------
D_OUT_VALUES  = [512, 1024, 2048, 4096, 8192]
T             = 4
R             = 16           # Tr = T * R = 64
WARMUP_ITERS  = 5
TIMED_ITERS   = 20
DTYPE         = torch.float32
SEEDS         = [42, 43, 44, 45, 46]
RESULTS_DIR   = Path("./results/e3")


# ---------------------------------------------------------------------------
# Inline calibration functions (no src/ import — avoids Python overhead in timing)
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
# Timing harness
# ---------------------------------------------------------------------------

def time_fn(fn, *args, warmup: int = WARMUP_ITERS, iters: int = TIMED_ITERS):
    """
    Returns a list of per-iteration wall-clock times (seconds).
    GPU-synchronized before stopping the clock, per AGENT.md §7 requirements.
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

def run_sweep() -> list:
    results = []

    for d_out in D_OUT_VALUES:
        log.info("d_out = %d  (T=%d, r=%d, Tr=%d)", d_out, T, R, T * R)
        Tr = T * R

        pico_times_all = []
        wbp_times_all  = []
        pico_mem_all   = []
        wbp_mem_all    = []

        for seed in SEEDS:
            torch.manual_seed(seed)
            B_all = torch.randn(d_out, Tr, dtype=DTYPE, device="cuda")

            # ----- Pico timing -----
            pico_iter_times = time_fn(pico_calibrate, B_all, T)
            pico_times_all.extend(pico_iter_times)

            # ----- WBP timing -----
            wbp_iter_times = time_fn(wbp_calibrate, B_all, T)
            wbp_times_all.extend(wbp_iter_times)

            # ----- Peak memory (one shot per seed, after timing) -----
            pico_mem_all.append(measure_peak_mem(pico_calibrate, B_all, T))
            wbp_mem_all.append(measure_peak_mem(wbp_calibrate, B_all, T))

        pico_mean = statistics.mean(pico_times_all)
        pico_std  = statistics.stdev(pico_times_all)
        wbp_mean  = statistics.mean(wbp_times_all)
        wbp_std   = statistics.stdev(wbp_times_all)
        speedup   = pico_mean / wbp_mean if wbp_mean > 0 else float("nan")

        log.info(
            "  Pico: %.3f ms ± %.3f ms  |  WBP: %.3f ms ± %.3f ms  |  Speedup: %.2fx",
            pico_mean * 1e3, pico_std * 1e3,
            wbp_mean  * 1e3, wbp_std  * 1e3,
            speedup,
        )

        results.append({
            "d_out":   d_out,
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
# Plot generation
# ---------------------------------------------------------------------------

def make_plots(results: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d_outs     = [r["d_out"]               for r in results]
    pico_ms    = [r["pico"]["mean_time_s"] * 1e3 for r in results]
    pico_std   = [r["pico"]["std_time_s"]  * 1e3 for r in results]
    wbp_ms     = [r["wbp"]["mean_time_s"]  * 1e3 for r in results]
    wbp_std    = [r["wbp"]["std_time_s"]   * 1e3 for r in results]
    pico_mb    = [r["pico"]["peak_mem_bytes"] / 1e6 for r in results]
    wbp_mb     = [r["wbp"]["peak_mem_bytes"]  / 1e6 for r in results]

    # ---- Timing plot (log-log) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(d_outs, pico_ms, yerr=pico_std, label="Pico (SVD)",
                color="steelblue", marker="o", capsize=4, linewidth=2)
    ax.errorbar(d_outs, wbp_ms,  yerr=wbp_std,  label="WBP (Woodbury)",
                color="darkorange", marker="s", capsize=4, linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("d_out", fontsize=12)
    ax.set_ylabel("Wall-clock time (ms)", fontsize=12)
    ax.set_title(f"Calibration Wall-Clock Time vs. d_out\n(T={T}, r={R}, RTX 6000, float32)")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_xticks(d_outs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.tight_layout()
    timing_path = RESULTS_DIR / "timing_plot.png"
    fig.savefig(timing_path, dpi=300)
    plt.close(fig)
    log.info("Timing plot saved to %s", timing_path)

    # ---- Memory plot (linear axes) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(d_outs, pico_mb, label="Pico (SVD)",    color="steelblue",  marker="o", linewidth=2)
    ax.plot(d_outs, wbp_mb,  label="WBP (Woodbury)", color="darkorange", marker="s", linewidth=2)
    ax.set_xlabel("d_out", fontsize=12)
    ax.set_ylabel("Peak GPU memory (MB)", fontsize=12)
    ax.set_title(f"Peak GPU Memory vs. d_out\n(T={T}, r={R}, RTX 6000, float32)")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    mem_path = RESULTS_DIR / "memory_plot.png"
    fig.savefig(mem_path, dpi=300)
    plt.close(fig)
    log.info("Memory plot saved to %s", mem_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        log.error(
            "CUDA is not available. E3 requires the Lab RTX 6000. "
            "MPS (Mac) results are not acceptable as primary thesis numbers. Exiting."
        )
        raise SystemExit(1)

    device_name = torch.cuda.get_device_name(0)
    log.info("GPU: %s", device_name)
    log.info("E3 sweep: d_out=%s  T=%d  r=%d  seeds=%s  warmup=%d  iters=%d",
             D_OUT_VALUES, T, R, SEEDS, WARMUP_ITERS, TIMED_ITERS)

    results = run_sweep()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "experiment":   "E3",
        "hardware":     device_name,
        "dtype":        "float32",
        "T":            T,
        "r":            R,
        "Tr":           T * R,
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
        log.warning("Plot generation failed: %s  (results JSON is still valid)", e)

    log.info("E3 complete.")


if __name__ == "__main__":
    main()
