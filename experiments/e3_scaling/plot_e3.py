import json
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    results_dir = Path("results/e3")
    summary_path = results_dir / "results.json"
    
    if not summary_path.exists():
        print(f"File not found: {summary_path}")
        return

    with open(summary_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("No results found.")
        return

    d_outs = [r["d_out"] for r in results]
    
    pico_time = [r["pico"]["mean_time_s"] * 1000 for r in results] # ms
    pico_std = [r["pico"]["std_time_s"] * 1000 for r in results]
    wbp_time = [r["wbp"]["mean_time_s"] * 1000 for r in results]
    wbp_std = [r["wbp"]["std_time_s"] * 1000 for r in results]
    
    pico_mem = [r["pico"]["peak_mem_bytes"] / (1024**2) for r in results] # MB
    wbp_mem = [r["wbp"]["peak_mem_bytes"] / (1024**2) for r in results]

    # Timing Plot (Log-Log)
    plt.figure(figsize=(8, 6))
    plt.errorbar(d_outs, pico_time, yerr=pico_std, label="Pico (SVD)", marker="o", capsize=4, color="blue")
    plt.errorbar(d_outs, wbp_time, yerr=wbp_std, label="WBP (Woodbury)", marker="s", capsize=4, color="orange")
    
    plt.xscale("log", base=2)
    plt.yscale("log", base=10)
    plt.xlabel("d_out (Output Dimension)")
    plt.ylabel("Wall-clock time (ms)")
    
    T = data.get("T", "N/A")
    r = data.get("r", "N/A")
    hw = data.get("hardware", "N/A")
    plt.title(f"Calibration Wall-Clock Time vs. d_out\n(T={T}, r={r}, {hw})")
    
    plt.xticks(d_outs, labels=[str(d) for d in d_outs])
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    timing_out = results_dir / "timing_plot.png"
    plt.savefig(timing_out, dpi=300)
    print(f"Saved {timing_out}")
    plt.close()

    # Memory Plot (Linear)
    plt.figure(figsize=(8, 6))
    plt.plot(d_outs, pico_mem, label="Pico (SVD)", marker="o", color="blue")
    plt.plot(d_outs, wbp_mem, label="WBP (Woodbury)", marker="s", color="orange")
    
    plt.xlabel("d_out (Output Dimension)")
    plt.ylabel("Peak GPU Memory (MB)")
    plt.title(f"Peak GPU Memory vs. d_out\n(T={T}, r={r}, {hw})")
    
    plt.xticks(d_outs, labels=[str(d) for d in d_outs])
    plt.legend()
    plt.grid(True, ls="--", alpha=0.7)
    
    mem_out = results_dir / "memory_plot.png"
    plt.savefig(mem_out, dpi=300)
    print(f"Saved {mem_out}")
    plt.close()

if __name__ == "__main__":
    main()
