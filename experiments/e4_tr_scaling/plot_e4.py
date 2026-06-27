import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    results_dir = Path("results/e4/timing")
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

    # Extract unique T and r values
    Ts = sorted(list(set(r["T"] for r in results)))
    Rs = sorted(list(set(r["r"] for r in results)))
    
    # Heatmap
    speedup_matrix = np.zeros((len(Rs), len(Ts)))
    for r_item in results:
        t_idx = Ts.index(r_item["T"])
        r_idx = Rs.index(r_item["r"])
        speedup_matrix[r_idx, t_idx] = r_item["speedup"]

    plt.figure(figsize=(8, 6))
    plt.imshow(speedup_matrix, origin="lower", cmap="RdYlGn", vmin=0.5, vmax=max(2.0, np.max(speedup_matrix)))
    plt.colorbar(label="Speedup (Pico Time / WBP Time)")
    
    plt.xticks(np.arange(len(Ts)), labels=[str(t) for t in Ts])
    plt.yticks(np.arange(len(Rs)), labels=[str(r) for r in Rs])
    plt.xlabel("Number of Tasks (T)")
    plt.ylabel("LoRA Rank (r)")
    
    hw = data.get("hardware", "N/A")
    d_out = data.get("d_out", "N/A")
    plt.title(f"WBP Speedup vs Pico\n({hw}, d_out={d_out})")
    
    for i in range(len(Rs)):
        for j in range(len(Ts)):
            val = speedup_matrix[i, j]
            plt.text(j, i, f"{val:.1f}x", ha="center", va="center", color="black" if 0.8 < val < 1.5 else "white")
            
    heatmap_out = results_dir / "01_speedup_heatmap.png"
    plt.savefig(heatmap_out, dpi=300)
    print(f"Saved {heatmap_out}")
    plt.close()

    # Timing lines per r
    for r_val in Rs:
        t_vals = []
        pico_times = []
        wbp_times = []
        
        for r_item in results:
            if r_item["r"] == r_val:
                t_vals.append(r_item["T"])
                pico_times.append(r_item["pico"]["mean_time_s"] * 1000)
                wbp_times.append(r_item["wbp"]["mean_time_s"] * 1000)
                
        # Sort by T
        sorted_indices = np.argsort(t_vals)
        t_vals = np.array(t_vals)[sorted_indices]
        pico_times = np.array(pico_times)[sorted_indices]
        wbp_times = np.array(wbp_times)[sorted_indices]
        
        plt.figure(figsize=(8, 6))
        plt.plot(t_vals, pico_times, label="Pico (SVD)", marker="o", color="blue")
        plt.plot(t_vals, wbp_times, label="WBP (Woodbury)", marker="s", color="orange")
        
        plt.xlabel("Number of Tasks (T)")
        plt.ylabel("Wall-clock time (ms)")
        plt.title(f"Time vs T (r={r_val}, {hw})")
        plt.xticks(t_vals)
        plt.legend()
        plt.grid(True, ls="--", alpha=0.7)
        
        line_out = results_dir / f"time_vs_T_r{r_val}.png"
        plt.savefig(line_out, dpi=300)
        print(f"Saved {line_out}")
        plt.close()

if __name__ == "__main__":
    main()
