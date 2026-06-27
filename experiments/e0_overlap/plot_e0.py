import json
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    results_dir = Path("results/e0")
    summary_path = results_dir / "overlap_summary.json"
    
    if not summary_path.exists():
        print(f"File not found: {summary_path}")
        return

    with open(summary_path, "r") as f:
        data = json.load(f)

    # Plot q_proj and v_proj separately or in subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, proj in enumerate(["q_proj", "v_proj"]):
        ax = axes[i]
        proj_data = data.get(proj, [])
        if not proj_data:
            continue
            
        ranks = [d["rank"] for d in proj_data]
        mean_o_a = [d["mean_O_A"] for d in proj_data]
        mean_o_b = [d["mean_O_B"] for d in proj_data]
        gaps = [d["gap"] for d in proj_data]
        
        ax.plot(ranks, mean_o_a, label="O_A (Gram Matrix)", marker='o')
        ax.plot(ranks, mean_o_b, label="O_B (Weight Matrix)", marker='s')
        ax.plot(ranks, gaps, label="Gap", marker='^', linestyle='--')
        
        ax.set_title(f"Overlap vs Rank ({proj})")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Overlap Metric")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ranks)
        ax.set_xticklabels([str(r) for r in ranks])
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_path = results_dir / "overlap_vs_rank.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
