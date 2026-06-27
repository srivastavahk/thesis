import json
import matplotlib.pyplot as plt
from pathlib import Path
import re

def main():
    results_dir = Path("results/e1")
    summary_path = results_dir / "results.json"
    
    if not summary_path.exists():
        print(f"File not found: {summary_path}")
        return

    with open(summary_path, "r") as f:
        data = json.load(f)

    per_layer = data.get("per_layer_results", [])
    if not per_layer:
        print("No per-layer results found.")
        return

    # Organize data by module type
    modules = {"q_proj": {"layers": [], "max_rel_error": [], "max_abs_error": []},
               "v_proj": {"layers": [], "max_rel_error": [], "max_abs_error": []}}
               
    for d in per_layer:
        layer_name = d["layer"]
        # Extract layer index and module type (e.g. q_proj or v_proj)
        match = re.search(r'layers\.(\d+)\..*\.(q_proj|v_proj)', layer_name)
        if match:
            l_idx = int(match.group(1))
            mod = match.group(2)
            
            modules[mod]["layers"].append(l_idx)
            modules[mod]["max_rel_error"].append(d["max_rel_error"])
            modules[mod]["max_abs_error"].append(d["max_abs_error"])

    # Sort each module's data by layer index
    for mod in modules:
        combined = list(zip(modules[mod]["layers"], modules[mod]["max_rel_error"], modules[mod]["max_abs_error"]))
        combined.sort(key=lambda x: x[0])
        modules[mod]["layers"] = [x[0] for x in combined]
        modules[mod]["max_rel_error"] = [x[1] for x in combined]
        modules[mod]["max_abs_error"] = [x[2] for x in combined]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    
    for i, mod in enumerate(["q_proj", "v_proj"]):
        ax = axes[i]
        
        ax.plot(modules[mod]["layers"], modules[mod]["max_rel_error"], 
                label="Max Relative Error", marker='o', color='red', linestyle='-')
        ax.plot(modules[mod]["layers"], modules[mod]["max_abs_error"], 
                label="Max Absolute Error", marker='x', color='blue', linestyle='--')
        
        ax.set_title(f"Operator-Level Equivalence ({mod})")
        ax.set_xlabel("Transformer Layer Index")
        ax.set_ylabel("Error (Pico vs WBP)")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()
        
        # Set x-ticks properly
        if modules[mod]["layers"]:
            ax.set_xticks(modules[mod]["layers"][::4]) # Show every 4th layer to avoid clutter
        
    plt.suptitle("Layer-wise Precision Differences (Pico vs WBP)", fontsize=16)
    plt.tight_layout()
    
    out_path = results_dir / "layerwise_error.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
