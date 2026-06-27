import json
import matplotlib.pyplot as plt
from pathlib import Path

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

    layers = [d["layer"] for d in per_layer]
    max_rel_errors = [d["max_rel_error"] for d in per_layer]
    mean_rel_errors = [d.get("rel_error", d.get("mean_rel_error")) for d in per_layer]
    
    # Extract layer numbers
    layer_indices = list(range(len(layers)))
    
    plt.figure(figsize=(12, 6))
    plt.plot(layer_indices, max_rel_errors, label="Max Rel Error", marker='o', color='red')
    plt.plot(layer_indices, mean_rel_errors, label="Mean Rel Error", marker='x', color='blue')
    
    plt.title("Layer-wise Relative Error (Pico vs WBP)")
    plt.xlabel("Layer Index")
    plt.ylabel("Relative Error")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    out_path = results_dir / "layerwise_error.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
