import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_metric(data, dataset, metric_key):
    if dataset in data:
        return data[dataset].get(metric_key, 0.0)
    return 0.0

def main():
    results_dir = Path("results/e2")
    agg_path = results_dir / "aggregated_results.json"
    
    if not agg_path.exists():
        print(f"File not found: {agg_path}")
        return

    with open(agg_path, "r") as f:
        data = json.load(f)["results"]
        
    datasets = [
        ("gsm8k", "exact_match,strict-match", "GSM8K (Math)"),
        ("humaneval", "pass@1,create_test", "HumanEval (Code)"),
        ("mmlu_high_school_macroeconomics_generative", "exact_match,get_response", "MMLU (Finance)"),
        ("medqa_4options", "acc_norm,none", "MedQA (Medical)")
    ]

    methods_ta = ["no_cal_ta", "pico_ta", "wbp_ta"]
    methods_ties = ["no_cal_ties", "pico_ties", "wbp_ties"]
    
    # Plot for Task Arithmetic
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, (methods, title) in enumerate([(methods_ta, "Task Arithmetic"), (methods_ties, "TIES")]):
        ax = axes[i]
        bar_width = 0.25
        x = np.arange(len(datasets))
        
        # Check which methods are actually present
        present_methods = [m for m in methods if m in data]
        if not present_methods:
            continue
            
        for j, method in enumerate(present_methods):
            scores = [get_metric(data[method], ds, mk) for ds, mk, _ in datasets]
            offset = (j - len(present_methods)/2.0 + 0.5) * bar_width
            ax.bar(x + offset, scores, bar_width, label=method)
            
        ax.set_title(title)
        ax.set_ylabel("Score / Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels([d[2] for d in datasets], rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_path = results_dir / "accuracy_plot.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
