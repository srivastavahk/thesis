import json
from pathlib import Path

def main():
    results_dir = Path("results/e5")
    agg_path = results_dir / "results.json"
    
    if not agg_path.exists():
        print(f"File not found: {agg_path}")
        return

    with open(agg_path, "r") as f:
        data = json.load(f)
        
    baselines = data.get("baselines", {})
    results = data.get("results", [])

    if not results:
        print("No results found.")
        return

    # Markdown Table
    print("### E5 Beta Sweep Downstream Accuracy")
    print()
    header = "| Beta | GSM8K | HumanEval | FinQA | MedMCQA | Average |"
    sep = "|---|---|---|---|---|---|"
    print(header)
    print(sep)
    
    for r in results:
        beta = r.get("beta", "N/A")
        gsm8k = r.get("gsm8k", 0.0)
        humaneval = r.get("humaneval", 0.0)
        finqa = r.get("finqa_exact_match", 0.0)
        medmcqa = r.get("medmcqa_accuracy", 0.0)
        avg = r.get("average", 0.0)
        
        row = [
            f"{beta}",
            f"{gsm8k:.4f}",
            f"{humaneval:.4f}",
            f"{finqa:.4f}",
            f"{medmcqa:.4f}",
            f"{avg:.4f}"
        ]
        print("| " + " | ".join(row) + " |")

    print("\n**Baselines (Average across all tasks):**")
    if "no_cal_average" in baselines:
        print(f"- Task Arithmetic (no cal): {baselines['no_cal_average']:.4f}")
    if "pico_average" in baselines:
        print(f"- Pico (SVD): {baselines['pico_average']:.4f}")

    # Save to file
    out_path = results_dir / "accuracy_table.md"
    with open(out_path, "w") as f:
        f.write("### E5 Beta Sweep Downstream Accuracy\n\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        
        for r in results:
            beta = r.get("beta", "N/A")
            gsm8k = r.get("gsm8k", 0.0)
            humaneval = r.get("humaneval", 0.0)
            finqa = r.get("finqa_exact_match", 0.0)
            medmcqa = r.get("medmcqa_accuracy", 0.0)
            avg = r.get("average", 0.0)
            
            row = [
                f"{beta}",
                f"{gsm8k:.4f}",
                f"{humaneval:.4f}",
                f"{finqa:.4f}",
                f"{medmcqa:.4f}",
                f"{avg:.4f}"
            ]
            f.write("| " + " | ".join(row) + " |\n")
            
        f.write("\n**Baselines (Average across all tasks):**\n")
        if "no_cal_average" in baselines:
            f.write(f"- Task Arithmetic (no cal): {baselines['no_cal_average']:.4f}\n")
        if "pico_average" in baselines:
            f.write(f"- Pico (SVD): {baselines['pico_average']:.4f}\n")
            
    print(f"\nSaved table to {out_path}")

if __name__ == "__main__":
    main()
