import json
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
        ("gsm8k", "exact_match,strict-match", "GSM8K"),
        ("humaneval", "pass@1,create_test", "HumanEval"),
        ("mmlu_high_school_macroeconomics_generative", "exact_match,get_response", "MMLU (Fin)"),
        ("medqa_4options", "acc_norm,none", "MedQA")
    ]

    # Available methods
    methods = ["no_cal_ta", "pico_ta", "wbp_ta", "no_cal_ties", "pico_ties", "wbp_ties"]
    present_methods = [m for m in methods if m in data]
    
    if not present_methods:
        print("No methods found in aggregated results.")
        return

    # Markdown Table
    print("### E2 Downstream Accuracy")
    print()
    header = "| Method | " + " | ".join([d[2] for d in datasets]) + " | Average |"
    sep = "|---|" + "|".join(["---"] * len(datasets)) + "|---|"
    print(header)
    print(sep)
    
    for method in present_methods:
        row = [method]
        total = 0
        for ds, mk, _ in datasets:
            score = get_metric(data[method], ds, mk)
            row.append(f"{score:.4f}")
            total += score
        avg = total / len(datasets)
        row.append(f"{avg:.4f}")
        print("| " + " | ".join(row) + " |")

    # Also save to file
    out_path = results_dir / "accuracy_table.md"
    with open(out_path, "w") as f:
        f.write("### E2 Downstream Accuracy\n\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for method in present_methods:
            row = [method]
            total = 0
            for ds, mk, _ in datasets:
                score = get_metric(data[method], ds, mk)
                row.append(f"{score:.4f}")
                total += score
            avg = total / len(datasets)
            row.append(f"{avg:.4f}")
            f.write("| " + " | ".join(row) + " |\n")
            
    print(f"\nSaved table to {out_path}")

if __name__ == "__main__":
    main()
