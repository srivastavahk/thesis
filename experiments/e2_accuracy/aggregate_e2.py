import json
from pathlib import Path
import os

def main():
    results_dir = Path("results/e2")
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        return

    # Base dictionary for aggregated results
    aggregated = {
        "experiment": "E2",
        "results": {}
    }

    # Evaluate all eval_*.json files in the directory
    for file in results_dir.glob("eval_*.json"):
        # Expecting filenames like eval_no_cal_ta.json
        filename = file.stem
        parts = filename.split("_")
        if len(parts) >= 2:
            # Assuming format: eval_{method}_{merge_strategy} or similar
            # E.g., eval_no_cal_ta -> method='no_cal', strategy='ta'
            # Or just use the filename without 'eval_' as the key
            method_key = filename.replace("eval_", "")
            
            with open(file, "r") as f:
                try:
                    data = json.load(f)
                    aggregated["results"][method_key] = data
                except json.JSONDecodeError:
                    print(f"Failed to parse {file}")
                    
    out_path = results_dir / "aggregated_results.json"
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)
        
    print(f"Saved aggregated results to {out_path}")

if __name__ == "__main__":
    main()
