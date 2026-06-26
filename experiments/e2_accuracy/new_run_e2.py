import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

def run_cmd(cmd_list, description):
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"CMD: {' '.join(str(c) for c in cmd_list)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd_list)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Master Coordinator for Disk-Based E2 Pipeline")
    parser.add_argument("--adapters_dir", type=Path, default=Path("adapters"))
    parser.add_argument("--base_model", type=str, default="unsloth/Meta-Llama-3.1-8B")
    parser.add_argument("--calibrated_dir", type=Path, default=Path("calibrated-adapters"))
    parser.add_argument("--merged_dir", type=Path, default=Path("merged-adapters"))
    parser.add_argument("--results_dir", type=Path, default=Path("results/e2"))
    args = parser.parse_args()

    python_exec = sys.executable

    # Define domains (expecting directories inside adapters_dir)
    domains = [d.name for d in args.adapters_dir.iterdir() if d.is_dir()]
    if not domains:
        print(f"[ERROR] No adapters found in {args.adapters_dir}")
        sys.exit(1)

    print(f"Detected domains: {domains}")

    # -------------------------------------------------------------------------
    # Phase 1: Calibration (Pico and WBP)
    # -------------------------------------------------------------------------
    pico_out = args.calibrated_dir / "pico"
    wbp_out = args.calibrated_dir / "wbp"
    
    # Check if Pico is already done
    pico_done = True
    for d in domains:
        if not (pico_out / d / "adapter_model.safetensors").exists():
            pico_done = False
            break
            
    if pico_done:
        print("\n[SKIP] Pico calibration already completed.")
    else:
        run_cmd([python_exec, "src/pico.py", 
                 "--adapters_dir", str(args.adapters_dir), 
                 "--output_dir", str(pico_out)], 
                "Calibrating adapters using Pico")

    # Check if WBP is already done
    wbp_done = True
    for d in domains:
        if not (wbp_out / d / "adapter_model.safetensors").exists():
            wbp_done = False
            break
            
    if wbp_done:
        print("\n[SKIP] WBP calibration already completed.")
    else:
        run_cmd([python_exec, "src/wbp.py", 
                 "--adapters_dir", str(args.adapters_dir), 
                 "--output_dir", str(wbp_out)], 
                "Calibrating adapters using WBP")


    # -------------------------------------------------------------------------
    # Phase 2: Merging (TA and TIES)
    # -------------------------------------------------------------------------
    merging_tasks = [
        {"name": "no_cal_ta",   "input": args.adapters_dir, "method": "ta"},
        {"name": "no_cal_ties", "input": args.adapters_dir, "method": "ties"},
        {"name": "pico_ta",     "input": pico_out,          "method": "ta"},
        {"name": "pico_ties",   "input": pico_out,          "method": "ties"},
        {"name": "wbp_ta",      "input": wbp_out,           "method": "ta"},
        {"name": "wbp_ties",    "input": wbp_out,           "method": "ties"},
    ]

    for task in merging_tasks:
        out_pt = args.merged_dir / f"{task['name']}.pt"
        if out_pt.exists():
            print(f"\n[SKIP] Merging {task['name']} already completed.")
        else:
            run_cmd([python_exec, "src/ties.py",
                     "--adapters_dir", str(task['input']),
                     "--output_file", str(out_pt),
                     "--method", task['method']],
                    f"Merging: {task['name']}")


    # -------------------------------------------------------------------------
    # Phase 3: Evaluation
    # -------------------------------------------------------------------------
    args.results_dir.mkdir(parents=True, exist_ok=True)
    all_scores = {}

    for task in merging_tasks:
        merged_pt = args.merged_dir / f"{task['name']}.pt"
        result_json = args.results_dir / f"eval_{task['name']}.json"

        if result_json.exists():
            print(f"\n[SKIP] Evaluation for {task['name']} already completed.")
            with open(result_json, "r") as f:
                all_scores[task['name']] = json.load(f)
        else:
            run_cmd([python_exec, "experiments/e2_accuracy/new_evaluation.py",
                     "--base_model", args.base_model,
                     "--merged_path", str(merged_pt),
                     "--output_file", str(result_json)],
                    f"Evaluating: {task['name']}")
            
            with open(result_json, "r") as f:
                all_scores[task['name']] = json.load(f)

    # -------------------------------------------------------------------------
    # Phase 4: Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("FINAL SUMMARY (All 6 Configurations)")
    print("="*60)
    print(f"{'Mode':<15} | {'GSM8K':<8} | {'HumanEv':<8} | {'MacroEcon':<9} | {'MedQA':<8}")
    print(f"{'Mode':<15} | {'GSM8K':<8} | {'HumanEv':<8} | {'MacroEcon':<9} | {'MedQA':<8}")
    print("-"*60)
    
    for name, s in all_scores.items():
        gsm8k = s.get("gsm8k", {}).get("exact_match,none-0", 0.0)
        humaneval = s.get("humaneval", {}).get("pass@1,none-0", 0.0)
        macro = s.get("mmlu_high_school_macroeconomics_generative", {}).get("exact_match,none-0", 0.0)
        medqa = s.get("medqa_4options", {}).get("acc,none-0", 0.0)
        
        print(f"{name:<15} | {gsm8k:<8.4f} | {humaneval:<8.4f} | {macro:<9.4f} | {medqa:<8.4f}")

if __name__ == "__main__":
    main()
