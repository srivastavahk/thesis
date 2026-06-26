import os
import sys
import json
import logging
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

BETA_VALUES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]

def run_cmd(cmd_list, description):
    logging.info(f"\n{'='*60}")
    logging.info(f"RUNNING: {description}")
    logging.info(f"CMD: {' '.join(str(c) for c in cmd_list)}")
    logging.info(f"{'='*60}")
    
    result = subprocess.run(cmd_list)
    if result.returncode != 0:
        logging.error(f"\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def extract_scores(eval_json_path):
    if not Path(eval_json_path).exists():
        return None
    with open(eval_json_path, "r") as f:
        data = json.load(f)
        
    gsm8k = data.get("gsm8k", {}).get("exact_match,strict-match", 0.0)
    humaneval = data.get("humaneval", {}).get("pass@1,create_test", 0.0)
    macro = data.get("mmlu_high_school_macroeconomics_generative", {}).get("exact_match,get_response", 0.0)
    medqa = data.get("medqa_4options", {}).get("acc,none", 0.0)
    
    average = (gsm8k + humaneval + macro + medqa) / 4.0
    return {
        "gsm8k": gsm8k,
        "humaneval": humaneval,
        "finqa_exact_match": macro,  # Mapping to MMLU Macroeconomics for E2 consistency
        "medmcqa_accuracy": medqa,
        "average": average
    }

def main():
    parser = argparse.ArgumentParser(description="Master Coordinator for E5 Phase 4 Generalization (Beta-sweep)")
    parser.add_argument("--adapters_dir", type=Path, default=Path("adapters"), help="Path to domain adapters")
    parser.add_argument("--base_model", type=str, default="unsloth/Meta-Llama-3.1-8B", help="Base model path or name")
    parser.add_argument("--calibrated_dir", type=Path, default=Path("calibrated-adapters"), help="Path for calibrated adapters output")
    parser.add_argument("--merged_dir", type=Path, default=Path("merged-adapters"), help="Path for merged weights output")
    parser.add_argument("--e2_results_dir", type=Path, default=Path("results/e2"), help="Path to E2 results directory to overlay baselines")
    parser.add_argument("--output_dir", type=Path, default=Path("results/e5"), help="Output directory for E5 results and plots")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    python_exec = sys.executable
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Define domains
    domains = [d.name for d in args.adapters_dir.iterdir() if d.is_dir()]
    if not domains:
        logging.error(f"No adapters found in {args.adapters_dir}")
        sys.exit(1)

    logging.info(f"Detected domains: {domains}")
    logging.info(f"Sweeping Beta values: {BETA_VALUES}")

    # -------------------------------------------------------------------------
    # Execution Loop for each beta
    # -------------------------------------------------------------------------
    beta_results = []
    
    for b in BETA_VALUES:
        beta_str = f"beta_{str(b).replace('.', '_')}"
        wbp_out = args.calibrated_dir / f"wbp_{beta_str}"
        merged_pt = args.merged_dir / f"wbp_{beta_str}_ta.pt"
        eval_json = args.output_dir / f"eval_wbp_{beta_str}_ta.json"
        
        # 1. Calibrate
        wbp_done = True
        for d in domains:
            if not (wbp_out / d / "adapter_model.safetensors").exists():
                wbp_done = False
                break
        
        if not wbp_done:
            run_cmd([python_exec, "src/wbp.py", 
                     "--adapters_dir", str(args.adapters_dir), 
                     "--output_dir", str(wbp_out),
                     "--beta", str(b)], 
                    f"Calibrating adapters using WBP (beta={b})")
        else:
            logging.info(f"[SKIP] WBP calibration (beta={b}) already completed.")
            
        # 2. Merge (Task Arithmetic)
        if not merged_pt.exists():
            run_cmd([python_exec, "src/ties.py",
                     "--adapters_dir", str(wbp_out),
                     "--output_file", str(merged_pt),
                     "--method", "ta"],
                    f"Merging WBP (beta={b}) via Task Arithmetic")
        else:
            logging.info(f"[SKIP] Merging WBP (beta={b}) already completed.")
            
        # 3. Evaluate
        if not eval_json.exists():
            run_cmd([python_exec, "experiments/e2_accuracy/new_evaluation.py",
                     "--base_model", args.base_model,
                     "--merged_path", str(merged_pt),
                     "--output_file", str(eval_json)],
                    f"Evaluating WBP (beta={b})")
        else:
            logging.info(f"[SKIP] Evaluation WBP (beta={b}) already completed.")
            
    logging.info("E5 Execution Complete. Run aggregate_e5.py next to compile results.")

if __name__ == "__main__":
    main()
