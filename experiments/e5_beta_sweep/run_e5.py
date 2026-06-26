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
        
    gsm8k = data.get("gsm8k", {}).get("exact_match,none-0", 0.0)
    humaneval = data.get("humaneval", {}).get("pass@1,none-0", 0.0)
    macro = data.get("mmlu_high_school_macroeconomics_generative", {}).get("exact_match,none-0", 0.0)
    medqa = data.get("medqa_4options", {}).get("acc,none-0", 0.0)
    
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
            
        # Extract Scores
        scores = extract_scores(eval_json)
        if scores:
            scores["beta"] = b
            beta_results.append(scores)
        else:
            logging.error(f"Failed to load scores for beta={b} from {eval_json}")

    # -------------------------------------------------------------------------
    # Baseline Extractions
    # -------------------------------------------------------------------------
    no_cal_json = args.e2_results_dir / "eval_no_cal_ta.json"
    pico_json = args.e2_results_dir / "eval_pico_ta.json"
    
    no_cal_scores = extract_scores(no_cal_json)
    pico_scores = extract_scores(pico_json)
    
    if no_cal_scores is None:
        logging.warning(f"Could not find no-cal baseline at {no_cal_json}. Using 0.0.")
        no_cal_scores = {"average": 0.0, "gsm8k": 0.0, "humaneval": 0.0, "finqa_exact_match": 0.0, "medmcqa_accuracy": 0.0}
        
    if pico_scores is None:
        logging.warning(f"Could not find Pico baseline at {pico_json}. Using 0.0.")
        pico_scores = {"average": 0.0, "gsm8k": 0.0, "humaneval": 0.0, "finqa_exact_match": 0.0, "medmcqa_accuracy": 0.0}

    # -------------------------------------------------------------------------
    # Generate Output JSON
    # -------------------------------------------------------------------------
    results_json = {
        "experiment": "E5",
        "hardware": "RTX 6000 24GB",
        "base_model": args.base_model,
        "dtype": args.dtype,
        "T": len(domains),
        "seed": args.seed,
        "beta_values": BETA_VALUES,
        "baselines": {
            "no_cal_average": no_cal_scores["average"],
            "pico_average": pico_scores["average"]
        },
        "results": beta_results
    }
    
    out_json_path = args.output_dir / "results.json"
    with open(out_json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logging.info(f"Saved results to {out_json_path}")

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    if not beta_results:
        logging.error("No results to plot.")
        sys.exit(1)
        
    betas = [r["beta"] for r in beta_results]
    averages = [r["average"] for r in beta_results]
    
    gsm8k_scores = [r["gsm8k"] for r in beta_results]
    humaneval_scores = [r["humaneval"] for r in beta_results]
    finqa_scores = [r["finqa_exact_match"] for r in beta_results]
    medmcqa_scores = [r["medmcqa_accuracy"] for r in beta_results]

    # Plot 1: Overall average accuracy vs. beta
    plt.figure(figsize=(8, 6))
    plt.plot(betas, averages, marker='o', linestyle='-', color='blue', label='WBP (Average)')
    
    # Baselines
    plt.axhline(y=no_cal_scores["average"], color='gray', linestyle='--', label='No-cal baseline')
    plt.axhline(y=pico_scores["average"], color='red', linestyle='--', label='Pico baseline (beta=1 equiv)')
    plt.axvline(x=1.0, color='green', linestyle=':', label='Pico-equivalent (beta=1.0)')
    
    plt.xscale('log')
    plt.xticks(BETA_VALUES, labels=[str(b) for b in BETA_VALUES])
    plt.xlabel(r'$\beta$ (Log Scale)')
    plt.ylabel('Average Accuracy across 4 benchmarks')
    plt.title('WBP $\beta$ sweep — Average benchmark accuracy (T=4, Meta-Llama-3.1-8B)')
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    
    plot1_path = args.output_dir / "beta_sweep_avg.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved average plot to {plot1_path}")

    # Plot 2: Per-benchmark accuracy vs. beta
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-benchmark accuracy vs. $\\beta$', fontsize=16)

    benchmarks = [
        (axs[0, 0], 'GSM8K (Exact Match)', gsm8k_scores, no_cal_scores["gsm8k"], pico_scores["gsm8k"]),
        (axs[0, 1], 'HumanEval (Pass@1)', humaneval_scores, no_cal_scores["humaneval"], pico_scores["humaneval"]),
        (axs[1, 0], 'MMLU Macroeconomics (Exact Match)', finqa_scores, no_cal_scores["finqa_exact_match"], pico_scores["finqa_exact_match"]),
        (axs[1, 1], 'MedQA (Accuracy)', medmcqa_scores, no_cal_scores["medmcqa_accuracy"], pico_scores["medmcqa_accuracy"])
    ]

    for ax, title, scores, no_cal, pico in benchmarks:
        ax.plot(betas, scores, marker='o', linestyle='-', color='blue', label='WBP')
        ax.axhline(y=no_cal, color='gray', linestyle='--', label='No-cal baseline')
        ax.axhline(y=pico, color='red', linestyle='--', label='Pico baseline')
        ax.axvline(x=1.0, color='green', linestyle=':', label='Pico-equivalent (beta=1.0)')
        
        ax.set_xscale('log')
        ax.set_xticks(BETA_VALUES)
        ax.set_xticklabels([str(b) for b in BETA_VALUES])
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel('Accuracy/Score')
        ax.set_title(title)
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        if title == 'GSM8K (Exact Match)': # Just show legend once to avoid clutter
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot2_path = args.output_dir / "beta_sweep_per_benchmark.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved per-benchmark plots to {plot2_path}")

    # -------------------------------------------------------------------------
    # Interpretation Guide
    # -------------------------------------------------------------------------
    peak_beta = betas[np.argmax(averages)]
    peak_val = max(averages)
    pico_val = pico_scores["average"]

    # DO NOT CLAIM from this experiment alone:
    # - That WBP-tuned "beats" Pico as a deployable result
    # - That the chosen beta generalizes to other models or domains
    # - That the data-free property is preserved (beta chosen via benchmark accuracy is NOT data-free)
    # These are explicitly acknowledged limitations in the thesis's Future Work section.
    
    logging.info("\n" + "="*60)
    logging.info("INTERPRETATION")
    logging.info("="*60)
    if peak_beta != 1.0 and peak_val > pico_val + 0.005:
        logging.info(f"FINDING: Curve peaks at beta={peak_beta} (avg={peak_val:.3f}), "
                     f"above Pico ({pico_val:.3f}). The family contains better points than beta=1. "
                     f"NOTE: A principled data-free beta selection method is still needed.")
    elif max(averages) - min(averages) < 0.01:
        logging.info("FINDING: Flat curve — Pico's beta=1 was already near-optimal in this regime.")
    elif averages[0] > averages[-1]:
        logging.info("FINDING: Monotonically decreasing with beta — over-shrinkage beyond beta=1.")
    else:
        logging.info("FINDING: Monotonically increasing with beta — under-shrinkage at beta=1.")
    logging.info("="*60)


if __name__ == "__main__":
    main()
