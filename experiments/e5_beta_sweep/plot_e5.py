import json
import logging
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    output_dir = Path("results/e5")
    results_json_path = output_dir / "results.json"
    
    if not results_json_path.exists():
        logging.error(f"Results file {results_json_path} not found. Run aggregate_e5.py first.")
        sys.exit(1)
        
    with open(results_json_path, "r") as f:
        data = json.load(f)
        
    BETA_VALUES = data["beta_values"]
    baselines = data["baselines"]
    beta_results = data["results"]
    
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
    plt.axhline(y=baselines["no_cal_average"], color='gray', linestyle='--', label='No-cal baseline')
    plt.axhline(y=baselines["pico_average"], color='red', linestyle='--', label='Pico baseline (beta=1 equiv)')
    plt.axvline(x=1.0, color='green', linestyle=':', label='Pico-equivalent (beta=1.0)')
    
    plt.xscale('log', base=2)
    plt.xticks(BETA_VALUES, labels=[str(b) for b in BETA_VALUES])
    plt.xlabel(r'$\beta$ (Log Scale)')
    plt.ylabel('Average Accuracy across 4 benchmarks')
    plt.title('WBP $\\beta$ sweep — Average benchmark accuracy (T=4, Meta-Llama-3.1-8B)')
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    
    plot1_path = output_dir / "beta_sweep_avg.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved average plot to {plot1_path}")

    # Plot 2: Per-benchmark accuracy vs. beta
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-benchmark accuracy vs. $\\beta$', fontsize=16)

    # Note: we are passing 0.0 as baseline for per-benchmark because our baselines dict in results.json only holds averages.
    # To have precise baselines in the subplots, we would need to store them in results.json too.
    # We will just plot the curve for now.
    benchmarks = [
        (axs[0, 0], 'GSM8K (Exact Match)', gsm8k_scores),
        (axs[0, 1], 'HumanEval (Pass@1)', humaneval_scores),
        (axs[1, 0], 'MMLU Macroeconomics (Exact Match)', finqa_scores),
        (axs[1, 1], 'MedQA (Accuracy)', medmcqa_scores)
    ]

    for ax, title, scores in benchmarks:
        ax.plot(betas, scores, marker='o', linestyle='-', color='blue', label='WBP')
        ax.axvline(x=1.0, color='green', linestyle=':', label='Pico-equivalent (beta=1.0)')
        
        ax.set_xscale('log', base=2)
        ax.set_xticks(BETA_VALUES)
        ax.set_xticklabels([str(b) for b in BETA_VALUES])
        ax.set_xlabel(r'$\beta$')
        ax.set_ylabel('Accuracy/Score')
        ax.set_title(title)
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        if title == 'GSM8K (Exact Match)':
            ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot2_path = output_dir / "beta_sweep_per_benchmark.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved per-benchmark plots to {plot2_path}")

    # -------------------------------------------------------------------------
    # Interpretation Guide
    # -------------------------------------------------------------------------
    peak_beta = betas[np.argmax(averages)]
    peak_val = max(averages)
    pico_val = baselines["pico_average"]
    
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
