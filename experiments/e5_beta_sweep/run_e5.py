import os
import sys
import json
import argparse
import subprocess
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

BETA_VALUES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]

# DO NOT CLAIM from this experiment alone:
# - That WBP-tuned "beats" Pico as a deployable result
# - That the chosen beta generalizes to other models or domains
# - That the data-free property is preserved (beta chosen via benchmark accuracy is NOT data-free)
# These are explicitly acknowledged limitations in the thesis's Future Work section.

def run_cmd(cmd_list, description):
    logging.info(f"{'='*60}")
    logging.info(f"RUNNING: {description}")
    logging.info(f"CMD: {' '.join(str(c) for c in cmd_list)}")
    logging.info(f"{'='*60}")
    
    result = subprocess.run(cmd_list)
    if result.returncode != 0:
        logging.error(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="E5 - Beta sweep for decoupled-lambda WBP calibration")
    parser.add_argument("--adapters_dir", type=Path, default=Path("adapters"))
    parser.add_argument("--base_model", type=str, default="unsloth/Meta-Llama-3.1-8B")
    parser.add_argument("--output_dir", type=Path, default=Path("results/e5"))
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--e2_results_json", type=Path, default=None, help="Path to E2 results JSON to overlay Pico and no-cal baselines")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    python_exec = sys.executable

    # Baselines
    no_cal_avg = None
    pico_avg = None
    e2_results = {}
    if args.e2_results_json and args.e2_results_json.exists():
        with open(args.e2_results_json, "r") as f:
            e2_results = json.load(f)
            baselines = e2_results.get("baselines", {})
            no_cal_avg = baselines.get("no_cal_average")
            pico_avg = baselines.get("pico_average")

    all_results = []
    averages = []

    for beta in BETA_VALUES:
        beta_str = str(beta).replace(".", "_")
        wbp_out = args.output_dir / f"wbp_beta_{beta_str}"
        merged_pt = args.output_dir / f"merged_beta_{beta_str}.pt"
        result_json = args.output_dir / f"eval_beta_{beta_str}.json"

        # 1. Calibrate using WBP with specific beta
        if not (wbp_out / "math" / "adapter_model.safetensors").exists() and not (wbp_out / "math" / "adapter_model.bin").exists() and not (wbp_out / "gsm8k" / "adapter_model.safetensors").exists():
            run_cmd([python_exec, "src/wbp.py",
                     "--adapters_dir", str(args.adapters_dir),
                     "--output_dir", str(wbp_out),
                     "--beta", str(beta)],
                    f"WBP Calibration with beta={beta}")

        # 2. Merge using TA
        if not merged_pt.exists():
            run_cmd([python_exec, "src/ties.py",
                     "--adapters_dir", str(wbp_out),
                     "--output_file", str(merged_pt),
                     "--method", "ta"],
                    f"Merging TA for beta={beta}")

        # 3. Evaluate
        if not result_json.exists():
            run_cmd([python_exec, "experiments/e2_accuracy/new_evaluation.py",
                     "--base_model", args.base_model,
                     "--merged_path", str(merged_pt),
                     "--output_file", str(result_json)],
                    f"Evaluation for beta={beta}")

        # 4. Parse results
        with open(result_json, "r") as f:
            scores = json.load(f)

        gsm8k = scores.get("gsm8k", {}).get("exact_match,none-0", 0.0)
        humaneval = scores.get("humaneval", {}).get("pass@1,none-0", 0.0)
        macro = scores.get("mmlu_high_school_macroeconomics_generative", {}).get("exact_match,none-0", 0.0)
        medqa = scores.get("medqa_4options", {}).get("acc,none-0", 0.0)

        avg_score = (gsm8k + humaneval + macro + medqa) / 4.0
        averages.append(avg_score)

        all_results.append({
            "beta": beta,
            "gsm8k_exact_match": gsm8k,
            "humaneval_pass_at_1": humaneval,
            "macroeconomics_exact_match": macro,
            "medqa_accuracy": medqa,
            "average": avg_score
        })

    # Output JSON structure
    final_output = {
        "experiment": "E5",
        "hardware": "RTX 6000 24GB",
        "base_model": args.base_model,
        "dtype": args.dtype,
        "T": 4,
        "seed": args.seed,
        "beta_values": BETA_VALUES,
        "baselines": {
            "no_cal_average": no_cal_avg,
            "pico_average": pico_avg
        },
        "results": all_results
    }

    final_json_path = args.output_dir / "e5_results.json"
    with open(final_json_path, "w") as f:
        json.dump(final_output, f, indent=2)

    logging.info(f"Results saved to {final_json_path}")

    # Plot 1: Overall average accuracy vs. beta
    plt.figure(figsize=(8, 6))
    plt.plot(BETA_VALUES, averages, marker='o', linewidth=2, label='WBP (beta sweep)')
    
    if no_cal_avg is not None:
        plt.axhline(y=no_cal_avg, color='red', linestyle='--', label='No-Cal Baseline')
    if pico_avg is not None:
        plt.axhline(y=pico_avg, color='green', linestyle='--', label='Pico Baseline (beta=1.0 eq)')

    plt.axvline(x=1.0, color='gray', linestyle=':', label='Pico-equivalent (beta=1.0)')
    
    plt.xscale('log', base=2)
    plt.xticks(BETA_VALUES, labels=[str(b) for b in BETA_VALUES])
    plt.xlabel(r'$\beta$ (log scale)')
    plt.ylabel('Average Accuracy across 4 Benchmarks')
    plt.title('WBP $\\beta$ sweep — Average benchmark accuracy (T=4, Meta-Llama-3.1-8B)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    avg_plot_path = args.output_dir / "beta_sweep_avg.png"
    plt.savefig(avg_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Per-benchmark accuracy vs. beta (4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    benchmarks = [
        ('gsm8k_exact_match', 'GSM8K (Exact Match)'),
        ('humaneval_pass_at_1', 'HumanEval (Pass@1)'),
        ('macroeconomics_exact_match', 'Macroeconomics (Exact Match)'),
        ('medqa_accuracy', 'MedQA (Accuracy)')
    ]

    for i, (key, title) in enumerate(benchmarks):
        y_vals = [res[key] for res in all_results]
        axes[i].plot(BETA_VALUES, y_vals, marker='o', color='C0')
        axes[i].axvline(x=1.0, color='gray', linestyle=':', label='Pico-equivalent')
        
        # We don't have per-benchmark no-cal/pico in the simple baselines dict.
        axes[i].set_xscale('log', base=2)
        axes[i].set_xticks(BETA_VALUES)
        axes[i].set_xticklabels([str(b) for b in BETA_VALUES])
        axes[i].set_title(title)
        axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend()

    plt.tight_layout()
    per_bench_plot_path = args.output_dir / "beta_sweep_per_benchmark.png"
    plt.savefig(per_bench_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Plots saved to {avg_plot_path} and {per_bench_plot_path}")

    # Interpretation
    peak_beta = BETA_VALUES[np.argmax(averages)]
    peak_val = max(averages)
    
    # Use pico_avg if available, otherwise just use beta=1.0 value
    pico_val_for_comparison = pico_avg if pico_avg is not None else all_results[BETA_VALUES.index(1.0)]["average"]

    logging.info("\n" + "="*60)
    logging.info("INTERPRETATION GUIDE")
    logging.info("="*60)
    if peak_beta != 1.0 and peak_val > pico_val_for_comparison + 0.005:
        logging.info(f"FINDING: Curve peaks at beta={peak_beta} (avg={peak_val:.3f}), "
                     f"above Pico ({pico_val_for_comparison:.3f}). The family contains better points than beta=1. "
                     f"NOTE: A principled data-free beta selection method is still needed.")
    elif max(averages) - min(averages) < 0.01:
        logging.info("FINDING: Flat curve — Pico's beta=1 was already near-optimal in this regime.")
    elif averages[0] > averages[-1]:
        logging.info("FINDING: Monotonically decreasing with beta — over-shrinkage beyond beta=1.")
    else:
        logging.info("FINDING: Monotonically increasing with beta — under-shrinkage at beta=1.")

if __name__ == "__main__":
    main()
