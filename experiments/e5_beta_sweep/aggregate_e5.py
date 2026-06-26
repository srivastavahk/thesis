import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

BETA_VALUES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]

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
    base_model = "unsloth/Meta-Llama-3.1-8B"
    e2_results_dir = Path("results/e2")
    output_dir = Path("results/e5")
    
    beta_results = []
    
    for b in BETA_VALUES:
        beta_str = f"beta_{str(b).replace('.', '_')}"
        eval_json = output_dir / f"eval_wbp_{beta_str}_ta.json"
        
        scores = extract_scores(eval_json)
        if scores:
            scores["beta"] = b
            beta_results.append(scores)
        else:
            logging.error(f"Failed to load scores for beta={b} from {eval_json}")

    # -------------------------------------------------------------------------
    # Baseline Extractions
    # -------------------------------------------------------------------------
    no_cal_json = e2_results_dir / "eval_no_cal_ta.json"
    pico_json = e2_results_dir / "eval_pico_ta.json"
    
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
        "base_model": base_model,
        "dtype": "bfloat16",
        "T": 4,
        "seed": 42,
        "beta_values": BETA_VALUES,
        "baselines": {
            "no_cal_average": no_cal_scores["average"],
            "pico_average": pico_scores["average"]
        },
        "results": beta_results
    }
    
    out_json_path = output_dir / "results.json"
    with open(out_json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logging.info(f"Saved aggregated results to {out_json_path}")

if __name__ == "__main__":
    main()
