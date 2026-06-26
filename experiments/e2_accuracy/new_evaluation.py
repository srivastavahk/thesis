import os
import gc
import json
import argparse
from pathlib import Path

import torch
from unsloth import FastLanguageModel
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

def apply_dense_update(model, state_dict_path: Path):
    """Loads a dense \Delta W state_dict and adds it to the base model weights."""
    print(f"Loading dense merged updates from {state_dict_path}...")
    update_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    
    model_state = model.state_dict()
    matched = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in update_dict:
                delta = update_dict[name].to(param.device).to(param.dtype)
                param.add_(delta)
                matched += 1
                
    print(f"Patched {matched} parameters.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a base model patched with a dense \Delta W update.")
    parser.add_argument("--base_model", type=str, default="unsloth/Meta-Llama-3.1-8B", help="Base model path or HF name.")
    parser.add_argument("--merged_path", type=Path, required=True, help="Path to the dense merged .pt file.")
    parser.add_argument("--output_file", type=Path, required=True, help="Path to save evaluation results JSON.")
    args = parser.parse_args()

    # 1. Environment and Cleanup
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 2. Load Base Model
    print(f"Loading base model {args.base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.base_model,
        max_seq_length = 2048,
        dtype = torch.bfloat16,
        load_in_4bit = False, # We load in bf16 since we'll patch it directly
    )

    model = FastLanguageModel.for_inference(model)

    # 3. Patch the dense merged update
    apply_dense_update(model, args.merged_path)

    # 4. Evaluate
    eval_model = HFLM(pretrained=model, tokenizer=tokenizer)
    
    # Thesis specific tasks: gsm8k, humaneval, mmlu_high_school_macroeconomics_generative, medmcqa
    # Note: These exact task names must be available in lm_eval
    tasks = ["gsm8k", "humaneval", "mmlu_high_school_macroeconomics_generative", "medmcqa"]

    print(f"Starting benchmark for tasks: {tasks} ...")
    
    task_manager = TaskManager(include_path="experiments/e2_accuracy/custom_tasks")
    
    results = lm_eval.simple_evaluate(
        model=eval_model,
        tasks=tasks,
        task_manager=task_manager,
        batch_size="auto",
        device="cuda",
        confirm_run_unsafe_code=True
    )

    # 5. Save and Display
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results["results"], f, indent=2)

    print("\n--- Benchmark Results ---")
    for task, metrics in results['results'].items():
        print(f"\nTask: {task}")
        print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
