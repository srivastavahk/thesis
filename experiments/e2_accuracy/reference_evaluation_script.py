import os
import gc
import torch
import json
from unsloth import FastLanguageModel
import lm_eval
from lm_eval.models.huggingface import HFLM

# 1. Environment and Cleanup
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# 2. Load Base Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Load the adapter
model = FastLanguageModel.for_inference(model)
model.load_adapter("mml2024003/math_adapter_mini")


eval_model = HFLM(pretrained=model, tokenizer=tokenizer)
tasks = ["mmlu_elementary_mathematics", "humaneval", "mmlu_anatomy", "mmlu_high_school_macroeconomics"]

print(f"Starting benchmark for tasks: {tasks} ...")

results = lm_eval.simple_evaluate(
    model=eval_model,
    tasks=tasks,
    #num_fewshot=0,
    batch_size="auto",
    device="cuda",
    confirm_run_unsafe_code=True
)

# 4. Save and Display
with open("mathadapter_benchmark.json", "w") as f:
    json.dump(results["results"], f, indent=2)

print("\n--- Benchmark Results ---")
for task, metrics in results['results'].items():
    print(f"\nTask: {task}")
    print(json.dumps(metrics, indent=2))