> **ATTENTION:** You are the **E1 Sub-Agent**. Your role is to write code and execute Phase 2 (Operator-level equivalence). When you make progress, encounter roadblocks, or finish your task, you MUST append a status update to the `### E1 Status` section in `../SHARED_CONTEXT.md` so the Master Agent can track your work. Do not modify other agents' sections.

# Agent Prompt: E1 — Operator-level Equivalence (Phase 2)

> **Platform:** Mac Mini M4 (CPU / MPS — no CUDA required)
> **Your task:** Write a script that loads real trained LoRA adapters and verifies that Pico and WBP produce numerically identical calibrated B matrices on every layer.

---

## Context

Read `../SHARED_CONTEXT.md` for full mathematical background, notation, and existing code.

**E1 Objective:** Confirm that `S_wbp == S_pico` as operators — and therefore `B_tilde_wbp == B_tilde_pico` — holds on real trained LoRA weights (not just random matrices). This validates the implementation before trusting E2/E4/E5 results.

**Hardware note:** This experiment is pure linear algebra on small matrices. It runs entirely on CPU (or MPS). No CUDA GPU is needed.

---

## What to Implement

### File: `experiments/e1_equivalence/run_e1.py`

#### Inputs
- `--adapters_dir`: path to directory containing adapter subdirectories (e.g., `/Users/demid/thesis/adapters/`)
- `--base_model`: HuggingFace model name (default: `unsloth/Meta-Llama-3.1-8B`)
- `--output_dir`: where to save results JSON (default: `/Users/demid/thesis/results/e1/`)
- `--dtype`: `float32` or `float64` (default: `float64` for maximum precision)
- `--seed`: random seed (default: 42)

#### Algorithm

1. **Load LoRA adapters from disk.** Do NOT load the base model weights (too large for Mac RAM). Instead, directly read the adapter weight files (`adapter_model.safetensors` or `adapter_model.bin`) using `safetensors.torch.load_file()` or `torch.load()`.

2. **Extract B and A matrices per layer.** LoRA adapter files store weights as flat state dicts with keys like:
   ```
   base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight  # shape (d_out, r)
   base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight  # shape (r, d_in)
   ```
   Parse all keys and group by layer+module name. Build a mapping:
   ```python
   # layer_map[layer_name] = {"B": [B_1, B_2, B_3, B_4], "A": [A_1, A_2, A_3, A_4]}
   ```
   where each element of the list corresponds to one domain adapter.

3. **For each layer, run both Pico and WBP:**
   ```python
   from src.pico import merge_pico
   from src.wbp import merge_wbp
   
   B_pico, A_pico = merge_pico(B_list, A_list)
   B_wbp, A_wbp = merge_wbp(B_list, A_list, beta=1.0)
   ```

4. **Compute error metrics per layer:**
   ```python
   abs_diff = (B_pico - B_wbp).norm(p='fro')
   rel_diff = abs_diff / (B_pico.norm(p='fro') + 1e-12)
   ```
   Track: `max_rel_error`, `mean_rel_error`, `max_abs_error`, `mean_abs_error` across all layers.

5. **Write results to JSON:**
   ```json
   {
     "experiment": "E1",
     "hardware": "Mac Mini M4 (CPU)",
     "dtype": "float64",
     "base_model": "unsloth/Meta-Llama-3.1-8B",
     "T": 4,
     "num_layers_checked": <int>,
     "max_rel_error": <float>,
     "mean_rel_error": <float>,
     "max_abs_error": <float>,
     "mean_abs_error": <float>,
     "per_layer_results": [
       {"layer": "...", "max_rel_error": ..., "rel_error": ...},
       ...
     ],
     "passed": <bool>   // true if max_rel_error < 1e-4
   }
   ```

6. **Print a summary to stdout** using `logging`, not `print()`.

#### Pass/fail criterion
- `max_rel_error < 1e-5` → pass (float32 machine precision range)  
- `max_rel_error < 1e-4` → acceptable (mild conditioning issue, note it)
- `max_rel_error > 1e-4` → flag as potential issue (log a WARNING, still save results)

#### Edge case guard verification
Also include a dedicated check:
```python
# T=1 guard: single-adapter case should return unchanged
B_single, A_single = merge_pico([B_list[0]], [A_list[0]])
assert torch.allclose(B_single, B_list[0])
B_single_w, A_single_w = merge_wbp([B_list[0]], [A_list[0]])
assert torch.allclose(B_single_w, B_list[0])
logging.info("T=1 guard: OK")
```

---

## Dependencies

```python
# Standard dependencies (should already be in .venv)
import torch, json, logging, argparse, os
from pathlib import Path
# May need: pip install safetensors
from safetensors.torch import load_file
```

The `src/` package is at `/Users/demid/thesis/src/`. Run with `PYTHONPATH=/Users/demid/thesis`.

---

## Deliverables

- `experiments/e1_equivalence/run_e1.py`
- No README needed — see `experiments/e1_equivalence/README.md` (already provided)
