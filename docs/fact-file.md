# Project Fact File

> **Purpose:** This file is the single, immutable source of truth for all quantitative constraints, hardware limits, and domain facts for the thesis project. Any agent or collaborator working on this project MUST adhere to these facts and never invent or hallucinate alternatives.

## 1. Core Identity
- **Thesis Title:** Woodbury B-Space Preconditioning: Fast and Generalized B-Space Calibration for Model Merging
- **Phases:** 
  1. Verification (E0)
  2. Novel Algorithm (E1)
  3. Benchmarking & Equivalence (E2, E3, E4)
  4. Generalization (E5)

## 2. Hardware Constraints
- **Primary Lab PC:** NVIDIA Quadro RTX 6000 (24 GB VRAM limit). Used for full model inference and operator-level tests (E1) since adapters are stored here.
  - *Implication:* Inference batch size must remain $\leq 4$ to prevent OOM errors alongside the 16 GB base model.
- **Secondary Machine:** Mac Mini M4 (CPU / MPS). Used for script verification and theoretical proofs, not for full model inference.

## 3. Model & Data Truths
- **Base Model:** `unsloth/Meta-Llama-3.1-8B` (Base, NOT instruct variant).
- **LoRA Parameters:**
  - **Rank:** $r=16$ (primary), $r=8$ (secondary).
  - **Target Modules:** `q_proj` and `v_proj` inside attention layers only.
- **Domains and Benchmarks (E2):**
  - **Math:** GSM8K
  - **Coding:** HumanEval
  - **Finance:** MMLU High School Macroeconomics Generative (`mmlu_high_school_macroeconomics_generative`)
  - **Medical:** MedQA (`medqa_4options`)

## 4. Algorithmic Truths
- **Complexity:** Both Pico and WBP scale as $O(d_{out} \cdot (Tr)^2)$. Neither is asymptotically "better" in Big-O notation, but WBP avoids iterative SVD and utilizes GPU GEMMs much faster in practice.
- **TIES Merger:** The default density threshold is strictly `0.2`.

### 4.1 Pseudocode References
- **Pico Merge Algorithm:** See [docs/algorithm-pico.md](file:///Users/demid/thesis/docs/algorithm-pico.md) for the exact implementation steps and SVD logic.
- **WBP Merge Algorithm:** See [docs/algorithm-wbp.md](file:///Users/demid/thesis/docs/algorithm-wbp.md) for the exact implementation steps and Woodbury identity logic.

## 5. Agent Interaction Rules
- **Explicit Approval for File Modifications:** The Master Agent and Sub-Agents MUST NEVER modify project files (code, documentation, configuration) immediately upon receiving a request. They must first present an implementation plan detailing exactly what changes they intend to make, explicitly pause, and wait for the user's approval.
- **Fact File Maintenance:** If a Sub-Agent modifies any experiment parameter (e.g., changing a dataset, altering an evaluation metric, adjusting $T$ or $r$, or modifying $d_{out}$ limits), it MUST immediately update this `fact-file.md` document to reflect the new truth. 
- **Multi-Agent Protocol:** Sub-agents execute specific experiments and write their findings/errors to their designated sections in `experiments/SHARED_CONTEXT.md`. The Master Agent synthesizes these findings into the thesis but does not write code. The Master Agent will rely EXCLUSIVELY on this `fact-file.md` for project facts, and any clashes with other files are resolved in favor of this file.
