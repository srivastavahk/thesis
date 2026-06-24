"""
evaluate.py — Benchmark Evaluation Functions for E2 and E5
===========================================================
Platform : Lab RTX 6000 (CUDA)
Purpose  : One evaluation function per benchmark. Each takes
           (model, tokenizer, device) and returns a float score ∈ [0, 1].

Benchmarks
----------
  eval_gsm8k      → GSM8K exact-match on numeric answers (~1319 examples)
  eval_humaneval  → HumanEval pass@1 with subprocess sandboxing (164 problems)
  eval_finqa      → FinQA exact-match on answer strings (sample 500)
  eval_medmcqa    → MedMCQA multiple-choice accuracy (sample 1000)

All functions use greedy decoding and log.info() for progress.
"""

import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared inference helper
# ---------------------------------------------------------------------------

def _batch_greedy(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    batch_size: int = 4,
    device: str = "cuda",
) -> List[str]:
    """
    Run greedy decoding on a list of prompts in batches.
    Handles left-padding for causal LMs.
    Returns a list of decoded generated strings (prompt stripped off).
    """
    # Causal LM needs left-padding so all sequences are right-aligned
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (strip off the prompt tokens)
        prompt_lengths = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, prompt_lengths:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend(decoded)

    tokenizer.padding_side = original_padding_side
    return outputs


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def _extract_gsm8k_number(text: str) -> Optional[float]:
    """Extract the last number from a GSM8K-style answer string."""
    # Remove commas, dollar signs, percent, then find all numbers
    cleaned = text.replace(",", "").replace("$", "").replace("%", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def eval_gsm8k(
    model,
    tokenizer,
    device: str,
    n_samples: Optional[int] = None,
    batch_size: int = 4,
) -> float:
    """
    Evaluate on GSM8K (test split, ~1319 examples).
    Returns exact-match fraction on numeric answers.
    """
    from datasets import load_dataset

    log.info("Loading GSM8K test set ...")
    dataset = load_dataset("gsm8k", "main", split="test")
    if n_samples is not None:
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    prompts = [f"Question: {ex['question']}\nAnswer:" for ex in dataset]
    gold_answers = [_extract_gsm8k_number(ex["answer"]) for ex in dataset]

    log.info("Running GSM8K inference on %d examples ...", len(prompts))
    predictions = _batch_greedy(
        model, tokenizer, prompts, max_new_tokens=256, batch_size=batch_size, device=device
    )

    correct = 0
    for pred_text, gold in zip(predictions, gold_answers):
        pred = _extract_gsm8k_number(pred_text)
        if pred is not None and gold is not None and math.isclose(pred, gold, rel_tol=1e-4):
            correct += 1

    score = correct / len(dataset)
    log.info("GSM8K exact-match: %.4f  (%d/%d)", score, correct, len(dataset))
    return score


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------

_HUMANEVAL_TIMEOUT = 10  # seconds per problem


def _run_humaneval_problem(completion: str, test_code: str, entry_point: str) -> bool:
    """
    Run a single HumanEval problem in a subprocess sandbox.
    Returns True if all test cases pass.
    """
    # Build the full program: completion + test harness
    full_code = textwrap.dedent(f"""
{completion}

{test_code}

check({entry_point})
""")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            timeout=_HUMANEVAL_TIMEOUT,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def eval_humaneval(
    model,
    tokenizer,
    device: str,
    batch_size: int = 4,
) -> float:
    """
    Evaluate on HumanEval (164 problems), pass@1 with greedy decoding.
    Code is executed in a sandboxed subprocess — NEVER with exec() in-process.
    Returns fraction of problems where all test cases pass.
    """
    from datasets import load_dataset

    log.info("Loading HumanEval test set ...")
    dataset = load_dataset("openai_humaneval", split="test")

    prompts = [ex["prompt"] for ex in dataset]
    log.info("Running HumanEval inference on %d problems ...", len(prompts))

    completions = _batch_greedy(
        model, tokenizer, prompts, max_new_tokens=512, batch_size=batch_size, device=device
    )

    passed = 0
    for i, (ex, completion) in enumerate(zip(dataset, completions)):
        # Full completion = the original prompt + generated continuation
        full_completion = ex["prompt"] + completion
        ok = _run_humaneval_problem(
            full_completion, ex["test"], ex["entry_point"]
        )
        if ok:
            passed += 1
        if (i + 1) % 20 == 0:
            log.info("  HumanEval progress: %d/%d (pass so far: %d)", i + 1, len(dataset), passed)

    score = passed / len(dataset)
    log.info("HumanEval pass@1: %.4f  (%d/%d)", score, passed, len(dataset))
    return score


# ---------------------------------------------------------------------------
# FinQA
# ---------------------------------------------------------------------------

def eval_finqa(
    model,
    tokenizer,
    device: str,
    n_samples: int = 500,
    seed: int = 42,
    batch_size: int = 4,
) -> float:
    """
    Evaluate on FinQA (test split), exact match on answer string.
    Returns fraction correct.
    """
    from datasets import load_dataset

    log.info("Loading FinQA test set (sample=%d) ...", n_samples)
    dataset = load_dataset("dreamerdeo/finqa", split="test")
    dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))

    prompts = []
    gold_answers = []
    for ex in dataset:
        # Build context from the table and pre/post-text if available
        context = ex.get("pre_text", "") + " " + ex.get("post_text", "")
        question = ex.get("question", "")
        answer = str(ex.get("answer", "")).strip().lower()
        prompts.append(f"Context: {context.strip()}\nQuestion: {question}\nAnswer:")
        gold_answers.append(answer)

    log.info("Running FinQA inference on %d examples ...", len(prompts))
    predictions = _batch_greedy(
        model, tokenizer, prompts, max_new_tokens=256, batch_size=batch_size, device=device
    )

    correct = 0
    for pred_text, gold in zip(predictions, gold_answers):
        # Extract first line / first sentence of prediction; normalize
        pred_first = pred_text.strip().split("\n")[0].strip().lower()
        if pred_first == gold or gold in pred_first:
            correct += 1

    score = correct / len(dataset)
    log.info("FinQA exact-match: %.4f  (%d/%d)", score, correct, len(dataset))
    return score


# ---------------------------------------------------------------------------
# MedMCQA
# ---------------------------------------------------------------------------

_MEDMCQA_CHOICE_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def eval_medmcqa(
    model,
    tokenizer,
    device: str,
    n_samples: int = 1000,
    seed: int = 42,
    batch_size: int = 4,
) -> float:
    """
    Evaluate on MedMCQA (validation split), multiple-choice accuracy.
    Returns fraction correct.
    """
    from datasets import load_dataset

    log.info("Loading MedMCQA validation set (sample=%d) ...", n_samples)
    dataset = load_dataset("openlifescienceai/medmcqa", split="validation")
    dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))

    prompts = []
    gold_letters = []
    for ex in dataset:
        q = ex.get("question", "")
        opa = ex.get("opa", "")
        opb = ex.get("opb", "")
        opc = ex.get("opc", "")
        opd = ex.get("opd", "")
        cop = ex.get("cop", 0)  # correct option index 0–3
        prompt = (
            f"Question: {q}\n"
            f"A. {opa}\nB. {opb}\nC. {opc}\nD. {opd}\n"
            f"Answer:"
        )
        prompts.append(prompt)
        gold_letters.append(_MEDMCQA_CHOICE_MAP.get(cop, "A"))

    log.info("Running MedMCQA inference on %d examples ...", len(prompts))
    predictions = _batch_greedy(
        model, tokenizer, prompts, max_new_tokens=8, batch_size=batch_size, device=device
    )

    correct = 0
    for pred_text, gold in zip(predictions, gold_letters):
        # Expect the model to output "A", "B", "C", or "D"
        pred_letter = pred_text.strip()[:1].upper()
        if pred_letter == gold:
            correct += 1

    score = correct / len(dataset)
    log.info("MedMCQA accuracy: %.4f  (%d/%d)", score, correct, len(dataset))
    return score
