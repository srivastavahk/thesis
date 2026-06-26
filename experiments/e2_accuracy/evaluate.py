"""
evaluate.py — Benchmark Evaluation Functions for E2 and E5
===========================================================
Platform : Lab RTX 6000 (CUDA)
Purpose  : One evaluation function per benchmark, backed by the EleutherAI
           lm_eval harness for reproducible, citation-grade results.

Benchmarks
----------
  eval_gsm8k      → GSM8K 5-shot exact-match on numeric answers
  eval_humaneval  → HumanEval pass@1 (uses lm_eval with HF_ALLOW_CODE_EVAL=1)
  eval_finqa      → FinQA 0-shot exact-match
  eval_medmcqa    → MedMCQA 0-shot multiple-choice accuracy (normalized)

All functions accept a pre-loaded HuggingFace model + tokenizer, wrap them
in an lm_eval.models.huggingface.HFLM, and call lm_eval.simple_evaluate.
This ensures prompts, scoring, and normalisation match published standards.

IMPORTANT: HumanEval code execution requires the env variable:
    HF_ALLOW_CODE_EVAL=1
This is set automatically inside eval_humaneval(). Code runs inside lm_eval's
own sandboxed subprocess pool (not exec() in-process).
"""

import logging
import os
from typing import Optional

import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared lm_eval wrapper
# ---------------------------------------------------------------------------

def _run_lm_eval(
    model,
    tokenizer,
    tasks: list,
    n_samples: Optional[int],
    batch_size: str | int,
    seed: int,
    extra_kwargs: Optional[dict] = None,
) -> dict:
    """
    Wrap a HuggingFace model in an HFLM and call lm_eval.simple_evaluate.

    Parameters
    ----------
    model        : A HuggingFace CausalLM already loaded and on the correct device.
    tokenizer    : Matching tokenizer.
    tasks        : List of lm_eval task names.
    n_samples    : Limit evaluation to this many examples (None = full dataset).
    batch_size   : Passed to HFLM. Use "auto" to let lm_eval tune for VRAM.
    seed         : Random seed for shuffling.
    extra_kwargs : Additional kwargs forwarded to simple_evaluate (e.g. num_fewshot).

    Returns
    -------
    dict of {metric_name: float} extracted from lm_eval results.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    kwargs = dict(
        model=eval_model,
        tasks=tasks,
        limit=n_samples,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        log_samples=False,
    )
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    results = lm_eval.simple_evaluate(**kwargs)
    return results["results"]


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def eval_gsm8k(
    model,
    tokenizer,
    device: str,
    n_samples: Optional[int] = 150,
    batch_size: str | int = "auto",
    seed: int = 42,
    **_,
) -> float:
    """
    Evaluate on GSM8K (5-shot, exact-match on numeric answer).
    Returns accuracy ∈ [0, 1].
    """
    log.info("Running GSM8K (n=%s) ...", n_samples or "full")
    raw = _run_lm_eval(
        model, tokenizer,
        tasks=["gsm8k"],
        n_samples=n_samples,
        batch_size=batch_size,
        seed=seed,
        extra_kwargs={"num_fewshot": 5},
    )
    # lm_eval reports "exact_match,strict-match" or "exact_match,flexible-extract"
    task_results = raw.get("gsm8k", {})
    score = (
        task_results.get("exact_match,strict-match")
        or task_results.get("exact_match,flexible-extract")
        or task_results.get("acc,none")
        or 0.0
    )
    log.info("GSM8K exact-match: %.4f", score)
    return float(score)


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------

def eval_humaneval(
    model,
    tokenizer,
    device: str,
    n_samples: Optional[int] = 150,
    batch_size: str | int = "auto",
    seed: int = 42,
    **_,
) -> float:
    """
    Evaluate on HumanEval (pass@1, greedy decoding via lm_eval).
    Sets HF_ALLOW_CODE_EVAL=1 so lm_eval can run the generated code.
    Returns pass@1 ∈ [0, 1].
    """
    log.info("Running HumanEval (n=%s) ...", n_samples or "full")

    # lm_eval requires this env var to execute code (uses its own sandbox)
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    raw = _run_lm_eval(
        model, tokenizer,
        tasks=["humaneval"],
        n_samples=n_samples,
        batch_size=batch_size,
        seed=seed,
        extra_kwargs={"confirm_run_unsafe_code": True},
    )
    task_results = raw.get("humaneval", {})
    score = (
        task_results.get("pass@1,none")
        or task_results.get("pass@1")
        or 0.0
    )
    log.info("HumanEval pass@1: %.4f", score)
    return float(score)


# ---------------------------------------------------------------------------
# Finance — MMLU High School Macroeconomics
# (finqa is not available in lm_eval; mmlu_high_school_macroeconomics is the
# closest available finance-domain task and is directly comparable across papers)
# ---------------------------------------------------------------------------

FINANCE_TASK = "mmlu_high_school_macroeconomics"


def eval_finance(
    model,
    tokenizer,
    device: str,
    n_samples: Optional[int] = 150,
    batch_size: str | int = "auto",
    seed: int = 42,
    **_,
) -> float:
    """
    Evaluate on MMLU High School Macroeconomics (0-shot, normalized accuracy).
    Substitutes FinQA (not available in lm_eval harness).
    Returns acc_norm ∈ [0, 1].
    """
    log.info("Running %s (n=%s) ...", FINANCE_TASK, n_samples or "full")
    raw = _run_lm_eval(
        model, tokenizer,
        tasks=[FINANCE_TASK],
        n_samples=n_samples,
        batch_size=batch_size,
        seed=seed,
        extra_kwargs={"num_fewshot": 0},
    )
    task_results = raw.get(FINANCE_TASK, {})
    score = (
        task_results.get("acc_norm,none")
        or task_results.get("acc,none")
        or 0.0
    )
    log.info("%s acc_norm: %.4f", FINANCE_TASK, score)
    return float(score)


# ---------------------------------------------------------------------------
# MedMCQA
# ---------------------------------------------------------------------------

def eval_medmcqa(
    model,
    tokenizer,
    device: str,
    n_samples: Optional[int] = 150,
    batch_size: str | int = "auto",
    seed: int = 42,
    **_,
) -> float:
    """
    Evaluate on MedMCQA (0-shot multiple-choice accuracy, normalized).
    Returns accuracy ∈ [0, 1].
    """
    log.info("Running MedMCQA (n=%s) ...", n_samples or "full")
    raw = _run_lm_eval(
        model, tokenizer,
        tasks=["medmcqa"],
        n_samples=n_samples,
        batch_size=batch_size,
        seed=seed,
        extra_kwargs={"num_fewshot": 0},
    )
    task_results = raw.get("medmcqa", {})
    score = (
        task_results.get("acc_norm,none")
        or task_results.get("acc,none")
        or 0.0
    )
    log.info("MedMCQA accuracy: %.4f", score)
    return float(score)
