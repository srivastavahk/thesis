#!/usr/bin/env bash
# =============================================================================
# run_all_domains_unsloth.sh — Train all 4 LoRA adapters sequentially with Unsloth
# =============================================================================
#
# Usage (from project root ~/thesis/):
#   chmod +x experiments/adapter_training/run_all_domains_unsloth.sh
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx          # WRITE-scope token required
#   bash experiments/adapter_training/run_all_domains_unsloth.sh
#
# Safe to re-run after a cancel:
#   - Domains with a completed adapter (adapters/{domain}/adapter_model.safetensors)
#     are SKIPPED automatically.
#   - Domains with only partial checkpoints (checkpoints/{domain}/checkpoint-N/)
#     are RESUMED from the latest checkpoint automatically.
#
# Recommended: run inside tmux so SSH disconnects don't kill the job.
#   tmux new -s training
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
#   bash experiments/adapter_training/run_all_domains_unsloth.sh
#   # Detach: Ctrl+B, D   |   Reattach: tmux attach -t training
#
# Logs:
#   logs/train_unsloth_math.log
#   logs/train_unsloth_coding.log
#   logs/train_unsloth_finance.log
#   logs/train_unsloth_medical.log
#   logs/train_all_unsloth_summary.log     ← timing table written at the end
#
# Adapters saved to:
#   adapters/{domain}/             ← local copy always saved first
#   HuggingFace Hub:  mml2024003/Meta-Llama-3.1-8B_{domain}
# =============================================================================

set -euo pipefail   # exit on error, unset variables, pipe failures

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_adapter_unsloth.py"
LOG_DIR="${PROJECT_ROOT}/logs"
SUMMARY_LOG="${LOG_DIR}/train_all_unsloth_summary.log"
DOMAINS=(math coding finance medical)

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  WBP Thesis — Adapter Training (Unsloth)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# 1. Verify we are NOT on Mac Mini (no CUDA → bail immediately)
python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print('ERROR: CUDA not available. This script must run on the Lab RTX 6000.')
    print('       Do NOT run on Mac Mini.')
    sys.exit(1)
print(f'GPU OK: {torch.cuda.get_device_name(0)}  '
      f'({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)')
"

# 2. Verify HF_TOKEN is set (needed for gated Llama + Hub push)
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo ""
    echo "ERROR: HF_TOKEN is not set."
    echo "       Export a HuggingFace token with WRITE scope:"
    echo "         export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx"
    echo "       Then re-run this script."
    exit 1
fi
echo "HF_TOKEN: set (length=${#HF_TOKEN})"

# 3. Verify the training script exists
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: Training script not found at ${TRAIN_SCRIPT}"
    exit 1
fi

# 4. Create output directories
mkdir -p "${LOG_DIR}"
mkdir -p "${PROJECT_ROOT}/adapters"
mkdir -p "${PROJECT_ROOT}/checkpoints"

echo ""
echo "Project root : ${PROJECT_ROOT}"
echo "Log dir      : ${LOG_DIR}"
echo "Domains      : ${DOMAINS[*]}"
echo ""

# ---------------------------------------------------------------------------
# Helper: run one domain and track elapsed time
# ---------------------------------------------------------------------------
declare -A DOMAIN_STATUS
declare -A DOMAIN_ELAPSED

run_domain() {
    local domain="$1"
    local log_file="${LOG_DIR}/train_unsloth_${domain}.log"
    local adapter_file="${PROJECT_ROOT}/adapters/${domain}/adapter_model.safetensors"
    local start_ts
    start_ts=$(date +%s)

    # ------------------------------------------------------------------
    # Skip if adapter already exists (fully completed domain)
    # ------------------------------------------------------------------
    if [[ -f "${adapter_file}" ]]; then
        echo "------------------------------------------------------------"
        echo "  SKIPPING domain: ${domain}"
        echo "  Adapter already exists: ${adapter_file}"
        echo "  Delete it to force a re-run."
        echo "------------------------------------------------------------"
        DOMAIN_STATUS["${domain}"]="SKIPPED"
        DOMAIN_ELAPSED["${domain}"]="0m 0s"
        return 0
    fi

    # ------------------------------------------------------------------
    # Check for partial checkpoint (resume case)
    # ------------------------------------------------------------------
    local checkpoint_dir="${PROJECT_ROOT}/checkpoints/${domain}"
    local latest_ckpt
    latest_ckpt=$(ls -d "${checkpoint_dir}/checkpoint-"* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)

    echo "------------------------------------------------------------"
    if [[ -n "${latest_ckpt}" ]]; then
        echo "  RESUMING domain: ${domain}"
        echo "  From checkpoint: ${latest_ckpt}"
    else
        echo "  Starting domain: ${domain} (from step 0)"
    fi
    echo "  Log: ${log_file}  (appending)"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"

    # Append to existing log (preserves history from cancelled run)
    echo "" >> "${log_file}"
    echo "=== Run started at $(date '+%Y-%m-%d %H:%M:%S') ==" >> "${log_file}"

    # Run training; tee appends to log file
    if PYTHONPATH="${PROJECT_ROOT}" python3 "${TRAIN_SCRIPT}" \
            --domain "${domain}" \
            2>&1 | tee -a "${log_file}"; then
        DOMAIN_STATUS["${domain}"]="SUCCESS"
    else
        DOMAIN_STATUS["${domain}"]="FAILED"
        echo ""
        echo "WARNING: Training for '${domain}' exited with a non-zero status."
        echo "         Check ${log_file} for details."
        echo "         Continuing with next domain..."
    fi

    local end_ts
    end_ts=$(date +%s)
    local elapsed=$(( end_ts - start_ts ))
    local elapsed_min=$(( elapsed / 60 ))
    local elapsed_sec=$(( elapsed % 60 ))
    DOMAIN_ELAPSED["${domain}"]="${elapsed_min}m ${elapsed_sec}s"

    echo ""
    echo "  ${domain} finished — status: ${DOMAIN_STATUS[${domain}]}  elapsed: ${DOMAIN_ELAPSED[${domain}]}"
    echo ""
}

# ---------------------------------------------------------------------------
# Run all domains sequentially
# ---------------------------------------------------------------------------
OVERALL_START=$(date +%s)

for domain in "${DOMAINS[@]}"; do
    run_domain "${domain}"
done

OVERALL_END=$(date +%s)
OVERALL_ELAPSED=$(( (OVERALL_END - OVERALL_START) / 60 ))

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  TRAINING COMPLETE"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Total elapsed: ${OVERALL_ELAPSED} min"
echo "============================================================"
echo ""
echo "Domain     Status    Elapsed"
echo "---------- --------- -------"
for domain in "${DOMAINS[@]}"; do
    printf "%-10s %-9s %s\n" \
        "${domain}" \
        "${DOMAIN_STATUS[${domain}]:-SKIPPED}" \
        "${DOMAIN_ELAPSED[${domain}]:-n/a}"
done
echo ""

# Write summary to log file
{
    echo "============================================================"
    echo "  Adapter Training Summary (Unsloth)"
    echo "  Completed: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Total elapsed: ${OVERALL_ELAPSED} min"
    echo "============================================================"
    echo ""
    echo "Domain     Status    Elapsed"
    echo "---------- --------- -------"
    for domain in "${DOMAINS[@]}"; do
        printf "%-10s %-9s %s\n" \
            "${domain}" \
            "${DOMAIN_STATUS[${domain}]:-SKIPPED}" \
            "${DOMAIN_ELAPSED[${domain}]:-n/a}"
    done
    echo ""
    echo "Adapter locations:"
    for domain in "${DOMAINS[@]}"; do
        echo "  ${domain}: ${PROJECT_ROOT}/adapters/${domain}/"
        echo "           https://huggingface.co/mml2024003/Meta-Llama-3.1-8B_${domain}"
    done
} | tee "${SUMMARY_LOG}"

echo "Summary written to: ${SUMMARY_LOG}"
echo ""

# Exit non-zero if any domain failed
for domain in "${DOMAINS[@]}"; do
    if [[ "${DOMAIN_STATUS[${domain}]:-}" == "FAILED" ]]; then
        echo "One or more domains failed. Review logs in ${LOG_DIR}/"
        exit 1
    fi
done

echo "All domains trained successfully."
