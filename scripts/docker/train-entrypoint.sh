#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/train-common.sh"

TRAIN_JOB_NAME="${TRAIN_JOB_NAME:?TRAIN_JOB_NAME is required}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/checkpoints}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${TRAIN_JOB_NAME}"
BASE_MODEL="${BASE_MODEL:-/models/base}"
TRAIN_CONFIG="${TRAIN_CONFIG:-}"
TRAIN_GPU_COUNT="${TRAIN_GPU_COUNT:-all}"
TRAIN_DP_MODE="${TRAIN_DP_MODE:-auto}"
TRAIN_CP_SIZE="${TRAIN_CP_SIZE:-1}"
TRAIN_TP_SIZE="${TRAIN_TP_SIZE:-1}"

validate_job_name "${TRAIN_JOB_NAME}"
if [[ "${TRAIN_GPU_COUNT}" == "all" ]]; then
    TRAIN_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
fi
[[ "${TRAIN_GPU_COUNT}" =~ ^[1-9][0-9]*$ ]] || die "No visible GPU found"
[[ "${TRAIN_CP_SIZE}" =~ ^[1-9][0-9]*$ ]] ||
    die "Invalid cp_size: ${TRAIN_CP_SIZE}"
[[ "${TRAIN_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] ||
    die "Invalid tp_size: ${TRAIN_TP_SIZE}"
TRAIN_WORLD_PER_DP=$(( TRAIN_CP_SIZE * TRAIN_TP_SIZE ))
if (( TRAIN_GPU_COUNT % TRAIN_WORLD_PER_DP != 0 )); then
    die "GPU count (${TRAIN_GPU_COUNT}) must be divisible by cp_size x tp_size (${TRAIN_WORLD_PER_DP})"
fi
TRAIN_DP_SIZE=$(( TRAIN_GPU_COUNT / TRAIN_WORLD_PER_DP ))
if [[ "${TRAIN_DP_MODE}" == "auto" ]]; then
    if (( TRAIN_GPU_COUNT > 1 )); then
        TRAIN_DP_MODE=ddp
    else
        TRAIN_DP_MODE=none
    fi
fi
[[ "${TRAIN_DP_MODE}" =~ ^(none|ddp|fsdp)$ ]] || die "Invalid dp_mode: ${TRAIN_DP_MODE}"
if [[ "${TRAIN_DP_MODE}" == "none" ]] && (( TRAIN_GPU_COUNT != 1 )); then
    die "dp_mode none requires exactly one GPU"
fi
if [[ "${TRAIN_DP_MODE}" != "none" ]] && (( TRAIN_GPU_COUNT < 2 )); then
    die "dp_mode ${TRAIN_DP_MODE} requires at least two GPUs"
fi
if [[ -n "${TRAIN_CONFIG}" ]]; then
    [[ -f "${TRAIN_CONFIG}" ]] || die "Training config not found: ${TRAIN_CONFIG}"
fi
export CHECKPOINT_EXTRA_FILES="$(checkpoint_extra_files "${TRAIN_CONFIG}")"
[[ -r /data ]] || die "Training data directory is not readable: /data"

mkdir -p "${CHECKPOINT_DIR}"
[[ -w "${CHECKPOINT_DIR}" ]] || die "Checkpoint directory is not writable: ${CHECKPOINT_DIR}"

latest_checkpoint="$(find_latest_checkpoint "${CHECKPOINT_DIR}" || true)"

train_args=(
    python scripts/tools/train.py
    --ckpt_dir "${CHECKPOINT_DIR}"
    --cp_size "${TRAIN_CP_SIZE}"
    --tp_size "${TRAIN_TP_SIZE}"
    --dp_size "${TRAIN_DP_SIZE}"
)

if [[ -n "${TRAIN_CONFIG}" ]]; then
    train_args+=(--config "${TRAIN_CONFIG}")
fi

train_args+=(--dp_mode "${TRAIN_DP_MODE}")

if [[ -n "${latest_checkpoint}" ]]; then
    log_info "Resuming ${TRAIN_JOB_NAME} from ${latest_checkpoint}"
    train_args+=(--param_path "${latest_checkpoint}" --resume)
else
    [[ -s "${BASE_MODEL}/config.json" ]] || die "Base model config not found: ${BASE_MODEL}/config.json"
    [[ -s "${BASE_MODEL}/model.safetensors" ]] || die "Base model weights not found: ${BASE_MODEL}/model.safetensors"
    log_info "Starting ${TRAIN_JOB_NAME} from ${BASE_MODEL}"
    train_args+=(--param_path "${BASE_MODEL}")
fi

log_info "GPUs=${TRAIN_GPU_COUNT} (dp=${TRAIN_DP_SIZE} x cp=${TRAIN_CP_SIZE} x tp=${TRAIN_TP_SIZE}), dp_mode=${TRAIN_DP_MODE}, checkpoints=${CHECKPOINT_DIR}"

# Replace the shell so the container init forwards SIGTERM to the trainer.
exec "${train_args[@]}" "$@"
