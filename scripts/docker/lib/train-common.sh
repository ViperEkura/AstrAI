#!/usr/bin/env bash

log_info() {
    printf '[INFO] %s\n' "$*"
}

log_warn() {
    printf '[WARN] %s\n' "$*" >&2
}

die() {
    printf '[ERROR] %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

validate_job_name() {
    [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
        die "Invalid TRAIN_JOB_NAME '$1'; use letters, numbers, dot, underscore, or dash"
}

checkpoint_extra_files() {
    # Additional files a complete checkpoint must contain for the strategy
    # configured in the given training YAML. PPO persists critic state as
    # checkpoint extras (value_model.pt / value_optimizer.pt); a resume
    # without them must not look complete.
    local config="$1"

    [[ -n "${config}" && -f "${config}" ]] || return 0
    if grep -Eq '^[[:space:]]*train_type:[[:space:]]*["'\'']?online_ppo' "${config}"; then
        printf 'value_model.pt value_optimizer.pt'
    fi
}

checkpoint_is_complete() {
    local checkpoint="$1"
    local file

    [[ -d "${checkpoint}" ]] || return 1

    for file in meta.json config.json model.safetensors optimizer.pt scheduler.pt ${CHECKPOINT_EXTRA_FILES:-}; do
        [[ -s "${checkpoint}/${file}" ]] || return 1
    done

    return 0
}

checkpoint_coordinates() {
    local name

    name="$(basename "$1")"
    [[ "${name}" =~ ^epoch_([0-9]+)_step_([0-9]+)$ ]] || return 1
    printf '%d %d\n' "$((10#${BASH_REMATCH[1]}))" "$((10#${BASH_REMATCH[2]}))"
}

list_complete_checkpoints() {
    local checkpoint_dir="$1"
    local checkpoint coordinates epoch step

    for checkpoint in "${checkpoint_dir}"/epoch_*_step_*; do
        [[ -d "${checkpoint}" ]] || continue
        coordinates="$(checkpoint_coordinates "${checkpoint}")" || continue
        checkpoint_is_complete "${checkpoint}" || continue
        read -r epoch step <<<"${coordinates}"
        printf '%012d %012d %s\n' "${epoch}" "${step}" "${checkpoint}"
    done | sort -n -k1,1 -k2,2
}

find_latest_checkpoint() {
    local checkpoint_dir="$1"
    local latest

    latest="$(list_complete_checkpoints "${checkpoint_dir}" | tail -n 1)"
    [[ -n "${latest}" ]] || return 1
    printf '%s\n' "${latest#* * }"
}
