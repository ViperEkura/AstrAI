#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/docker/lib/train-common.sh"

COMPOSE_BASE=(
    docker compose
    --project-directory "${ROOT_DIR}"
    --file "${ROOT_DIR}/docker-compose.yml"
    --profile train
)

usage() {
    cat <<'EOF'
Usage: scripts/train.sh <command> [CONFIG] [options]

CONFIG defaults to ./train.yaml. The same file declares host runtime settings
under `runtime:` and trainer settings under model/data/parallel/training/ckpt/log.

Commands:
  init [CONFIG]                     Create runtime directories
  preflight [CONFIG]                Validate Docker, paths, GPUs, and Compose
  build [CONFIG]                    Build the trainer image
  start [CONFIG] [--foreground] [-- ARGS...]
                                    Start or resume training
  stop [CONFIG]                     Gracefully stop and checkpoint training
  restart [CONFIG]                  Stop, then start training
  logs [CONFIG]                     Follow trainer logs
  status [CONFIG]                   Show container and checkpoint status
  latest [CONFIG]                   Print the latest complete checkpoint
  list [CONFIG]                     List complete checkpoints
  clean [CONFIG] [--keep N] [--force]
                                    Preview or remove old checkpoints
EOF
}

resolve_path() {
    if [[ "$1" = /* ]]; then
        printf '%s\n' "$1"
    else
        printf '%s/%s\n' "${ROOT_DIR}" "${1#./}"
    fi
}

load_config() {
    CONFIG_FILE="$(resolve_path "$1")"
    [[ -f "${CONFIG_FILE}" ]] || die "Training config not found: ${CONFIG_FILE}"
    require_command python3
    python3 -c 'import yaml' >/dev/null 2>&1 ||
        die "PyYAML is required on the host (install python3-yaml)"

    local exports
    exports="$(python3 "${ROOT_DIR}/scripts/docker/train_runtime.py" exports "${CONFIG_FILE}")" ||
        die "Failed to load runtime configuration"
    eval "${exports}"
    validate_job_name "${TRAIN_JOB_NAME}"
    export CHECKPOINT_EXTRA_FILES="$(checkpoint_extra_files "${CONFIG_FILE}")"
}

compose() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        ASTRAI_UID="$(id -u)" ASTRAI_GID="$(id -g)" "${COMPOSE_BASE[@]}" "$@"
    else
        ASTRAI_UID="$(id -u)" ASTRAI_GID="$(id -g)" \
            env -u CUDA_VISIBLE_DEVICES "${COMPOSE_BASE[@]}" "$@"
    fi
}

checkpoint_dir() {
    printf '%s/%s\n' "${TRAIN_CHECKPOINT_DIR}" "${TRAIN_JOB_NAME}"
}

container_name() {
    printf 'astrai-trainer-%s\n' "${TRAIN_JOB_NAME}"
}

timer_pid_file() {
    printf '/tmp/astrai-timer-%s.pid\n' "${TRAIN_JOB_NAME}"
}

timer_log_file() {
    printf '/tmp/astrai-timer-%s.log\n' "${TRAIN_JOB_NAME}"
}

cancel_timer() {
    local pid_file pid
    pid_file="$(timer_pid_file)"
    [[ -f "${pid_file}" ]] || return 0
    pid="$(<"${pid_file}")"
    if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
    fi
    rm -f -- "${pid_file}"
}

schedule_timer() {
    (( TRAIN_MAX_DURATION_SECONDS > 0 )) || return 0
    cancel_timer
    local pid_file log_file
    pid_file="$(timer_pid_file)"
    log_file="$(timer_log_file)"
    (
        sleep "${TRAIN_MAX_DURATION_SECONDS}"
        "${ROOT_DIR}/scripts/train.sh" stop "${CONFIG_FILE}" --from-timer
    ) >"${log_file}" 2>&1 &
    printf '%s\n' "$!" >"${pid_file}"
    log_info "Automatic stop scheduled in ${TRAIN_MAX_DURATION_SECONDS}s"
}

init_environment() {
    mkdir -p "${TRAIN_DATA_DIR}" "${TRAIN_MODEL_DIR}" "${TRAIN_CHECKPOINT_DIR}"
    log_info "Data: ${TRAIN_DATA_DIR}"
    log_info "Model: ${TRAIN_MODEL_DIR}"
    log_info "Checkpoints: ${TRAIN_CHECKPOINT_DIR}"
}

preflight() {
    local latest visible_count
    require_command docker
    docker info >/dev/null 2>&1 || die "Docker daemon is unavailable"
    [[ -d "${TRAIN_DATA_DIR}" ]] || die "Training data directory not found: ${TRAIN_DATA_DIR}"
    mkdir -p "$(checkpoint_dir)"
    [[ -w "$(checkpoint_dir)" ]] || die "Checkpoint directory is not writable: $(checkpoint_dir)"

    latest="$(find_latest_checkpoint "$(checkpoint_dir)" || true)"
    if [[ -z "${latest}" ]]; then
        [[ -s "${TRAIN_MODEL_DIR}/config.json" ]] ||
            die "Model config not found: ${TRAIN_MODEL_DIR}/config.json"
        [[ -s "${TRAIN_MODEL_DIR}/model.safetensors" ]] ||
            die "Model weights not found: ${TRAIN_MODEL_DIR}/model.safetensors"
    else
        log_info "Resume candidate: ${latest}"
    fi

    if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
        IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
        visible_count="${#visible_gpus[@]}"
        (( visible_count == TRAIN_GPU_COUNT )) ||
            die "Configured GPU count and visible device list disagree"
    fi

    compose config --quiet
    log_info "Preflight passed for ${TRAIN_JOB_NAME} (GPU request: ${TRAIN_GPU_COUNT}, dp_mode: ${TRAIN_DP_MODE})"
}

runtime_environment_args() {
    RUNTIME_ENV_ARGS=()
    local pair
    while IFS= read -r -d '' pair; do
        RUNTIME_ENV_ARGS+=(--env "${pair}")
    done < <(python3 "${ROOT_DIR}/scripts/docker/train_runtime.py" environment "${CONFIG_FILE}")
}

start_training() {
    local foreground="$1"
    shift
    local container running
    local -a run_options
    preflight
    runtime_environment_args
    container="$(container_name)"
    running="$(docker inspect --format '{{.State.Running}}' "${container}" 2>/dev/null || true)"
    [[ "${running}" != "true" ]] || die "Trainer is already running: ${container}"
    docker rm "${container}" >/dev/null 2>&1 || true

    run_options=(
        --volume "${CONFIG_FILE}:/run/astrai/train.yaml:ro"
        --env TRAIN_CONFIG=/run/astrai/train.yaml
        "${RUNTIME_ENV_ARGS[@]}"
    )
    if [[ "${foreground}" == "true" ]]; then
        compose run --rm "${run_options[@]}" trainer "$@"
    else
        compose run -d --name "${container}" "${run_options[@]}" trainer "$@"
        schedule_timer
        log_info "Training started; run scripts/train.sh logs ${CONFIG_FILE} to follow it"
    fi
}

stop_training() {
    local from_timer="$1"
    [[ "${from_timer}" == "true" ]] || cancel_timer
    log_info "Stopping trainer with ${TRAIN_STOP_TIMEOUT}s grace period"
    docker stop --timeout "${TRAIN_STOP_TIMEOUT}" "$(container_name)" >/dev/null 2>&1 ||
        log_warn "Trainer container is not running"
    [[ "${from_timer}" != "true" ]] || rm -f -- "$(timer_pid_file)"
}

show_status() {
    local latest
    docker ps -a --filter "name=^/$(container_name)$"
    latest="$(find_latest_checkpoint "$(checkpoint_dir)" || true)"
    if [[ -n "${latest}" ]]; then
        log_info "Latest checkpoint: ${latest}"
    else
        log_info "No complete checkpoint found for ${TRAIN_JOB_NAME}"
    fi
}

clean_checkpoints() {
    local keep="$1" force="$2" dir count remove_count index path
    local -a checkpoints=()
    [[ "${keep}" =~ ^[1-9][0-9]*$ ]] || die "--keep must be a positive integer"
    dir="$(checkpoint_dir)"
    while IFS= read -r line; do
        [[ -n "${line}" ]] && checkpoints+=("${line#* * }")
    done < <(list_complete_checkpoints "${dir}")

    count="${#checkpoints[@]}"
    remove_count=$((count - keep))
    if (( remove_count <= 0 )); then
        log_info "Nothing to clean; ${count} complete checkpoint(s), keeping ${keep}"
        return
    fi
    for ((index = 0; index < remove_count; index++)); do
        path="${checkpoints[index]}"
        if [[ "${force}" == "true" ]]; then
            rm -rf -- "${path}"
            log_info "Removed ${path}"
        else
            printf 'Would remove %s\n' "${path}"
        fi
    done
    [[ "${force}" == "true" ]] || log_warn "Preview only; add --force to delete"
}

main() {
    local command="${1:-}" config="${TRAIN_CONFIG_FILE:-${ROOT_DIR}/train.yaml}"
    local foreground=false keep force=false from_timer=false
    local -a train_args=()
    [[ -n "${command}" ]] || { usage; exit 1; }
    shift || true

    if [[ "${command}" =~ ^(help|-h|--help)$ ]]; then
        usage
        return
    fi

    if [[ $# -gt 0 && "$1" != --* ]]; then
        config="$1"
        shift
    fi
    load_config "${config}"
    keep="${CHECKPOINT_KEEP_LAST}"

    case "${command}" in
        init) init_environment ;;
        preflight) preflight ;;
        build) preflight; compose build trainer ;;
        start)
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --foreground) foreground=true; shift ;;
                    --) shift; train_args=("$@"); break ;;
                    *) die "Unknown start option: $1 (put trainer arguments after --)" ;;
                esac
            done
            start_training "${foreground}" "${train_args[@]}"
            ;;
        stop)
            [[ "${1:-}" != "--from-timer" ]] || from_timer=true
            stop_training "${from_timer}"
            ;;
        restart)
            stop_training false
            start_training false
            ;;
        logs) docker logs -f --tail "${TRAIN_LOG_TAIL:-200}" "$(container_name)" ;;
        status) show_status ;;
        latest) find_latest_checkpoint "$(checkpoint_dir)" || die "No complete checkpoint found" ;;
        list)
            list_complete_checkpoints "$(checkpoint_dir)" | while read -r _epoch _step path; do
                printf '%s\n' "${path}"
            done
            ;;
        clean)
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --keep) [[ $# -ge 2 ]] || die "--keep requires a value"; keep="$2"; shift 2 ;;
                    --force) force=true; shift ;;
                    *) die "Unknown clean option: $1" ;;
                esac
            done
            clean_checkpoints "${keep}" "${force}"
            ;;
        *) die "Unknown command: ${command}" ;;
    esac
}

main "$@"
