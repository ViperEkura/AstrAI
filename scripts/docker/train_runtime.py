"""Parse the host-side runtime section of a training configuration."""

import argparse
import math
import re
import shlex
from pathlib import Path

import yaml

ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DP_MODES = {"auto", "none", "ddp", "fsdp"}


def _mapping(value, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"runtime.{name} must be a mapping")
    return value


def _path(value, name: str, config_dir: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"runtime.paths.{name} is required")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def load_runtime(config_path: str) -> dict[str, str]:
    path = Path(config_path).resolve()
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError("training configuration must be a mapping")

    runtime = _mapping(config.get("runtime"), "runtime")
    if not runtime:
        raise ValueError("top-level runtime section is required")
    paths = _mapping(runtime.get("paths"), "paths")
    gpu = _mapping(runtime.get("gpu"), "gpu")
    container = _mapping(runtime.get("container"), "container")
    environment = _mapping(runtime.get("environment"), "environment")

    job_name = runtime.get("job_name")
    if not isinstance(job_name, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", job_name
    ):
        raise ValueError(
            "runtime.job_name must use letters, numbers, dot, underscore, or dash"
        )

    devices = gpu.get("devices", "all")
    visible_devices = None
    if devices == "all":
        gpu_count = "all"
    elif isinstance(devices, list) and devices:
        normalized = []
        for device in devices:
            text = str(device)
            if not text.isdigit():
                raise ValueError(
                    "runtime.gpu.devices entries must be non-negative integers"
                )
            normalized.append(text)
        if len(set(normalized)) != len(normalized):
            raise ValueError("runtime.gpu.devices must not contain duplicates")
        gpu_count = str(len(normalized))
        visible_devices = ",".join(normalized)
    else:
        raise ValueError("runtime.gpu.devices must be 'all' or a non-empty list")

    dp_mode = str(gpu.get("dp_mode", "auto"))
    if dp_mode not in DP_MODES:
        raise ValueError("runtime.gpu.dp_mode must be auto, none, ddp, or fsdp")
    if gpu_count != "all":
        count = int(gpu_count)
        if dp_mode == "none" and count != 1:
            raise ValueError("dp_mode none requires exactly one GPU")
        if dp_mode in {"ddp", "fsdp"} and count < 2:
            raise ValueError(f"dp_mode {dp_mode} requires at least two GPUs")

    # The parallel section feeds the trainer's --dp_size/--cp_size/--tp_size;
    # the process count is derived from the degrees, so the device list must
    # decompose along cp_size * tp_size.  dp_size itself is derived per launch
    # (gpu count / (cp_size * tp_size)) unless declared, in which case it must
    # agree.
    parallel = _mapping(config.get("parallel"), "parallel")

    def _degree(name: str) -> int:
        value = parallel.get(name, 1)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"parallel.{name} must be an integer >= 1")
        return value

    cp_size = _degree("cp_size")
    tp_size = _degree("tp_size")
    yaml_dp_size = _degree("dp_size")
    if gpu_count != "all":
        count = int(gpu_count)
        world = cp_size * tp_size
        if count % world != 0:
            raise ValueError(
                f"runtime.gpu.devices count ({count}) must be divisible by "
                f"parallel.cp_size * tp_size ({world})"
            )
        if "dp_size" in parallel and count != yaml_dp_size * world:
            raise ValueError(
                f"parallel degrees dp_size x cp_size x tp_size = "
                f"{yaml_dp_size * world} must match the "
                f"runtime.gpu.devices count ({count})"
            )

    max_hours = container.get("max_duration_hours", 0)
    try:
        max_seconds = math.ceil(float(max_hours) * 3600) if max_hours else 0
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "runtime.container.max_duration_hours must be a number"
        ) from exc
    if max_seconds < 0:
        raise ValueError("runtime.container.max_duration_hours must not be negative")

    values = {
        "TRAIN_JOB_NAME": job_name,
        "TRAIN_DATA_DIR": _path(paths.get("data"), "data", path.parent),
        "TRAIN_MODEL_DIR": _path(paths.get("model"), "model", path.parent),
        "TRAIN_CHECKPOINT_DIR": _path(
            paths.get("checkpoints"), "checkpoints", path.parent
        ),
        "TRAIN_GPU_COUNT": gpu_count,
        "TRAIN_DP_MODE": dp_mode,
        "TRAIN_CP_SIZE": str(cp_size),
        "TRAIN_TP_SIZE": str(tp_size),
        "CUDA_TAG": str(container.get("cuda_tag", "cu128")),
        "TRAIN_IPC_MODE": str(container.get("ipc", "host")),
        "TRAIN_STOP_GRACE_PERIOD": str(container.get("stop_grace_period", "10m")),
        "TRAIN_STOP_TIMEOUT": str(container.get("stop_timeout_seconds", 600)),
        "CHECKPOINT_KEEP_LAST": str(container.get("checkpoint_keep_last", 5)),
        "TRAIN_MAX_DURATION_SECONDS": str(max_seconds),
    }
    if visible_devices is not None:
        values["CUDA_VISIBLE_DEVICES"] = visible_devices

    for name, value in environment.items():
        if not isinstance(name, str) or not ENV_NAME.fullmatch(name):
            raise ValueError(f"invalid runtime.environment name: {name!r}")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"runtime.environment.{name} must be a scalar")
    values["environment"] = environment
    return values


def shell_exports(runtime: dict[str, str]) -> str:
    return "\n".join(
        f"export {name}={shlex.quote(value)}"
        for name, value in runtime.items()
        if name != "environment"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("exports", "environment"))
    parser.add_argument("config")
    args = parser.parse_args()

    try:
        runtime = load_runtime(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    if args.command == "exports":
        print(shell_exports(runtime))
        return
    for name, value in runtime["environment"].items():
        rendered = "" if value is None else str(value)
        print(f"{name}={rendered}", end="\0")


if __name__ == "__main__":
    main()
