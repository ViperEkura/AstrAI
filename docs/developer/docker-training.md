# Containerized Training Deployment

AstrAI uses one training YAML as the declaration for both host-side container
runtime settings and in-container training settings. Do not invoke the trainer
with raw `docker compose up`; use `scripts/train.sh` so preflight validation,
checkpoint recovery, and graceful shutdown remain active.

## Architecture

```text
train.yaml
  ├── runtime                parsed on the host before Docker starts
  └── model/data/...         parsed by train.py inside the container
        │
scripts/train.sh             preflight, Compose wrapper, lifecycle, timer
  └── docker-compose.yml     GPU passthrough, mounts, image, container limits
        └── scripts/docker/train-entrypoint.sh   process count, parallel mode, auto-resume
              └── train.py --config /run/astrai/train.yaml
```

The two parsers deliberately own different sections. `scripts/docker/train_runtime.py`
reads only `runtime`; `scripts/tools/train.py` reads only
`model/data/parallel/training/ckpt/log`. Explicit trainer arguments after `--`
override training YAML values.

## Runtime Schema

```yaml
runtime:
  job_name: astrai-train
  paths:
    data: ./data
    model: ./params
    checkpoints: ./checkpoints
  gpu:
    devices: all
    dp_mode: auto  # one GPU: none; multiple GPUs: ddp
  container:
    cuda_tag: cu128
    ipc: host
    stop_grace_period: 10m
    stop_timeout_seconds: 600
    checkpoint_keep_last: 5
    # max_duration_hours: 12
  # Optional; entries are passed verbatim into the trainer container
  # (see "Per-Job Environment"):
  # environment:
  #   ASTR_LOG_LEVEL: DEBUG
  #   ASTR_BACKEND: torch_native
```

- Relative paths resolve from the YAML file's directory, not the current shell.
- `devices` is either `all` or a non-empty physical GPU index list. Compose
  passes all GPUs once; `CUDA_VISIBLE_DEVICES` performs the only filtering.
- The process count is derived from `devices`. With `all`, the entrypoint uses
  `torch.cuda.device_count()` after Docker starts.
- `dp_mode: auto` selects `none` for one GPU and `ddp` for multiple GPUs.
  Use `fsdp` explicitly when model sharding is required.
- To select specific physical GPUs, replace `all` with a list such as
  `devices: [0, 1]`.
- `environment` entries apply only to the job defined by this YAML file, not to
  the host or to other jobs. Keep the section omitted unless this job's GPU
  selection needs it; see [Per-Job Environment](#per-job-environment).
- `max_duration_hours` starts a detached host timer that calls the same graceful
  `stop` command. A manual stop cancels the timer.

## Per-Job Environment

`runtime.environment` is scoped to one job. `start` passes only the entries of
the config file it was given, so a variable reaches exactly the GPUs declared
in that file's `runtime.gpu.devices` and nothing else. Two jobs on the same
machine can therefore differ: a job whose GPUs have working peer-to-peer keeps
the section omitted, a job whose GPUs cross broken PCIe/NVLink paths declares
the NCCL workarounds, and a job on an NVSwitch fabric can pin the NVLink fast
path on.

Because of that scoping, the effective pattern is one YAML per GPU group
rather than one shared YAML that gets edited whenever the device list changes:

```yaml
# train-local.yaml: GPUs with working peer-to-peer; nothing to declare
runtime:
  gpu:
    devices: [0, 1]

# train-cross-pcie.yaml: this GPU set crosses broken paths, so only this job
# declares the workarounds (confirm first; see docs/guides/distributed.md)
runtime:
  gpu:
    devices: [4, 5, 6, 7]
  environment:
    NCCL_P2P_DISABLE: "1"
    NCCL_NET_GDR_LEVEL: "0"
```

The same mechanism carries positive tuning, not just workarounds. On an
NVSwitch node (Hopper-class GPUs with fabric manager running), NVLink SHARP
multicast (NVLS) is the fast allreduce path and NCCL enables it automatically
where supported. A job may pin it on explicitly and raise channel parallelism
when benchmarks show the NVLink bandwidth is underused:

```yaml
# train-nvlink.yaml: NVSwitch node; keep the disables OUT and pin the fast
# path on instead (verify support with NCCL_DEBUG=INFO first)
runtime:
  gpu:
    devices: [0, 1, 2, 3]
  environment:
    NCCL_NVLS_ENABLE: "1"
    NCCL_MIN_NCHANNELS: "8"
    # NCCL_ALGO: NVLS   # force one algorithm; unsupported values fail loudly
```

NVLS requires NVSwitch multicast support; on plain NVLink bridges or PCIe-only
sets, keep the section omitted and let NCCL pick Ring/Tree with P2P. Newer
drivers list the actual interconnect and NVLS support directly in
`nvidia-smi topo -m`, so check that before assuming.

Confirm a variable is needed before adding it, and only in the YAML of the job
that hits the problem:

```bash
nvidia-smi topo -m    # check P2P support between exactly the selected GPUs
NCCL_DEBUG=INFO       # confirm NCCL transport errors before disabling them
```

See `docs/guides/distributed.md` for what each troubleshooting variable
disables. The two directions are mutually exclusive: `NCCL_P2P_DISABLE` and
`NCCL_NET_GDR_LEVEL` remove bandwidth and must never appear in the same
environment as the NVLink entries above.

Semantics:

- Values must be scalars and are rendered with `str()`, so quote them
  explicitly (`"1"`, `"0"`) instead of relying on YAML booleans or numbers.
- A `null` value exports the name with an empty value.
- This section is the only path for extra host variables into the trainer
  container; variables exported in the host shell do not pass through Compose.

## Fixed Container Paths

| Runtime path | Container path | Access |
|---|---|---|
| `runtime.paths.data` | `/data` | read-only |
| `runtime.paths.model` | `/models/base` | read-only |
| `runtime.paths.checkpoints` | `/checkpoints` | read-write |
| the selected YAML | `/run/astrai/train.yaml` | read-only |

Training configuration must therefore use `data_root_path: /data`. The source
code is baked into `/app`; `start` reuses the existing image, so run
`bash scripts/train.sh build [CONFIG]` after code changes.

## Operations

The config argument defaults to `./train.yaml`:

```bash
bash scripts/train.sh init [CONFIG]
bash scripts/train.sh preflight [CONFIG]
bash scripts/train.sh start [CONFIG]
bash scripts/train.sh start [CONFIG] --foreground -- --dry-run
bash scripts/train.sh logs [CONFIG]
bash scripts/train.sh status [CONFIG]
bash scripts/train.sh stop [CONFIG]
bash scripts/train.sh restart [CONFIG]
bash scripts/train.sh clean [CONFIG] --keep 5
bash scripts/train.sh clean [CONFIG] --keep 5 --force
```

`init` creates the declared runtime directories but does not generate or mutate
the YAML. `preflight` validates Docker, paths, base model files, checkpoint
writability, GPU configuration, and rendered Compose configuration.

## Checkpoint Recovery

Checkpoints are stored below
`runtime.paths.checkpoints/<job_name>/epoch_<N>_step_<N>`. A checkpoint is
complete only when it contains:

```text
meta.json
config.json
model.safetensors
optimizer.pt
scheduler.pt
manifest.json
```

`online_ppo` jobs additionally require `value_model.pt` and
`value_optimizer.pt` (the critic state); the completeness check derives this
from the training config's `train_type`.

New checkpoints write `manifest.json` after every payload file, sync the complete
staging directory, and then atomically rename that directory into place. Legacy
checkpoints without a manifest remain resumable when the original required files
are complete.

`start` resumes the latest complete checkpoint and ignores partial writes. If no
complete checkpoint exists, `/models/base/config.json` and
`/models/base/model.safetensors` are required. `stop` sends `SIGTERM`; the
trainer finishes at a batch boundary and saves an emergency checkpoint before
the Docker timeout expires.

## Hard Rules

1. Keep Docker settings in `runtime` and trainer settings in the remaining YAML sections.
2. Filter GPUs once: Compose passes `count: all`; `devices` becomes `CUDA_VISIBLE_DEVICES`.
3. Do not force DDP for a model that requires FSDP; declare the mode explicitly.
4. Do not use `kill -9` for routine shutdown; use `scripts/train.sh stop CONFIG`.
5. The image user is built with the host UID/GID so mounted checkpoints retain usable ownership.
6. Scope `runtime.environment` to the job YAML that needs it; do not copy NCCL
   workarounds into every config.

> Document Update Time: 2026-08-29
