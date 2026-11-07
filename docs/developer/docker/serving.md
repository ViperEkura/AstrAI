# Containerized Serving Deployment

AstrAI uses one serving YAML as the declaration for both host-side container
runtime settings and in-container server settings. `scripts/serve.sh` wraps the
Compose commands so preflight validation and container lifecycle stay
consistent with the trainer.

## Architecture

```text
serve.yaml
  ├── runtime                parsed on the host before Docker starts
  └── server                 parsed by server.py inside the container
        │
scripts/serve.sh             preflight, Compose wrapper, lifecycle
  └── docker-compose.yml     GPU passthrough, mounts, image, port mapping
        └── server.py --config /run/astrai/serve.yaml
```

`scripts/docker/serve_runtime.py` reads `runtime:` plus the two container-side
values Compose needs (`server.port` for the port mapping, `server.device` for
the preflight GPU check). `scripts/tools/server.py --config` reads `server:`.
Explicit CLI arguments to `server.py` override `server:` YAML values.

## Runtime Schema

```yaml
runtime:
  job_name: serve
  port: 8000
  paths:
    param: ./params
  gpu:
    enabled: true  # false → cpu profile (server-cpu service)
    devices: all   # all | [0]
  container:
    cuda_tag: cu130
  # environment:
  #   TOKENIZERS_PARALLELISM: "false"

server:
  host: 0.0.0.0
  port: 8000
  device: cuda      # cuda | cpu
  dtype: bfloat16   # bfloat16 | float16 | float32
  max_batch_size: 16
  max_seq_len: null # falls back to model config
```

- Relative paths resolve from the YAML file's directory, not the current shell.
- `runtime.port` is the host publish port; `server.port` is the port the
  container listens on. The Compose mapping is
  `${SERVE_PORT}:${SERVE_CONTAINER_PORT}`.
- `runtime.gpu.enabled: true` (default) selects the `server` service with an
  NVIDIA device reservation; `false` selects `server-cpu` (no GPU passthrough).
  When disabled, `server.device` must be `cpu`.
- `runtime.gpu.devices` is `all` (default) or a single-device list such as `[0]`;
  the list becomes `CUDA_VISIBLE_DEVICES`. Compose passes `count: all`; the
  env var performs the only filtering.
- `environment` values are explicitly passed to the serving container. Keep
  host-specific settings here; they are not universal defaults.
- `server.device` must agree with `runtime.gpu.enabled`; `preflight` enforces it.

## Fixed Container Paths

| Runtime path | Container path | Access |
|---|---|---|
| `runtime.paths.param` | `/app/params` | read-only |
| the selected YAML | `/run/astrai/serve.yaml` | read-only |

`server.param_path` is optional: the server default is
`project_root/params`, which is exactly `/app/params` inside the container
(the working directory is `/app`). Set it explicitly only when serving from a
different location; in Docker it must be a container path.

## Operations

The config argument defaults to `./serve.yaml`:

```bash
bash scripts/serve.sh init [CONFIG]
bash scripts/serve.sh preflight [CONFIG]
bash scripts/serve.sh up [CONFIG]
bash scripts/serve.sh run [CONFIG]
bash scripts/serve.sh down [CONFIG]
bash scripts/serve.sh restart [CONFIG]
bash scripts/serve.sh logs [CONFIG]
bash scripts/serve.sh status [CONFIG]
```

`preflight` validates Docker, the model directory
(`config.json` + `model.safetensors`), GPU/device consistency, and the
rendered Compose configuration. `up` starts the container detached; `run`
keeps it in the foreground. Both reuse the existing image; run
`bash scripts/serve.sh build [CONFIG]` after code changes. The wrapper manages a fixed container name
(`astrai-server` or `astrai-server-<job_name>`); the plain
`docker compose up -d` / `docker compose --profile cpu up -d` path keeps
working with defaults (port 8000, `./params`).

## Hard Rules

1. Keep Docker settings in `runtime` and server settings in `server`.
2. Filter GPUs once: Compose passes `count: all`; a `devices`
   list becomes `CUDA_VISIBLE_DEVICES`.
3. `runtime.gpu.enabled: false` requires `server.device: cpu`.
4. In Docker, `server.port` must match the published container port (default
   `8000`); change `runtime.port` to publish on a different host port.
5. The image user is built with the host UID/GID so the mounted model
   directory stays readable.
