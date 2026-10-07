# Newton-Schulz integration in Muon

## Recurrence

```text
X = G / max(||G||, eps), oriented so rows <= columns
A = X @ X.T
B = b * A + c * (A @ A)
X = a * X + B @ X
```

The reusable `backend.newton_schulz.newton_schulz` function owns this recurrence,
its scratch buffers and layout selection. It accepts coefficients,
iteration count and epsilon from the caller. Muon owns momentum, Nesterov,
weight decay, learning-rate adjustment and parameter routing, then calls NS.
The NS backend resolves the three [symmetric operations](symmetric.md) once
before its loop, using independently measured plans for the Gram matrix,
polynomial and final update.

```python
from astrai.extension import newton_schulz

result = newton_schulz(matrix, (3.4445, -4.775, 2.0315), steps=5, backend="auto")
```

`backend="torch"` is the default. Enable measured CUDA operations for Muon with
`MuonAdamW(..., use_ns_kernels=True)` or the training option
`muon_ns_kernels=True`. Unsupported inputs and unknown shape plans retain
Torch matrix operations. The separate `reuse_ns_buffers` option enables only
scratch reuse. Neither changes checkpoint keys. Sharded DTensor updates
retain the existing gather/orthogonalize/scatter path.

BF16 inputs are normalized in place, matching Torch Muon semantics. Later
iterations use separate buffers, preserving the normalized caller input and
persistent momentum state. Dense transposed inputs are passed to BLAS
operations as column-major views. When a measured Gram plan prefers row-major
scratch, the first update writes that layout directly; the final update writes
the caller's orientation directly. No standalone transpose kernel or packing
copy is needed. Other inputs retain the Torch path. The CUDA module exposes
only SYRK, SYMM and candidate enumeration.

## Verification and measurement

BF16 accumulation order and symmetry enforcement can change rounding.
Tests bound five-step relative L2 error to one percent on the tested matrix
shapes. Repeated optimizer updates compare parameter and exact momentum state
with Nesterov enabled and disabled. Other checks cover independent addends,
non-production dimensions, each compiled tile, output overlap, shape fallback,
injected plans, deterministic settings and CUDA Graph replay.

```bash
python scripts/benchmark/muon_ns.py --model <model-directory> --mode graph
python scripts/benchmark/muon_ns.py --rows 6912 --cols 1536 --mode eager
python scripts/benchmark/muon_ns.py --model <model-directory> --plan <plan.json>
python scripts/benchmark/muon_ns.py --model <model-directory> --scope step --dtype bf16
```

The benchmark reads safetensors headers without loading model weights onto
the GPU. It compares Torch Muon, scratch-buffer reuse and measured kernels
in interleaved order. Graph mode measures five calls per replay; eager mode
includes dispatch gaps. Header shape counts produce a weighted NS estimate.
This excludes momentum, parameter updates, AdamW and model forward/backward,
so it is not an end-to-end training speedup. Plan files may combine unique
rows from the SYRK and SYMM sweeps.

`--scope step` includes momentum, NS, weight decay and the matrix parameter
update. It still reports a weighted estimate over independently timed matrix
shapes, excluding AdamW and the model execution. `--dtype bf16|fp32` selects
input/parameter dtype; NS computation remains BF16.
