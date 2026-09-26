# CUDA Kernels

AstrAI includes optional custom CUDA kernels for attention, rotary
embedding, and the quantized GEMM family. They are built when `nvcc` is
available and CUDA is detected. This folder is the home of the
per-kernel-family documentation — math contract first, then design notes;
the family-wide infrastructure (build system, python extension layers,
testing, file layout) lives at the bottom of this file. One folder here
maps to one family of translation units under `csrc/` (headers live in the `csrc/include/` stage tree).

## Overview
| Kernel | File | Description |
|--------|------|-------------|
| `attn_decode` | `attention/decode.cu` | GQA decode attention (split-KV) |
| `attn_prefill` | `attention/prefill.cu` | GQA prefill attention (split-Q) |
| `attn_paged_decode` | `attention/paged_decode.cu` | Paged KV cache decode attention |
| `attn_paged_prefill` | `attention/paged_prefill.cu` | Paged KV cache prefill attention (ragged batch) |
| `rotary_emb` | `rotary_emb.cu` | Fused rotary embedding (cos/sin lookup + rotation) |
| `quantize` | `quantize.cu` | FP8 quantization kernels (sm_89+) |
| `gemm` | `gemm/gemm.cu` + per-dtype-pair `gemm_*.cu` | dtype-generic tensor-core GEMM binding + one explicit `gemm_dispatch` instantiation per dtype pair (fp8 / W8A16 / W8A8 / W16A16, sm_89+) |

Additionally, optimized `.cuh` variants with tensor-core MMA (Matrix Multiply-Accumulate) exist:

| Variant | File | Optimization |
|---------|------|--------------|
| Split-KV MMA decode | `attention/decode_split_kv_mma.cuh` | Split KV across warps + MMA (sm_80+) |
| Split-Q MMA prefill | `attention/prefill_split_q_mma.cuh` | Split Q across warps + MMA (sm_80+) |

## Kernel index

| Operator | Doc | Kernel module | Python entry |
|---|---|---|---|
| Quantize (FP8) | [quantize.md](quantize.md) | `csrc/quantize.cu` (headers: `csrc/include/`) | `astrai/extension/ops/quantize.py`; strategy layer `astrai/extension/quantize.py` (`fp8_autocast`, aten::linear override) |
| GEMM / Linear (bf16 · fp8 · w8a16 · w8a8) | [gemm.md](gemm.md) | `csrc/gemm/` (headers: `csrc/include/`) | adapter `astrai/extension/ops/gemm.py` |
| Attention (decode / paged / split-Q prefill, MMA variants) | [attention.md](attention.md) | `csrc/attention/` (headers: `csrc/include/`) | `astrai/extension/ops/attention.py`; dispatch `astrai/extension/backend/attention.py` |
| Rotary embedding | [rotary.md](rotary.md) | `csrc/rotary_emb.cu` | `astrai/extension/ops/rotary.py`; dispatch `astrai/extension/backend/rotary.py` |

One entry the table does not spell out: `gemm/` also carries the **fp8
training** linear — `fp8_linear.cu` (the composed forward *and* backward in
one C++ `autograd::Function`) with its state machine `fp8_state.cuh` (rings,
weight cast cache, checkpoint snapshot). Its Python entry is the strategy
layer `astrai/extension/quantize.py` (`fp8_autocast`, recipe/format policy),
and it ships inside the `gemm` module so the dispatch state — plan table,
planner mode, staging switches — has exactly one owner.

Conventions: every operator doc leads with its **math contract** — if the
contract and the kernel disagree, one of them is a bug, fixed in the same
commit as the kernel change. Performance numbers carry the measurement
discipline (production dispatch path, interleaved A/B rounds, median).
Required-reading gates: tile-vocabulary / planner / fp8-recipe changes →
[gemm.md](gemm.md); backend dispatch changes → the operator doc first.

## Build System
### Auto-detection

Kernels are built when **both** of these conditions are met:
1. `nvcc` is available on `PATH`
2. `torch.cuda.is_available()` returns `True`

Unless `CSRC_KERNELS=false` is set explicitly.

### Manual build

```bash
# During install
CSRC_KERNELS=true pip install -e . --no-build-isolation

# Rebuild after editing .cu/.cuh files
CSRC_KERNELS=true python setup.py build_ext --inplace
# Output: astrai/extension/lib/*.so

# Or invoke CMake directly (ASTRAI_CUDA_ARCH is required here too —
# without an arch token this configures fine and builds nothing)
cmake -S csrc -B build/cmake \
  -DASTRAI_CUDA_ARCH=120a \
  -DTORCH_HOME=<site-packages>/torch \
  -DPYTHON_INCLUDE_DIR=<python include> \
  -DPY_SOABI=cpython-312-x86_64-linux-gnu
cmake --build build/cmake -j 16
```

### Architecture flags

`setup.py` passes the GPU compute capability to CMake via `ASTRAI_CUDA_ARCH`
(a semicolon list, e.g. `"80;89;120a"`, produces one multi-arch fatbin: one
image per token, the driver picks the slice per device). A token is `<CC
digits>` with an optional arch-specific `a` suffix; every numeric gate
compares the suffix-stripped value.

There is **no CMake default**: an unset or empty `ASTRAI_CUDA_ARCH` builds no
kernel targets at all, so a GPU-less configure installs the pure-Python
package and `loader.py` simply finds no `.so` files. When the env is unset,
`setup.py` auto-detects the real GPU through `torch.cuda.get_device_capability()`
(CC 12.0 reports `120a`, so a dev build gets the full-rate fp8 cell):

- **sm_80+** (Ampere and later): the minimum for the attention family — the
  kernels are tensor-core only (`mma.sync.m16n8k16.bf16` for bf16 attention,
  `mma.sync.m16n8k32` for FP8); there is no scalar fallback.
- **sm_89+**: required for the FP8 family (`quantize`) — FP8 tensor-core
  instructions only exist on Ada/Hopper and newer. On older architectures,
  CMake emits a warning and skips the `quantize` target so the remaining CUDA
  kernels still build successfully.
- **sm_120a** (consumer Blackwell): the arch-specific pass that activates the
  fp8 `block_scale` cell. Listing plain `120` for the gemm target warns at
  configure time — the plain fp8 instruction decodes at half rate on sm_120,
  and the runtime route keys on the device's `cc == 120`, so the mismatch is
  silent without the warning.

### Build configuration

`csrc/CMakeLists.txt` defines the CUDA extension build:

```
NVCC_FLAGS = -O3 --expt-relaxed-constexpr --use_fast_math
             --ptxas-options=-O3,-v --extra-device-vectorization
             -Xfatbin --compress-all --threads=16
```

`--compress-all` stores every fatbin entry (SASS and PTX) compressed; the
driver decompresses at module load, so the executed code is unchanged and the
`.so` is ~4x smaller (measured per gemm TU: 8.03 MB -> 1.89 MB).

Each kernel in `astrai/extension/lib` is compiled as an independent pybind11 module (one `.so` per kernel, named `<kernel>.cpython-*-x86_64-linux-gnu.so`). CMake builds all registered kernel targets in parallel via `cmake --build -j N`. The target list is the **single source of truth**: the `KERNEL_MODULES` registry in `csrc/CMakeLists.txt`, one entry per module as `name|srcs...` (a module may span several TUs — gemm is split into per-dtype-pair instantiation units so the heavy template work parallelizes). Entries append conditionally: the five attention/rotary modules always, `quantize` and `gemm` only when the arch list reaches sm_89+, and nothing at all when `ASTRAI_CUDA_ARCH` is unset. `astrai/extension/loader.py` auto-discovers the compiled `.so` files, so adding a kernel means one registry entry and nothing else.

## Python Extension Architecture

The Python extension package separates low-level kernel bindings from execution
policy:

```text
astrai/extension/
├── __init__.py             # Stable public API
├── loader.py               # Optional compiled-module discovery and loading
├── dispatch.py             # The shared operator dispatcher (resolve/explain/set_op)
├── plan.py                 # The GEMM plan: config / configure / override / probe / facts /
│                           #   tiles + the runtime autotuner (policy, not marshalling)
├── quantize.py             # FP8/int8 strategy layer (fp8_autocast, recipes, quantizers)
├── ops/                    # Stateless kernel wrappers — one adapter per compiled module
│   ├── attention.py        # Stateless attention kernel wrappers
│   ├── quantize.py         # Stateless FP8 primitive wrappers
│   ├── gemm.py             # quant_gemm + the flat plan views (set_*/state/probe, raw shapes)
│   └── rotary.py           # Stateless rotary kernel wrapper
└── backend/
    ├── attention.py        # Backend selection, KV cache I/O, and fallback
    └── rotary.py           # Per-call CUDA/torch rotary dispatch
```

The dependency direction is one-way:

```text
model / inference
       |
       v
extension public API
       |
       v
backend policy  --->  ops wrappers  --->  loader  --->  compiled .so
       |
       +----------->  torch / flash-attn fallback
```

`ops` must not import `backend`. This keeps direct kernel bindings independent
of model, cache, fallback, and backend-selection policy.

### Ops Layer

`astrai.extension.ops` is the low-level boundary around compiled extensions:

- Wrappers are stateless and map Python arguments to pybind or
  `torch.library.custom_op` calls.
- Wrappers validate kernel availability and raise `RuntimeError` when a
  requested extension was not built.
- Wrappers do not choose another implementation, gather KV cache entries, or
  decide whether an input is supported by a backend.
- Tests that specifically exercise a compiled kernel may import from
  `astrai.extension.ops`.

For example, `attn_prefill(...)` means "run this CUDA kernel" rather than "run
attention using the best available implementation":

```python
from astrai.extension.ops import attn_prefill

output = attn_prefill(q, k, v, mask=mask, is_causal=True)
```

If the kernel is unavailable, this call fails. Callers that need fallback and
capability dispatch must use the public `attention(...)` entry point instead.

### Backend Layer

`astrai.extension.backend` owns execution policy:

- It selects CUDA, FlashAttention, or torch-native attention.
- It checks per-call constraints such as dtype, shape, head dimension, cache
  availability, and installed optional dependencies.
- It owns KV cache writes and reads because those operations differ by backend.
- It provides torch fallbacks and raises when an explicitly requested backend
  cannot handle a call.
- Rotary dispatch follows the same boundary without a backend class: the
  policy layer chooses the fused op for supported inference calls and otherwise
  uses the autograd-compatible torch implementation.

Normal model and inference code should import the stable API from
`astrai.extension`:

```python
from astrai.extension import ATTN_BACKEND, attention, attn_backend

output = attention(q, k, v, kv_cache=cache, layer_id=layer_id, fwd="decode")

with attn_backend(ATTN_BACKEND.TORCH_NATIVE):
    output = attention(q, k, v)
```

The package root re-exports the supported high-level API and selected direct
kernel wrappers. Internal code should use `astrai.extension.backend` only when
it needs a backend type or policy implementation, and `astrai.extension.ops`
only when it deliberately requires one exact kernel.

### Placement Rules

When extending this package:

| Change | Location |
|--------|----------|
| Add a pybind call for a compiled kernel | `astrai/extension/ops/` |
| Add argument translation required by the compiled ABI | `astrai/extension/ops/` |
| Add capability checks or implementation selection | `astrai/extension/backend/` |
| Add a torch or third-party fallback | `astrai/extension/backend/` |
| Add attention KV cache behavior | `astrai/extension/backend/attention.py` |
| Expose a supported user-facing symbol | `astrai/extension/__init__.py` |

Imports belong at module scope. Optional dependencies such as `flash_attn` may
use a module-level guarded import. Type-only imports that would create a runtime
cycle belong under `TYPE_CHECKING`.

## Standalone Testing

Each `csrc/tests/*.cu` file has the `nvcc` compile command in its header comment. Those lines carry the **reference box's** arch token (`sm_89`); substitute your own — this workspace's box is 8x RTX 5090 (`sm_120`), and an `sm_89` build only runs there through driver JIT. Example:

```bash
nvcc -I csrc/include -arch=sm_120 -O3 --use_fast_math \
     --ptxas-options=-O3,-v --extra-device-vectorization \
     -Xcompiler -fopenmp csrc/tests/attn_test.cu -o /tmp/test && /tmp/test
```

`quant_gemm_test.cu` additionally wants `-std=c++17` and has no `-fopenmp`
dependency.

Test files:
- `attn_test.cu` — decode + prefill kernels (correctness tables + benchmarks)
- `attn_paged_test.cu` — paged decode/prefill kernels
- `quant_gemm_test.cu` — quantized GEMM correctness: every dtype pair (fp8/int8/bf16 × layouts/K tiles/ragged shapes/scales/fp32 out) + a per-combo TFLOPS bench (sm_89+)

### Behavior-preservation gate (SASS digest)

> **Status (2026-09-26): the tool is in the tree** (`csrc/bench/sass_digest.py`,
> 283 lines, retrieved from `backup/csrc-stage-relayout` and extended with
> `--normalize-anon`: nvcc re-hashes a TU's on-disk path into the
> anonymous-namespace segment of its symbols, so a pure file move keeps every
> SASS body identical while renaming six symbols — that mode strips the
> segment and is the adjudication form for move-only refactors).

A refactor that claims to change nothing must prove it: `csrc/bench/sass_digest.py`
hashes every kernel symbol's SASS (`cuobjdump -sass`) out of the compile-tree
`.o` files, with REG usage riding along, and `--compare` exits nonzero on any
added/removed/changed function. Device-side dead-code deletion and host-side
dedup gate on 658/658 identity instead of re-benchmarking (2026-09-19 audit).
The one drift class a POD layout change causes (param-field offsets shift,
ptxas re-selects load widths and renumbers registers) is adjudicated by
instruction-count + mnemonic-histogram equality and the bitwise pytest suite.
The same audit closed the sass digest for shared-helper extraction in hot
device code: pulling verbatim-duplicated straight-line blocks into
`__forceinline__` helpers moves ptxas scheduling across inline boundaries
(42 mixed-pair kernels drifted SASS). That family lands under a benchmark
gate instead (2026-09-20, `d521f14`): big-shape A-B-A-B x3 (ms-scale
kernels, ±0.3% noise) plus a 6-round alternating re-measurement of the
worst cell, with the sass-identical kernels as a noise control — measured
flat (1.00x on every combo, several mixed pairs slightly faster, matching
the -0.56% instruction delta and -1..-2 registers).

## Benchmarks

Reproduce (decode + prefill in `attn_test.cu`, paged in `attn_paged_test.cu`):
```bash
nvcc -I csrc/include -arch=sm_120 -O3 --use_fast_math \
     --ptxas-options=-O3,-v --extra-device-vectorization \
     -Xcompiler -fopenmp csrc/tests/attn_test.cu -o /tmp/test && /tmp/test
```

## Known Optimization Targets

- **Decode D=256**: spill eliminated (BC=16 + STAGES=2), but still 248 regs — further tiling could help.
- **Prefill single-batch**: bandwidth low at q=kv=2048 — compute-bound near the bf16 ceiling for non-causal.
- **Decode single-batch**: bandwidth low at kv=512 — small kv underutilizes SMs despite split-KV; scales well at B=16+.

## File Layout

```
csrc/
├── CMakeLists.txt                    # CMake build: the KERNEL_MODULES registry (module name | its source TUs) + torch/pybind11 linking
├── __init__.py                       # build-time marker only (keeps `csrc` a setuptools package)
├── include/                          # THE include root: every header, angle-bracket root-qualified (<kernel/gemm.cuh>)
│   ├── policy.cuh                    #   cross-cutting: tile vocabulary + smem budget + GemmPolicy/manifests AND the runtime planning vocabulary (GemmRecipe/PlanQuery/GemmPerfClass/PlanDecision, GemmConfig + the three launch-side knobs)
│   ├── scheduler.cuh                 #   cross-cutting: grouped/plain raster mapping
│   ├── kernel/                       # entry __global__ and their composition (humming's rule: what the code IS, not which family owns it)
│   │   ├── gemm.cuh                  #     GEMM orchestrator + launch machinery (no torch; reaches the planner through policy.cuh declarations)
│   │   ├── gemm_mainloop.cuh         #     stage rings + pipelined mma.sync mainloop (+ dequantized fragment paths)
│   │   ├── attention_launch.cuh      #     pure-CUDA launch vocabulary: launchers + tile-config maps + dispatch_decode/prefill(_paged) funnels, split-K math
│   │   ├── attention_split_kv.cuh    #     decode kernel (split-KV FlashDecoding, GQA head packing) + split-combine
│   │   ├── attention_split_q.cuh     #     prefill kernel (split-Q, GQA head packing)
│   │   └── quantize.cuh              #     quantize kernels: vectorized + 64×32-tile transpose (out_layout 0/1/2, Dual as a template param)
│   ├── memory/                       # data movement with stage semantics
│   │   ├── load.cuh                  #     gemm operand loaders (typed staged tiles, congruous cp.async + zfill, crosswise direct, trans staging, PrefetchCarry)
│   │   ├── pipeline.cuh              #     raw cp.async 16B emitters + mbarrier PTX sites + PipelineSync stage pipeline
│   │   ├── tma.cuh                   #     TMA staging (sm_90+): device cp.async.bulk.tensor emitters + host tensor-map encoding + exact-match cache
│   │   └── layout_policies.cuh       #     attention KV addressing: DenseQSchedule/PackedQSchedule, ContigKV/PagedKV
│   ├── mma/                          # tensor-core backends and fragment helpers
│   │   ├── mma.cuh                   #     shared mma_sync<InT> + shapes + ldmatrix cores + typed fragment cells
│   │   └── utils.cuh                 #     attention's ldmatrix/pack helpers + online-softmax tile path
│   ├── epilogue/                     # moving/merging/writing results out
│   │   └── writer.cuh                #     gemm fused bias + scale folding + bf16/fp32 smem scatter + copy-out
│   ├── arith/                        # value transforms on register fragments
│   │   ├── softmax.cuh               #     shared online-softmax recurrence (scalar kernels, MMA tile, split-KV combine)
│   │   └── reduce.cuh                #     plus/maximum functors, warp_reduce/group_reduce, atomic_max_float
│   ├── datatype/                     # dtype traits and dequant primitives (stage-agnostic)
│   │   └── dequant.cuh               #     in-register dequant functors (DequantPair<SrcT, MmaT>: exact int8→bf16)
│   ├── utils/                        # stage-agnostic vocabulary — the sink of the include graph
│   │   ├── define.cuh                #     HOST/DEVICE_FORCEINLINE — the shared function-qualifier macros
│   │   ├── device.cuh                #     DeviceFacts geometry query (sms / smem opt-in / L2)
│   │   ├── dtype.cuh                 #     element-type words (aliases + ElemTrait, torch at::ScalarType naming)
│   │   ├── launch.cuh                #     launch-and-check macros, pure C
│   │   ├── shape.cuh / swizzle.cuh / tensor.cuh   # static geometry / staging swizzle / Tensor<Engine, Layout>
│   │   ├── gemm_common.h             #     layout tags, gemm_elem_traits, gemm_mma_traits, GemmParams POD
│   │   ├── attention_common.h        #     AttentionParams POD, TensorLayout enum
│   │   └── quantize_common.h         #     sm_at_least + kMinSmForFp8, QuantLayout, RingLayout, QuantParams POD
│   └── launcher/                     # THE HOST SURFACE — the only directory whose headers may touch torch/ATen/c10/Python (files named BY FAMILY: <family>*.h = that family's surface)
│       ├── api.h                     #     gemm C++ surface (declarations only, no py:: type)
│       ├── planning.h                #     the planner chain + recipe vocabulary + plan_raster; plan_dispatch defined non-inline — SINGLE-INCLUSION (one TU per binary: gemm.cu or a standalone harness)
│       ├── plan_table.h              #     AOT dispatch rows (TableRow): override/per-class builtin/degraded sources + GemmConfig seed
│       ├── attention.h               #     attention entry declarations (astrai::attention)
│       ├── attention_dtypes.h        #     attention ASTRAI_ATTN_DTYPE_LIST + generated unsupported-dtype refusal
│       ├── attention_entry.h         #     attention torch→POD marshalling (pack_*_params, split-partial allocation)
│       ├── gated_deltanet.h          #     the family's two entry declarations (astrai::gdn)
│       ├── quantize_entry.h          #     quantize launcher + composed one-call API + input-dtype list (torch-tensor level, no pybind)
│       └── fp8_checks.h              #     fp8 capability gate (check_fp8_device; shared by quantize + gemm bindings)
├── attention/                        # family translation units only (one torch entry each; kernels/launchers/dispatch in the shared kernel/ headers)
│   ├── decode.cu                     #   → module attn_decode
│   ├── prefill.cu                    #   → module attn_prefill
│   ├── paged_decode.cu               #   → module attn_paged_decode
│   └── paged_prefill.cu              #   → module attn_paged_prefill
├── gemm/                             # family translation units only (→ module gemm)
│   ├── gemm.cu                       #   typed host layer: dtype-pair registry + api.h implementations + the ONE planning.h includer
│   ├── bindings.cu                   #   pybind surface: marshalling, dict shapes, PYBIND11_MODULE
│   ├── fp8_linear.cu                 #   the fp8 training linear (fwd+bwd) as one C++ autograd::Function
│   ├── fp8_state.h                   #   the fp8 training state machine (delayed-scaling rings, cast caches, meta registry) — a TU-local split of fp8_linear.cu, quoted-include
│   └── gemm_bf16_* / gemm_*.cu       #   per-pair explicit gemm_dispatch instantiation units (one nvcc job each; plan_table-free)
├── quantize.cu                       # FP8 quantize binding only (module quantize; kernels in include/kernel/quantize.cuh)
├── gated_deltanet/                   # chunked GDN fwd/bwd kernels written in-TU + bindings.cu (→ module gated_deltanet)
├── rotary_emb.cu                     # rotary embedding (kernel + binding in one file) → module rotary_emb
├── bench/                            # measurement + dispatch-analysis tooling, run from the repo root as `python csrc/bench/<tool>.py`
│   ├── bench_tile_sweep.cu           #   cell-level tile/warp/kK sweep; standalone nvcc line in its header (no CMake target)
│   ├── benchmark_*.py                #   attention / layouts / logprobs / quant_gemm / quantize / rotary / gdn_ops benchmarks
│   ├── diff_rows.py                  #   plan rows as the measured DIFF of the model, interleaved A/B — the row-emission gate
│   ├── dispatch_grid.py              #   map the dispatch logic over a dense (m, n, k) grid, host-only
│   ├── model_capture.py              #   offline capture harness: score a planner rule against saved measurements
│   ├── sass_digest.py                #   the zero-behavior-refactor gate (per-symbol SASS digest; --normalize-anon for move-only refactors)
│   └── tune_plan_table.py            #   plan-table pipeline: sweep candidates / validate holdouts / install rows
└── tests/
    ├── test_utils.cuh                # shared test utilities (now_ms, f2bf, bf2f, randf) — harness-local, outside the include root
    ├── attn_test.cu                  # decode + prefill kernels
    ├── attn_paged_test.cu            # paged decode/prefill kernels
    └── quant_gemm_test.cu            # GEMM correctness + TFLOPS bench (includes launcher/planning.h: single-TU, torch-free)
```

Compiled `.so` files are placed in `astrai/extension/lib/`, separate from Python source files.

Three conventions the tree encodes. (1) **Stages, not families**: headers
live under `csrc/include/<stage>/` by what they do (kernel / memory / mma /
epilogue / arith / datatype / utils), the family directories hold only
translation units, and `launcher/` is the host surface — the only directory
whose headers may touch torch, which is what keeps the standalone
`csrc/tests/*.cu` harnesses torch-free. The gemm→quantize include edges
that forced a family-qualified tree before (`mainloop → dequant`,
`fp8_linear → quantize launch`) are plain kernel→datatype and TU→launcher
edges now. (2) The standalone harnesses stay out of the CMake registry on
purpose: each carries its own `nvcc` line so a correctness test runs
without torch; the `bench/` python tools are the reproduce path for the
numbers in the operator docs and workspace `notes/`. (3) **One include
root, `csrc/include`, every project include angle-bracket and
root-qualified** (`<kernel/gemm.cuh>`, `<policy.cuh>`); quoted includes are
the toolchain's plus the harness-local `test_utils.cuh`. A quoted project
path would resolve through the includer's own directory first and silently
change meaning on a move — the three same-named `common.h` are now
`utils/{gemm,attention,quantize}_common.h` precisely so a spelling names
one file. `tests/extension/test_csrc_layout.py` pins all of this — the
stage set is closed, stage headers never include `launcher/` (zero
exceptions since the planning split), `launcher/planning.h` has exactly one
includer per binary, every project include is root-qualified, unshadowed
and resolvable, and the harnesses stay torch-free. A new stage directory is
registered in that test's `STAGES` and in this section together.

Compiled `.so` files are placed in `astrai/extension/lib/`, separate from Python source files.

> Document Update Time: 2026-09-26
