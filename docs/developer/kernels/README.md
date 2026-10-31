# CUDA Kernels

AstrAI includes optional custom CUDA kernels for attention, rotary
embedding, SwiGLU, cross-entropy, and the quantized GEMM family. They are built when `nvcc` is
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
| `symmetric` | `symmetric.cu` | Generic BF16 SYRK/SYMM with measured tile dispatch |
| `rotary_emb` | `rotary_emb.cu` | Fused rotary forward and input gradient |
| `swiglu` | `swiglu.cu` | Fused SwiGLU forward and both input gradients |
| `cross_entropy` | `cross_entropy.cu` | Fused cross-entropy and chunked linear cross-entropy |
| `quantize` | `quantize/bindings.cu` + `quantize/entry.cu` | Shared FP8/int8 quantization kernels (sm_80+) |
| `gemm` | `gemm/entry.cu` + `gemm/instantiations/*.cu` | dtype-generic tensor-core GEMM binding + one explicit `gemm_dispatch` instantiation per dtype pair (fp8 / W8A16 / W8A8 / W16A16, sm_89+) |

Additionally, optimized `.cuh` variants with tensor-core MMA (Matrix Multiply-Accumulate) exist:

| Variant | File | Optimization |
|---------|------|--------------|
| Split-KV MMA decode | `kernel/attention/mainloop.cuh` | Split KV across warps + MMA (sm_80+) |
| Split-Q MMA prefill | `kernel/attention/mainloop.cuh` | Split Q across warps + MMA (sm_80+) |

## Kernel index

| Operator | Doc | Kernel module | Python entry |
|---|---|---|---|
| SYRK / SYMM | [symmetric.md](symmetric.md) | `csrc/newton_schulz/` | `astrai/extension/backend/newton_schulz.py`; adapter `astrai/extension/kernel/newton_schulz.py` |
| Muon Newton-Schulz | [muon_ns.md](muon_ns.md) | SYRK/SYMM or Torch | `astrai/extension/backend/newton_schulz.py` |
| Quantize (FP8 / int8) | [quantize.md](quantize.md) | `csrc/quantize/` (bindings + entry; headers: `csrc/include/`) | `astrai/extension/kernel/quantize.py`; FP8 strategy `astrai/extension/policy/quantization/autocast.py` (`fp8_autocast`, aten::linear override) |
| GEMM / Linear (bf16 · fp8 · w8a16 · w8a8) | [gemm.md](gemm.md) | `csrc/gemm/` (headers: `csrc/include/`) | adapter `astrai/extension/kernel/gemm.py` |
| Attention (decode / paged / split-Q prefill, MMA variants) | [attention.md](attention.md) | `csrc/attention/` (module `attention`; headers: `csrc/include/`) | `astrai/extension/kernel/attention.py`; dispatch `astrai/extension/backend/attention/dispatcher.py` |
| Gated DeltaNet (chunked fwd prep / bwd output stage) | [attention.md](attention.md) (§GDN) | `csrc/gated_deltanet/` (headers: `csrc/include/`) | `astrai/extension/kernel/gdn.py` |
| Rotary embedding | [rotary.md](rotary.md) | `csrc/rotary_emb.cu` | `astrai/extension/kernel/rotary.py`; dispatch `astrai/extension/backend/rotary.py` |
| SwiGLU | [swiglu.md](swiglu.md) | `csrc/swiglu.cu` | `astrai/extension/kernel/swiglu.py`; dispatch `astrai/extension/backend/swiglu.py` |
| Cross-entropy | [cross_entropy.md](cross_entropy.md) | `csrc/cross_entropy.cu` | `astrai/extension/kernel/cross_entropy.py`; `astrai/trainer/strategy.py` |

One entry the table does not spell out: `gemm/` also carries the **fp8
training** linear — `gemm/fp8/linear.cu` (the composed forward *and* backward in
one C++ `autograd::Function`) with scale rings in `gemm/fp8/ring.h`,
cast caches in `gemm/fp8/cache.h`, and registry definitions in
`gemm/fp8/state.h`. `gemm/fp8/runtime.cu` owns the single process state,
checkpoint interface, and Python registration. Its Python entry is the strategy
layer `astrai/extension/policy/quantization/autocast.py` (`fp8_autocast`, recipe/format policy),
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
# Output: astrai/extension/_C_*.so

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
  kernels use `mma.sync.m16n8k16.bf16` / `.f16` for BF16 / FP16 attention;
  there is no scalar fallback.
- **sm_80+**: required for shared quantization, BF16/INT8 GEMM and mixed
  BF16/INT8 GEMM. FP8 conversion does not require FP8 tensor-core MMA.
- **sm_89+**: required for native FP8 GEMM MMA, whose architecture-specific
  instantiations are selected independently of the quantization entry.
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
             -Xfatbin --compress-all --threads=4
```

`--compress-all` stores every fatbin entry (SASS and PTX) compressed; the
driver decompresses at module load, so the executed code is unchanged and the
`.so` is ~4x smaller (measured per gemm TU: 8.03 MB -> 1.89 MB).

Each kernel in `astrai/extension` is compiled as an independent pybind11 module (one `.so` per kernel, named `_C_<kernel>.cpython-*-x86_64-linux-gnu.so`). CMake builds all registered kernel targets in parallel via `cmake --build -j N`. The target list is the **single source of truth**: the `KERNEL_TARGETS` registry in `csrc/CMakeLists.txt`, one entry per target as `name|OBJECT/MODULE|sources...` (a module may span several TUs — gemm is split into per-dtype-pair instantiation units so the heavy template work parallelizes). The module registry is built for the selected architectures; GEMM schedules have separate architecture gates, and an unset `ASTRAI_CUDA_ARCH` builds no targets. The same creation loop handles objects and modules. Both `quantize` and `gemm` declare `$<TARGET_OBJECTS:quantize_entry>` in their source entries, so the shared Torch entry is compiled once. `astrai/extension/runtime/loader.py` auto-discovers the compiled `.so` files, so adding a kernel means one registry entry and nothing else.

## Python Extension Architecture

The Python extension package separates low-level kernel bindings from execution
policy:

```text
astrai/extension/
├── __init__.py             # Public operator API
├── runtime/                # Context, base spec/dispatcher, and lazy loader
│   ├── context.py
│   ├── operation.py
│   └── loader.py
├── backend/                # Per-operator dispatch and fallback policy
│   ├── attention/          # Dispatcher plus CUDA, FlashAttention, torch functions
│   ├── rotary.py
│   ├── swiglu.py
│   └── newton_schulz.py
├── kernel/                 # Stateless compiled bindings and INT8 primitives
│   ├── attention.py
│   ├── cross_entropy.py
│   ├── gdn.py
│   ├── gemm.py
│   ├── quantize.py
│   ├── rotary.py
│   └── swiglu.py
├── policy/                 # Operator-specific execution policy
│   ├── gemm/               # Plan configuration and runtime autotuning
│   │   ├── plan.py
│   │   └── autotune.py
│   └── quantization/       # FP8 region policy and stable module slots
│       ├── autocast.py
│       └── fp8_slots.py
└── lib/                    # Compiled .so modules
```

`gemm` is an operator family; `quantization` is policy shared by operators.
Both live under `policy/` because these modules own configuration and state.
`backend/` owns concrete execution strategies, while `kernel/` calls exact
bindings. Each operator owns its capability checks and implementation choice.

The dependency direction is one-way:

```text
model / inference
       |
       v
extension public API
       |
       v
backend policy  --->  kernel wrappers  --->  loader  --->  compiled .so
       |
       +----------->  torch / flash-attn fallback
```

`kernel` must not import `backend`. This keeps direct kernel bindings independent
of model, cache, fallback, and backend-selection policy.

### Ops Layer

`astrai.extension.kernel` is the low-level boundary around compiled extensions:

- Wrappers are stateless and map Python arguments to pybind or
  `torch.library.custom_op` calls.
- Wrappers validate kernel availability and raise `RuntimeError` when a
  requested extension was not built.
- Wrappers do not choose another implementation, gather KV cache entries, or
  decide whether an input is supported by a backend.
- Tests that specifically exercise a compiled kernel may import from
  `astrai.extension.kernel`.

For example, `attn_prefill(...)` means "run this CUDA kernel" rather than "run
attention using the best available implementation":

```python
from astrai.extension.kernel import attn_prefill

output = attn_prefill(q, k, v, mask=mask, is_causal=True)
```

If the kernel is unavailable, this call fails. Callers that need fallback and
capability dispatch must use the public `attention(...)` entry point instead.

### Operator dispatch

`astrai.extension.runtime` owns the `ContextVar` selection and the abstract
`OpSpec` and `OpDispatcher` contracts. Each operator has its own spec and
dispatcher in `astrai.extension.backend`; there is no shared implementation
registry. The spec records device, dtype, phase, backward need, and
operator-specific capability facts for one forward call.

Attention dispatches directly to module functions in `backend/attention/`.
Rotary and SwiGLU use their own dispatchers; SYRK/SYMM keep shape planning in
`policy.newton_schulz`. Exact compiled operations remain in `kernel/`. An
explicit argument or `op_backend(...)` context is strict when unsupported.
`ASTR_OPS` and `set_op(...)` are soft process defaults.

```python
from astrai.extension import attention, op_backend

with op_backend(attention="torch"):
    output = attention(q, k, v)
```

Inference snapshots the selection when its scheduler is created and installs
that snapshot in the worker. Requests do not carry an implementation choice.

### Placement Rules

When extending this package:

| Change | Location |
|--------|----------|
| Add a pybind call for a compiled kernel | `astrai/extension/kernel/` |
| Add argument translation required by the compiled ABI | `astrai/extension/kernel/` |
| Add an implementation to an operator | Extend that operator's `select` and capability check in `backend/` |
| Add a new operator | Add an `OpSpec` and `OpDispatcher` subclass plus its public entry point |
| Add an attention implementation | Add functions under `backend/attention/` and update `AttentionDispatcher.select` |
| Add attention KV cache behavior | `astrai/extension/backend/attention/` |
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
├── CMakeLists.txt                    # CMake build: the KERNEL_TARGETS registry (name | OBJECT/MODULE | source TUs) + torch/pybind11 linking
├── __init__.py                       # build-time marker only (keeps `csrc` a setuptools package)
├── include/                          # THE include root: every header, angle-bracket root-qualified (<kernel/gemm/kernel.cuh>)
│   ├── policy/                       # policy composition, grouped by operator
│   │   ├── gemm.cuh                  # composed GemmPolicy consumed by the kernel
│   │   └── gemm/
│   │       ├── traits.cuh            # compute promotion, MMA traits and shared-memory budget
│   │       ├── geometry.h            # pure C++ layout and ring-budget formulas
│   │       ├── manifest.h            # pure C++ named tile recipes and staging ladders
│   │       ├── cost.h                # geometry cost shared with SYRK/SYMM
│   │       └── config.h              # process-wide GEMM switches and environment seeding
│   ├── scheduler.cuh                 #   cross-cutting: grouped/plain raster mapping
│   ├── kernel/                       # device kernels and launch helpers, grouped by operator
│   │   ├── attention/
│   │   │   ├── mma.cuh               # attention fragments and softmax helpers
│   │   │   ├── mainloop.cuh          # prefill and decode mainloops
│   │   │   ├── epilogue.cuh          # output stores and split-KV combine
│   │   │   └── kernel.cuh            # mainloop/epilogue composition
│   │   ├── gemm/
│   │   │   ├── sm80.cuh              # staged mma.sync mainloop
│   │   │   └── kernel.cuh            # cp.async and TMA entry kernels
│   │   └── quantize/
│   │       └── kernel.cuh            # vectorized and transpose kernels
│   ├── memory/                       # data movement with stage semantics
│   │   ├── load_async.cuh            #     cp.async operand staging + PrefetchCarry (congruous and 16-bit transposed)
│   │   ├── load_crosswise.cuh        #     direct 8-bit crosswise staging + CrosswiseCarry
│   │   ├── load_crosswise_packed.cuh #     packed k-pair crosswise staging + PairPackCarry
│   │   ├── pipeline.cuh              #     raw cp.async 16B emitters + mbarrier PTX sites + PipelineSync stage pipeline
│   │   ├── tma.cuh                   #     TMA staging (sm_90+): device cp.async.bulk.tensor emitters + host tensor-map encoding + exact-match cache
│   │   └── layout_policies.cuh       #     attention KV addressing: DenseQSchedule/PackedQSchedule, ContigKV/PagedKV
│   ├── mma/                          # tensor-core backends and fragment helpers
│   │   ├── mma.cuh                   #     common typed fragment contract
│   │   ├── sm80.cuh / sm89.cuh       #     own BF16/INT8 and FP8 warp MMA instructions
│   │   ├── sm120.cuh                 #     block-scaled FP8 MMA instructions
│   │   └── ldmatrix.cuh              #     shared fragment loads
│   ├── epilogue/                     # moving/merging/writing results out
│   │   └── writer.cuh                #     gemm fused bias + scale folding + bf16/fp32 smem scatter + copy-out
│   ├── arith/                        # value transforms on register fragments
│   │   ├── softmax.cuh               #     shared online-softmax recurrence (scalar kernels, MMA tile, split-KV combine)
│   │   └── reduce.cuh                #     plus/maximum functors, warp_reduce/group_reduce, atomic_max_float
│   ├── datatype/                     # shared element traits and dequant primitives
│   │   ├── element.cuh               #     storage sizes, native pair conversions, FP8 formats
│   │   └── dequant.cuh               #     in-register dequant functors (DequantPair<SrcT, MmaT>: exact int8→bf16)
│   ├── utils/                        # stage-agnostic tools — the sink of the include graph
│   │   ├── define.cuh                #     HOST/DEVICE_FORCEINLINE — the shared function-qualifier macros
│   │   ├── device.cuh                #     DeviceFacts geometry query (sms / smem opt-in / L2)
│   │   ├── launch.cuh                #     launch-and-check macros, pure C
│   │   └── shape.cuh / swizzle.cuh / tensor.cuh   # static geometry / staging swizzle / Tensor<Engine, Layout>
│   ├── api/                          # caller contracts, split by framework dependency
│   │   ├── native/                   # no Torch/ATen/c10/Python; ordinary C++ value contracts
│   │   │   ├── gemm.h                # layout tags and unchanged GemmParams POD
│   │   │   ├── gemm_plan.h           # query/decision, config, capabilities and probe results
│   │   │   ├── device.h              # DeviceFacts value contract; querying stays in utils/device.cuh
│   │   │   ├── attention.h           # AttentionParams, tensor layouts and workspace constants
│   │   │   └── quantize.h            # QuantParams and shared RingLayout slot offsets
│   │   └── torch/                    # Tensor/ScalarType interfaces and framework adapters
│   │       ├── gemm.h                # quant_gemm_impl and ScalarType-based plan_probe
│   │       ├── attention.h           # attention Tensor entry declarations
│   │       ├── gated_deltanet.h       # GDN Tensor entry declarations
│   │       ├── quantize.h            # QuantizeOutputs, quantize and quantize_scaled
│   │       ├── dtype.h               # CUDA-free Torch dtype names, matching, classification and quantization ranges
│   │       ├── dispatch.cuh          # CUDA element/ScalarType mapping and type-list visitors
│   └── launcher/                     # GEMM CUDA launch and dispatch templates
│       ├── quantize.cuh              #     native fused quantize tile grid
│       ├── gemm_launch.cuh           #     typed CUDA launch and TMA descriptor setup
│       ├── gemm_tiles.cuh            #     manifest selection and staging resolution
│       └── gemm_dispatch.cuh         #     definitions, layout rewrite and native planner probe
├── attention/                        # private entry.h owns the BF16/FP16 type list; family translation units + one bindings.cu (→ module attention; kernels/launchers/dispatch in the shared kernel/ headers)
│   ├── entry.h                       #   attention torch→POD marshalling (pack_*_params, split-partial allocation) — TU-local impl header, quoted-include (gemm/fp8/state.h shape)
│   ├── decode.cu                     #   → module attn_decode
│   ├── prefill.cu                    #   → module attn_prefill
│   ├── paged_decode.cu               #   → module attn_paged_decode
│   └── paged_prefill.cu              #   → module attn_paged_prefill
├── gemm/                             # one module; each subdirectory has one responsibility
│   ├── bindings.cu                   # Python argument marshalling and module registration
│   ├── entry.cu                      # Tensor validation/packing, shared dtype/schedule selection and probe
│   ├── plan/                         # ordinary C++ host planning
│   │   ├── planner.cpp               # recipe ranking, geometry model and raster selection
│   │   ├── table.cpp / table.h        # row parsing, storage, configuration and lookup
│   │   └── builtin.cpp               # generated measured and degraded rows
│   ├── fp8/                          # training composition and persistent state
│   │   ├── linear.cu / linear.h       # forward/backward autograd and runtime declaration boundary
│   │   ├── runtime.cu                # registry implementation, one State instance and checkpoint boundary
│   │   ├── ring.h                    # delayed-scaling recipe and rings
│   │   ├── cache.h                   # versioned weight and bounded activation cast caches
│   │   └── state.h                   # typed registry, slots and checkpoint snapshots
│   └── instantiations/               # per-pair explicit CUDA instantiations, compiled per schedule
│       └── *.cu                      # BF16/BF16, BF16/INT8, INT8/INT8 and matching FP8 formats
├── symmetric/                        # BF16 SYRK/SYMM (→ module symmetric)
│   ├── bindings.cu                   #   pybind surface
│   ├── entry.cu                      #   tensor validation and GEMM parameter packing
│   ├── entry.h                       #   private host/launcher declarations
│   └── kernels.cu                    #   typed CUDA kernels, launch dispatch and resource planning
├── quantize/                         # family translation units (→ module quantize; entry.cu compiled once and linked into both modules)
│   ├── bindings.cu                   #   named pybind functions (quantize / quantize_scaled)
│   └── entry.cu                      #   shared validation, allocation, ring binding and dtype dispatch
├── gated_deltanet/                   # chunked GDN fwd/bwd kernels written in-TU + bindings.cu (→ module gated_deltanet)
├── rotary_emb.cu                     # rotary embedding (kernel + binding in one file) → module rotary_emb
├── bench/                            # measurement + dispatch-analysis tooling, run from the repo root as `python csrc/bench/<tool>.py`
│   ├── bench_tile_sweep.cu           #   cell-level tile/warp/kK sweep; standalone nvcc line in its header (no CMake target)
│   ├── benchmark_gemm.py             #   GEMM dtype and operand-layout comparisons (dtypes / layouts)
│   ├── benchmark_*.py                #   attention / logprobs / quantize / rotary / gdn_ops benchmarks
│   ├── dispatch_grid.py              #   map the dispatch logic over a dense (m, n, k) grid, host-only
│   ├── model_capture.py              #   offline capture harness: score a planner rule against saved measurements
│   ├── sass_digest.py                #   the zero-behavior-refactor gate (per-symbol SASS digest; --normalize-anon for move-only refactors)
│   └── tune_plan_table.py            #   plan-table pipeline: sweep / diff / validate / install rows
└── tests/
    ├── test_utils.cuh                # shared test utilities (now_ms, f2bf, bf2f, randf) — harness-local, outside the include root
    ├── attn_test.cu                  # decode + prefill kernels
    ├── attn_paged_test.cu            # paged decode/prefill kernels
    └── quant_gemm_test.cu            # GEMM correctness + TFLOPS bench (links gemm/plan/{planner,table,builtin}.cpp)
```

GEMM benchmark entry points:

```bash
.venv/bin/python csrc/bench/benchmark_gemm.py dtypes --help
.venv/bin/python csrc/bench/benchmark_gemm.py layouts --help
.venv/bin/python csrc/bench/tune_plan_table.py diff --help
```

Compiled `.so` modules are placed directly in `astrai/extension/`.

Three conventions the tree encodes. (1) **Stages, not families**: device
headers live under `csrc/include/<stage>/` by what they do (kernel / memory /
mma / epilogue / arith / datatype / utils); `kernel/` groups by operator and
`csrc/<operator>/` holds translation units and private implementation headers.
The caller contracts remain together in `api/`, with a framework boundary:
`api/torch/` may depend on `api/native/`, never the reverse. Native contracts,
planners and shared device headers must not transitively include Torch, ATen,
c10, pybind11 or Python. Native parameter structs use plain pointers; their
headers must compile as ordinary C++ without CUDA compatibility macros.
Family-private host headers may use Torch. Tensor
validation, allocation and packing stay in entries; Python marshalling stays
in binding implementations. `api/native/gemm_plan.h` contains ordinary C++
query/config/result types; `policy/gemm/config.h` retains the existing single
runtime config owner and helpers. Compute promotion lives in
`policy/gemm/traits.cuh`; the manifest and shared geometry remain ordinary C++.
(2) Standalone harnesses stay outside the module registry and carry their own
`nvcc` commands. GEMM harnesses link `gemm/plan/planner.cpp`, `gemm/plan/table.cpp` and
`gemm/plan/builtin.cpp`, without Torch headers or libraries. (3) One project
include root, `csrc/include`, and root-qualified angle includes for shared
headers, such as `<api/native/gemm.h>` and `<api/torch/gemm.h>`. Private
same-family implementation headers and harness-local helpers use quoted
includes.

Dtype dispatch uses `TypeList<Ts...>` and `DTypePair<A,B>` with shared
`visit_dtype<List>` / `visit_dtype_pair<List>` templates in
`api/torch/dispatch.cuh`. Each family owns its supported list: attention keeps
BF16/FP16 in its private `attention/entry.h`, GEMM keeps legal A/B pairs in
its entry, and quantize keeps BF16/FP16/FP32 in its entry. Names and support queries
derive from the same lists as dispatch. There is no Cartesian expansion of
GEMM operand types. Duplicate dispatch keys fail compilation. An empty list
supports queries and diagnostics, but cannot instantiate a visitor.

Visitors invoke the callback once on a match and preserve void, value or
reference returns. Every candidate must return the same type. A miss invokes
the caller's error formatter and throws `std::runtime_error`. Attention and
quantize share `unsupported_dtype_message<List>`. Callback exceptions
propagate unchanged. Compile-time recursion supplies the small
runtime comparison chain, with no runtime registry or allocation.

`api/torch/dtype.h` holds CUDA-free dtype names, metadata and FP8/int8 range
queries. Quantize bindings and `ScaleRing` use that header directly; their
range queries do not depend on CUDA dispatch. The typed visitors and CUDA
element mapping stay in `api/torch/dispatch.cuh`. GEMM's
Torch entry includes `launcher/gemm_dispatch.cuh` for native launch and probe,
with no local forward declarations. That header owns definitions, defaults,
and the external instantiations selected by the extension's `gemm_build.h`
manifest. The per-pair/schedule units still emit the kernels for their CUDA
architecture images; standalone native callers instantiate directly.
The instantiation macro shares the signature between these build paths.

Check rejected template configurations without GPU
launches (a C++17 compiler and CUDA element headers are required):

```bash
.venv/bin/python -m pytest tests/unit/extension/test_torch_dispatch.py -q
```

Check the dependency boundary and compile the host planners independently:

```bash
.venv/bin/python -m pytest tests/unit/extension/test_csrc_api.py -q
g++ -std=c++17 -I csrc/include -fsyntax-only \
    csrc/gemm/plan/planner.cpp csrc/gemm/plan/table.cpp \
    csrc/gemm/plan/builtin.cpp
```

The dependency test walks project includes transitively, including private
planner headers. The existing module names, Tensor interfaces, parameter layouts, FP8 state
ownership and compiled architecture selection are unchanged. Attention now
instantiates BF16 and FP16; FP32 continues to use the existing fallback or fail
at the direct native entry. The BF16 computation is unchanged.

A fourth convention, the mirror of the header fan-in rule: **implementation
code lives in the TU territory of its consumers**. An implementation header
whose every includer sits in one family directory lives beside them
(`attention/entry.h`, `gemm/fp8/state.h` — quoted same-directory include);
an implementation shared across modules becomes a .cu listed in each
module's CMake sources (`quantize/entry.cu`, compiled once and linked into both the
quantize and gemm modules), with its Tensor declaration in `api/torch/`.
`launcher/` owns GEMM CUDA dispatch. `api/native/gemm_plan.h` carries the shared query
and launch decision; `policy/gemm/config.h` owns internal configuration helpers. `gemm/plan/planner.cpp` ranks recipes, while `gemm/plan/table.cpp`
owns mutable row state and parsing; generated rows live in `gemm/plan/builtin.cpp`.
The dtype-pair
instantiation units include neither planner nor table headers. A family
gains a directory when it gains a second file; single-file families
(`rotary_emb.cu`) stay at the top level.

Compiled `.so` modules are placed directly in `astrai/extension/`.

> Document Update Time: 2026-10-05
