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
| `symmetric` | `symmetric.cu` | Generic BF16 SYRK/SYMM with measured tile dispatch |
| `rotary_emb` | `rotary_emb.cu` | Fused rotary embedding (cos/sin lookup + rotation) |
| `quantize` | `quantize/bindings.cu` + `quantize/entry.cu` | FP8 quantization kernels (sm_89+) |
| `gemm` | `gemm/gemm.cu` + per-dtype-pair `gemm_*.cu` | dtype-generic tensor-core GEMM binding + one explicit `gemm_dispatch` instantiation per dtype pair (fp8 / W8A16 / W8A8 / W16A16, sm_89+) |

Additionally, optimized `.cuh` variants with tensor-core MMA (Matrix Multiply-Accumulate) exist:

| Variant | File | Optimization |
|---------|------|--------------|
| Split-KV MMA decode | `attention/decode_split_kv_mma.cuh` | Split KV across warps + MMA (sm_80+) |
| Split-Q MMA prefill | `attention/prefill_split_q_mma.cuh` | Split Q across warps + MMA (sm_80+) |

## Kernel index

| Operator | Doc | Kernel module | Python entry |
|---|---|---|---|
| SYRK / SYMM | [symmetric.md](symmetric.md) | `csrc/symmetric/` | `astrai/extension/backend/symmetric.py`; adapter `astrai/extension/kernel/symmetric.py` |
| Muon Newton-Schulz | [muon_ns.md](muon_ns.md) | SYRK/SYMM or Torch | `astrai/extension/backend/newton_schulz.py` |
| Quantize (FP8) | [quantize.md](quantize.md) | `csrc/quantize/` (bindings + entry; headers: `csrc/include/`) | `astrai/extension/kernel/quantize.py`; strategy layer `astrai/extension/quantize.py` (`fp8_autocast`, aten::linear override) |
| GEMM / Linear (bf16 · fp8 · w8a16 · w8a8) | [gemm.md](gemm.md) | `csrc/gemm/` (headers: `csrc/include/`) | adapter `astrai/extension/kernel/gemm.py` |
| Attention (decode / paged / split-Q prefill, MMA variants) | [attention.md](attention.md) | `csrc/attention/` (module `attention`; headers: `csrc/include/`) | `astrai/extension/kernel/attention.py`; dispatch `astrai/extension/backend/attention.py` |
| Gated DeltaNet (chunked fwd prep / bwd output stage) | [attention.md](attention.md) (§GDN) | `csrc/gated_deltanet/` (headers: `csrc/include/`) | `astrai/extension/kernel/gdn.py` |
| Rotary embedding | [rotary.md](rotary.md) | `csrc/rotary_emb.cu` | `astrai/extension/kernel/rotary.py`; dispatch `astrai/extension/backend/rotary.py` |

One entry the table does not spell out: `gemm/` also carries the **fp8
training** linear — `fp8_linear.cu` (the composed forward *and* backward in
one C++ `autograd::Function`) with scale rings in `gemm/fp8_ring.h`,
cast caches in `gemm/fp8_cache.h`, and registry definitions in
`gemm/fp8_state.h`. `gemm/fp8_runtime.cu` owns the single process state,
checkpoint interface, and Python registration. Its Python entry is the strategy
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

Each kernel in `astrai/extension` is compiled as an independent pybind11 module (one `.so` per kernel, named `_C_<kernel>.cpython-*-x86_64-linux-gnu.so`). CMake builds all registered kernel targets in parallel via `cmake --build -j N`. The target list is the **single source of truth**: the `KERNEL_MODULES` registry in `csrc/CMakeLists.txt`, one entry per module as `name|srcs...` (a module may span several TUs — gemm is split into per-dtype-pair instantiation units so the heavy template work parallelizes). Entries append conditionally: the four base modules always, `quantize` and `gemm` only when the arch list reaches sm_89+, and nothing at all when `ASTRAI_CUDA_ARCH` is unset. `astrai/extension/runtime/loader.py` auto-discovers the compiled `.so` files, so adding a kernel means one registry entry and nothing else.

## Python Extension Architecture

The Python extension package separates low-level kernel bindings from execution
policy:

```text
astrai/extension/
├── __init__.py             # Public API and legacy module aliases
├── runtime/                # Registry/dispatch and lazy module loading
│   ├── dispatch.py
│   └── loader.py
├── backend/                # Operator strategy and fallback policy
│   ├── attention/          # Registry plus CUDA, FlashAttention, torch backends
│   └── rotary.py
├── kernel/                 # Stateless compiled bindings and INT8 primitives
│   ├── attention.py
│   ├── cross_entropy.py
│   ├── gdn.py
│   ├── gemm.py
│   ├── quantize.py
│   └── rotary.py
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
bindings. Legacy module paths remain aliases of the grouped modules.

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

An operator family supplies call axes and a safe fallback. External packages
can add implementations with `register_impl(ImplRecord(...))`; attention
backends use `@AttentionBackendFactory.register("name")`. Each record declares
`priority`, `modes` (`train`/`infer`), `available`, and a per-call capability.
The family supplies `mode` in its axes. Explicit and `with op_backend(...)`
selections raise when incapable; `ASTR_OPS` and `set_op(...)` fall through to
the next capable implementation. Inference-only implementations cannot run
on a training call, including when selected explicitly. Call
`unregister_impl(family, name)` when unloading an external operator.

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
it needs a backend type or policy implementation, and `astrai.extension.kernel`
only when it deliberately requires one exact kernel.

### Placement Rules

When extending this package:

| Change | Location |
|--------|----------|
| Add a pybind call for a compiled kernel | `astrai/extension/kernel/` |
| Add argument translation required by the compiled ABI | `astrai/extension/kernel/` |
| Add an implementation to an existing operator family | `register_impl(ImplRecord(...))` |
| Add a new operator family | `register_family(...)` and its family entry point |
| Add an attention backend | `astrai/extension/backend/attention/` and `AttentionBackendFactory.register(...)` |
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
├── CMakeLists.txt                    # CMake build: the KERNEL_MODULES registry (module name | its source TUs) + torch/pybind11 linking
├── __init__.py                       # build-time marker only (keeps `csrc` a setuptools package)
├── include/                          # THE include root: every header, angle-bracket root-qualified (<kernel/gemm/kernel.cuh>)
│   ├── policy.cuh                    #   composed GemmPolicy consumed by the kernel
│   ├── policy/                      #   compile-time GEMM tile rules
│   │   ├── traits.cuh              #     promoted MMA traits and shared-memory budget
│   │   └── manifest.cuh            #     named tile recipes, CTA classes and staging ladders
│   ├── scheduler.cuh                 #   cross-cutting: grouped/plain raster mapping
│   ├── kernel/                       # device kernels and launch helpers, grouped by operator
│   │   ├── attention/
│   │   │   ├── mma.cuh               # attention fragments and softmax helpers
│   │   │   ├── split_kv.cuh          # decode and split-combine kernels
│   │   │   ├── split_q.cuh           # prefill kernel
│   │   │   └── launch.cuh            # attention launch and dispatch
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
│   ├── api/                          # THE CALLER CONTRACT — declarations, the supported-set lists, the capability gates and the cross-layer family PODs; the only directory whose headers may touch torch/ATen/c10/Python (files named BY FAMILY: <family>*.h = that family's surface)
│   │   ├── dtype.h                   #     element type -> PyTorch ScalarType
│   │   ├── gemm.h                    #     gemm C++ surface (declarations only, no py:: type)
│   │   ├── attention.h               #     attention entry declarations (astrai::attention)
│   │   ├── attention_dtypes.h        #     attention ASTRAI_ATTN_DTYPE_LIST + generated unsupported-dtype refusal
│   │   ├── gated_deltanet.h          #     the family's two entry declarations (astrai::gdn)
│   │   ├── quantize.h                #     quantize declaration surface: QuantizeOutputs + run_quantize (the implementation is quantize/entry.cu)
│   │   ├── fp8_checks.h              #     fp8 capability gate (check_fp8_device; native FP8 GEMM)
│   │   ├── gemm_common.h             #     layout tags, ElemTrait, gemm_mma_traits, GemmParams POD
│   │   ├── attention_common.h        #     AttentionParams POD (cross-layer: stage headers include it)
│   │   └── quantize_common.h         #     sm_at_least + kMinSmForFp8, QuantLayout, RingLayout, QuantParams POD
│   └── launcher/                     # GEMM CUDA launch and dispatch templates
│       ├── gemm_launch.cuh           #     typed CUDA launch and TMA descriptor setup
│       ├── gemm_tiles.cuh            #     manifest selection and staging resolution
│       ├── gemm_dispatch.cuh         #     layout rewrite, typed entry and planner probe
│       └── plan_types.h              #     planner query and launch decision
├── attention/                        # family translation units + one bindings.cu (→ module attention; kernels/launchers/dispatch in the shared kernel/ headers)
│   ├── entry.h                       #   attention torch→POD marshalling (pack_*_params, split-partial allocation) — TU-local impl header, quoted-include (fp8_state.h shape)
│   ├── decode.cu                     #   → module attn_decode
│   ├── prefill.cu                    #   → module attn_prefill
│   ├── paged_decode.cu               #   → module attn_paged_decode
│   └── paged_prefill.cu              #   → module attn_paged_prefill
├── gemm/                             # family translation units and private headers (→ module gemm)
│   ├── gemm.cu                       #   typed host layer: dtype-pair registry + its two lookups + the planner's C++ face
│   ├── entry.h                       #   quant_gemm's op-entry ladder (dtype classify → device gate → scale contract → layout/shape validation → GemmParams pack → dispatch) + the empty-problem guard — TU-local impl header, quoted-include, included at the bottom of gemm.cu (the lookups it calls live there)
│   ├── bindings.cu                   #   quant_gemm/planner pybind surface, PYBIND11_MODULE and FP8 registration
│   ├── fp8_linear.cu                 #   fp8 training forward/backward as one C++ autograd::Function
│   ├── fp8_linear.h                  #   internal declaration boundary shared with fp8_runtime.cu
│   ├── fp8_runtime.cu                 #   one State instance, checkpoint/debug interface and pybind registration
│   ├── fp8_ring.h                    #   scale recipe and delayed-scaling ring
│   ├── fp8_cache.h                   #   versioned weight cast and bounded activation cache
│   ├── fp8_state.h                   #   meta registry, slots and restore helpers used by both FP8 translation units
│   ├── planning.cpp                 #   recipe enumeration, analytical model and raster selection
│   ├── plan_table.cpp               #   host row parsing, storage, config and row selection
│   ├── plan_table.h                 #   private row contract shared by the host planner TUs
│   ├── gemm_*.cu                    #   per-pair explicit instantiations, compiled per schedule
│   └── plan_table_builtin.cpp      #   generated measured and degraded rows
├── symmetric/                        # BF16 SYRK/SYMM (→ module symmetric)
│   ├── bindings.cu                   #   pybind surface
│   ├── entry.cu                      #   tensor validation and GEMM parameter packing
│   ├── entry.h                       #   private host/launcher declarations
│   └── kernels.cu                    #   typed CUDA kernels, launch dispatch and resource planning
├── quantize/                         # family translation units (→ module quantize; entry.cu also compiled into gemm to share the chain)
│   ├── bindings.cu                   #   pybind surface only (quantize / quantize_dual)
│   └── entry.cu                      #   the entry implementation: run_quantize + ring binding + dtype dispatch (ASTRAI_QUANT_IN_DTYPES lives here)
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
    └── quant_gemm_test.cu            # GEMM correctness + TFLOPS bench (links gemm/planning.cpp and plan_table.cpp)
```

GEMM benchmark entry points:

```bash
.venv/bin/python csrc/bench/benchmark_gemm.py dtypes --help
.venv/bin/python csrc/bench/benchmark_gemm.py layouts --help
.venv/bin/python csrc/bench/tune_plan_table.py diff --help
```

Compiled `.so` modules are placed directly in `astrai/extension/`.

Three conventions the tree encodes. (1) **Stages, not families**: headers
live under `csrc/include/<stage>/` by what they do (kernel / memory / mma /
epilogue / arith / datatype / utils); `kernel/` groups its headers by
operator, while the `csrc/<operator>/` directories hold translation units, and `api/` is the caller contract — declarations, the
supported-set lists and the capability gates; it is the only directory
whose headers may touch torch, which is what keeps the standalone
`csrc/tests/*.cu` harnesses torch-free. The gemm→quantize include edges
that forced a family-qualified tree before (`mainloop → dequant`,
`fp8_linear → quantize launch`) are plain kernel→datatype and TU→api
edges now. (2) The standalone harnesses stay out of the CMake registry on
purpose: each carries its own `nvcc` line so a correctness test runs
without torch; the `bench/` python tools are the reproduce path for the
numbers in the operator docs and workspace `notes/`. (3) **One include
root, `csrc/include`, every project include angle-bracket and
root-qualified** (`<kernel/gemm/kernel.cuh>`, `<policy.cuh>`); quoted includes are
the toolchain's plus the harness-local `test_utils.cuh`. A quoted project
path would resolve through the includer's own directory first and silently
change meaning on a move — the three same-named `common.h` are now
`api/{gemm,attention,quantize}_common.h` precisely so a spelling names
one file. (The layout test that once pinned this mechanically was retired
when the stage tree landed; the discipline lives here and in review.)

A fourth convention, the mirror of the header fan-in rule: **implementation
code lives in the TU territory of its consumers**. An implementation header
whose every includer sits in one family directory lives beside them
(`attention/entry.h`, `gemm/fp8_state.h` — quoted same-directory include);
an implementation shared across modules becomes a .cu listed in each
module's CMake sources (`quantize/entry.cu`, compiled into both the
quantize and gemm modules), with only its declaration in `api/`.
`launcher/` owns GEMM CUDA dispatch. `plan_types.h` carries the shared query
and launch decision. `gemm/planning.cpp` ranks recipes, while `gemm/plan_table.cpp`
owns mutable row state and parsing; generated rows live in `gemm/plan_table_builtin.cpp`.
The dtype-pair
instantiation units include neither planner nor table headers. A family
gains a directory when it gains a second file; single-file families
(`rotary_emb.cu`) stay at the top level.

Compiled `.so` modules are placed directly in `astrai/extension/`.

> Document Update Time: 2026-10-05
