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
| `attention` | `attention/attention.cu` | Dense attention |
| `attention_varlen` | `attention/varlen.cu` | Forward attention over contiguous packed segments |
| `attention_with_kvcache` | `attention/cache.cu` | Append K/V and attend over physical pages |
| `newton_schulz` | `newton_schulz/{bindings,entry,kernels}.cu` | Generic BF16 SYRK/SYMM with measured tile dispatch |
| `rotary_emb` | `rotary_emb.cu` | Fused rotary forward and input gradient |
| `swiglu` | `swiglu.cu` | Fused SwiGLU forward and both input gradients |
| `cross_entropy` | `cross_entropy.cu` | Fused cross-entropy and chunked linear cross-entropy |
| `quantize` | `quantize/bindings.cu` + `quantize/entry.cu` | Shared FP8/int8 quantization kernels (sm_80+) |
| `gemm` | `gemm/entry.cu` + `gemm/instantiations/*.cu` | Tensor-core GEMM: BF16/INT8 and mixed pairs on sm_80+, native FP8 on sm_89+ |

Additionally, optimized `.cuh` variants with tensor-core MMA (Matrix Multiply-Accumulate) exist:

| Variant | File | Optimization |
|---------|------|--------------|
| Split-KV MMA decode | `csrc/include/kernel/attention/mainloop.cuh` | Split KV across warps + MMA (sm_80+) |
| Split-Q MMA prefill | `csrc/include/kernel/attention/mainloop.cuh` | Split Q across warps + MMA (sm_80+) |

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
| Cross-entropy | [cross_entropy.md](cross_entropy.md) | `csrc/cross_entropy.cu` | `astrai/extension/kernel/cross_entropy.py`; `astrai/trainer/strategy/supervised.py` |

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

By default, kernels are built when **both** of these conditions are met:
1. `nvcc` is available on `PATH`
2. `torch.cuda.is_available()` returns `True`

`CSRC_KERNELS=false` disables compilation. `CSRC_KERNELS=true` forces the
build path; building without a visible GPU also requires an explicit
`ASTRAI_CUDA_ARCH` and a working CUDA toolchain.

### Manual build

```bash
# During install
CSRC_KERNELS=true pip install -e . --no-build-isolation

# Rebuild after editing .cu/.cuh files
CSRC_KERNELS=true python setup.py build_ext --inplace
# Output: astrai/extension/_C_*.so

# Or invoke CMake directly (ASTRAI_CUDA_ARCH is required here too —
# without an arch token no kernel targets are created; the CUDA toolchain
# is still required to configure the project)
cmake -S csrc -B build/cmake \
  -DASTRAI_CUDA_ARCH=120a \
  -DTORCH_HOME=<site-packages>/torch \
  -DPYTHON_INCLUDE_DIR=<python include> \
  -DPY_SOABI=cpython-312-x86_64-linux-gnu
cmake --build build/cmake -j 16
```

### Architecture flags

`setup.py` passes `ASTRAI_CUDA_ARCH` to CMake, for example `"80;89;120a"`.
Tokens contain compute-capability digits and an optional architecture-specific
`a` or family-specific `f` suffix. `csrc/cmake/arch.cmake` probes each exact
target with NVCC, rejects targets below SM80, and emits explicit SASS and PTX
images for each selected token. Feature-specific targets do not implicitly
add a baseline image.

There is **no CMake architecture default**. An empty list creates no kernel
targets, but direct CMake configuration still requires the CUDA toolchain.
When the environment variable is unset, `setup.py` detects the local GPU;
CC 12.0 selects `120a`. The normal installer skips compilation when CUDA is
unavailable, and the Python loader can run without compiled modules.

- **sm_80+** (Ampere and later): the minimum for the attention family — the
  kernels use `mma.sync.m16n8k16.bf16` / `.f16` for BF16 / FP16 attention;
  there is no scalar fallback.
- **sm_80+**: required for shared quantization, BF16/INT8 GEMM and mixed
  BF16/INT8 GEMM. FP8 conversion does not require FP8 tensor-core MMA.
- **sm_89+**: required for native FP8 GEMM MMA, whose architecture-specific
  instantiations are selected independently of the quantization entry.
- **sm_90+**: enables TMA schedule builds.
- **sm_120a / sm_120f**: enables the MX FP8 `block_scale` schedule. Plain
  `120` does not build MX. Runtime dispatch intersects device capabilities
  with the compiled-target manifest; a device capability alone cannot
  enable an absent schedule.

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

`KERNEL_TARGETS` in `csrc/CMakeLists.txt` lists object libraries and pybind11
modules as `name|OBJECT/MODULE|sources...`. A module can contain several
translation units: all four attention entry points share the `attention`
module, while GEMM splits dtype-pair instantiations for parallel compilation.
Both `quantize` and `gemm` link the shared `quantize_entry` object library.

Compiled `_C_*` modules live directly in `astrai/extension/` (`.so` on Linux,
`.pyd` on Windows). `runtime/loader.py` discovers them and imports each module
lazily. A new native module needs a build entry and bindings; its Python
wrapper and public API must also be added where applicable.

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
└── _C_*                    # Compiled .so or .pyd modules
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

Native attention tests can call the stateless compiled-kernel adapter:

```python
from astrai.extension.kernel.attention import attention

output = attention(q, k, v, mask=mask, is_causal=True)
```

Model code uses the three public attention functions described in [Attention](attention.md).

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

`EngineCore` snapshots the selection at construction and installs it when
constructing and running the worker. Requests do not carry an implementation
choice.

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

```text
csrc/
├── CMakeLists.txt           # Module/object targets and linking
├── cmake/arch.cmake         # Exact target validation and build manifest
├── include/                # Shared, root-qualified headers
│   ├── api/native/         # Framework-free caller contracts
│   ├── api/torch/          # Tensor interfaces and dtype dispatch
│   ├── policy/             # Tile geometry, recipes and configuration
│   ├── launcher/           # GEMM and quantization launch templates
│   ├── kernel/             # Device kernels, grouped by operator
│   ├── memory/             # Operand addressing and staged loads
│   ├── mma/                # Tensor-core instructions and fragments
│   ├── epilogue/           # Output transforms and stores
│   └── arith/, datatype/, utils/  # Shared primitives
├── attention/              # One module: bindings and four entry TUs
├── gemm/                   # Bindings, entry and dtype instantiations
│   ├── plan/               # Host planner and measured rows
│   └── fp8/                # Training composition and persistent state
├── newton_schulz/          # SYRK/SYMM bindings, entry and kernels
├── quantize/               # Bindings plus a shared entry object
├── gated_deltanet/         # Chunked GDN kernels and bindings
├── rotary_emb.cu, swiglu.cu, cross_entropy.cu
├── bench/                  # Measurement and dispatch analysis
└── tests/                  # Standalone native harnesses
```

GEMM benchmark entry points:

```bash
.venv/bin/python csrc/bench/benchmark_gemm.py dtypes --help
.venv/bin/python csrc/bench/benchmark_gemm.py layouts --help
.venv/bin/python csrc/bench/tune_plan_table.py diff --help
```

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
.venv/bin/python -m pytest tests/extension/test_torch_dispatch.py -q
```

Check the dependency boundary and compile the host planners independently:

```bash
.venv/bin/python -m pytest tests/extension/test_csrc_api.py -q
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
