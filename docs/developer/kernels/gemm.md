# GEMM / Linear Kernel

> Part of the [operator docs](README.md); required reading before changing
> the tile vocabulary, the planners, or the fp8 recipes.

AstrAI implements its own CUDA/PTX primitives and templates; CUTLASS is a
design reference, with no build or runtime dependency. BF16 and INT8 GEMM
start at SM80; native FP8 MMA starts at SM89.

## Architecture and dispatch

| Compiled target | Available GEMM paths |
|---|---|
| SM80–SM88 | `mma.sync`, cp.async; BF16, INT8 and mixed BF16/INT8 |
| SM89 | Above plus FP8 e4m3/e5m2 MMA |
| SM90–SM110 | Above plus TMA staging with warp MMA |
| SM120 | Warp MMA and TMA; `120a` / `120f` additionally enable block-scaled FP8 MMA |

WGMMA and tcgen05/TMEM mainloops are not implemented. SM90/SM100 support
uses the shared warp MMA core; it does not imply native peak performance.

`ASTRAI_CUDA_ARCH="80;89;90;100;120a"` selects exact compiler targets.
CMake validates them with nvcc and generates the capability manifest.
Each schedule has distinct template types and object files; runtime dispatch
intersects the current device with the compiled images. Generic PTX permits
forward JIT, `a` targets require the exact architecture, and `f` targets stay
within their family. An SM80-only build therefore uses cp.async even on a
5090. `kernel.gemm.capabilities()` reports the compiled paths available on
the current device; `set_staging()` can disable TMA or MX independently.

The device policy composes element types, tile geometry, shared-memory
layout, schedule and epilogue. Instruction wrappers own the register
contract; `static_assert` enforces architecture and layout constraints.
Host planning is a fixed sequence of ordinary functions: override rows,
injected rows, measured builtin rows, then heuristic,
subject to the selected planner mode. No virtual planner hierarchy is needed.

## Decode row (`m == 1`)

A decode step presents one row per linear. Every tile pads M up to the CTA
height (64 or 128 rows), so at `m == 1` the mainloop issues 63/64 or 127/128
of its MMA work for rows that do not exist while still streaming the whole
weight: the tile route is bound by padding work, not by the weight stream
that actually limits decode. `launcher/gemv.cuh` routes that shape to
`kernel/gemm/gemv.cuh` — one warp per output row, 16B weight chunks per lane,
warp-shuffle reduction, epilogue factors folded in lane 0 — and leaves every
other shape on the manifest dispatch.

The route is a strict subset of the tile contract, gated at compile time by
`gemv_eligible_v` (a pair without a dot product instantiates no kernel):

- `m == 1` and `batch == 1` — a batched decode keeps the tile pipeline;
- `crosswise_of<LayoutA, LayoutB>() == 0`, i.e. both operands
  contract-contiguous, so a weight row is `b + n * b_ld` (the NT linear route);
- row-major output — the symmetric-NN swap lowering keeps the tile path;
- bf16/fp16 activation with a bf16/fp16/int8 weight and a bf16/fp32 output.

The epilogue is the tile epilogue's, on the one live row:
`y[n] = (sum_k a[k] * b[n,k]) * a_scale(0) * b_scale(n) + bias[n]`, per-tensor
scalars folded first, bias added in fp32, one rounding to the output dtype.
`quant_gemm_test.cu` runs `m == 1` through all four layouts for W16A16 and
W8A16 (NT takes the GEMV, the crosswise layouts keep the tile route); both
agree with the reference at the same tolerance as the `m == 256` cells.

Kernel-only A/B through the production dispatch, 200 launches per cell after
20 warmups, mean over the whole event window, RTX 5060 Ti (sm_120), CUDA 13.4:

| Shape (m×n×k) | dtype | tile route | GEMV route | speedup | GEMV bandwidth |
|---|---|---|---|---|---|
| 1×4096×4096 | W16A16 | 0.0675 ms | **0.0193 ms** | 3.5× | 1739 GB/s¹ |
| 1×4096×4096 | W8A16 | 0.0674 ms | **0.0147 ms** | 4.6× | 1145 GB/s¹ |
| 1×11008×4096 | W16A16 | 0.3427 ms | **0.2123 ms** | 1.6× | 425 GB/s |
| 1×11008×4096 | W8A16 | 0.3364 ms | **0.1080 ms** | 3.1× | 418 GB/s |
| 4×4096×4096 | W16A16 | 0.0632 ms | 0.0620 ms | 1.02× | — |
| 4×4096×4096 | W8A16 | 0.0674 ms | 0.0676 ms | 1.00× | — |

¹ The 4096-wide weight fits the 32MB L2, so that row measures L2 limits. The
11008-wide row is DRAM-bound and lands at the card's measured streaming read
ceiling (437 GB/s), i.e. the GEMV route has no headroom left to recover; the
tile route leaves 2-3x on the table by computing padding. `m == 4` confirms the
route is decode-only: same kernel path, same latency.

## Geometry heuristic (`geom_cta`)

`plan.configure(planner="heuristic")` uses the same `geom_cta` rule as the
Python replay script. Explicit heuristic mode skips measured rows; default
`hybrid` checks rows first, then uses geom_cta. Both read compiled kernel
metadata without timing kernels. The older `model` is explicit-only.

### Formula

For effective CTA tile `(BM,BN,KK)`, warp tile `(WM,WN)`, `P` threads,
input byte widths `a,b`, and `S` SMs:

```text
w = P/32
L = ceil(K/KK)
G = batch*ceil(M/BM)*ceil(N/BN)
R = min(CUDA resident-CTA limit, ceil(G/S))
W = ceil(G/(S*R))

B = KK*(BM*a + BN*b)/512                 # copy work
I = (KK/MMA_K)*(BM/16)*(BN/8)/w           # per-warp MMA work
H = KK*(WM*a + WN*b)/512                 # fragment loads
D = KK*(WM*[a<b] + WN*[b<a])/64          # mixed-width conversion
E = output_bytes*BM*BN/512
J = 2 if TMA else B

C_mem      = W*R*(L*B + E)
C_mma      = W*R*L*w*I
C_serial   = W*L*(I + H + D)
C_fragment = W*R*L*w*(H + D)
C_control  = W*R*L*(2*w + J)
eta = R if cp.async else 1
score = log(C_mem) - log(eta) + log(C_mma) + log(C_serial)
      + log(C_fragment) + log(C_control)    # smaller wins
```

`MMA_K` follows compiled instruction traits; `[condition]` is 0 or 1.
The launcher resolves warp widening and output reclaim before scoring.
The cp.async memory discount assumes resident CTAs help hide copy latency;
`R` is a count, not a utilization fraction or measured speedup.

### Why geometric averaging and log space

Conceptually, each cost is normalized by that arm's best candidate, then
the five relative costs are geometrically averaged. This combines relative
work without fitting a bytes-to-instruction-time conversion; changing an
arm's units does not change the ranking. Normalization subtracts a common
constant in log space, and the fifth root divides every score by five, so
both are omitted. Python negates the score because its harness maximizes.

Both implementations factor products into log sums, e.g.
`log(W*R*L*w*I)=log(W)+log(R)+log(L)+log(w)+log(I)`. The memory sum uses:

```text
x = log(L) + log(B); y = log(E)
log(L*B + E) = max(x,y) + log1p(exp(-abs(x-y)))
```

No large cost products or `exp(score)` are formed; the one exponential has
a nonpositive argument. C++ uses `double`, Python uses double-precision
`float`. C++ avoids `n+d-1` overflow in heuristic ceil division and casts
grid axes before multiplication. M/N/K must be positive; nonfinite
scores are excluded. Rounding and near ties remain possible.
This protection applies to this score, not the older integer `model` or
every launch calculation. Logs run on the host; decisions are cached.

**Limits and validation.** These overlapping work proxies can trade one
advantage against another bottleneck, so their geometric mean is not a
latency prediction. Cache reuse, spills, scalar tails and crosswise loads
are approximate; a later TMA encoding fallback is not rescored. On 36
RTX 5090 same-process ABBA points (BF16/W8A16/W8A8, both paths, MX off),
geom_cta had -0.34% aggregate latency and +20.1% worst-point latency versus
`model`. Other GPUs are unvalidated. The log port passed 112 plan tests,
69 C++/Python selection checks and 108 historical candidate selections;
a Python-only synthetic integer-limit check kept 12 candidate scores finite.

In heuristic mode, `plan.probe(...).resources` exposes rows containing
`(cta,k_stages,k_tile)`, effective `(bm,bn,k_tile,wm,wn,threads)`, resident CTAs,
registers/thread and local bytes/thread. Metadata is cached per typed kernel
and device. Local allocation is not a dynamic spill count. Probe assumes
contiguous aligned inputs; actual views can choose a different staging path.

The replay script keeps `geom_cta`, `geom_barrier`, `ncu_spill` and the
`model_exact` baseline. `ncu_spill` uses a local-footprint risk estimate,
not NCU timings. Only geom_cta and model_exact have C++ counterparts:

```bash
python csrc/bench/model_capture.py results.json --staging cpasync
python csrc/bench/model_capture.py results.json --rule geom_cta --check
python csrc/bench/model_capture.py results.json --rule model_exact --check
```

## Source layout

Headers under `csrc/include` contain declarations and device templates;
`csrc/gemm` contains host implementations, bindings and explicit instantiations.

| File | Role |
|------|------|
| `api/native/quantize.h` | `QuantParams` and shared `RingLayout` slot offsets; no framework or CUDA dependency |
| `datatype/element.cuh` | shared CUDA element traits: storage and native pair conversion |
| `api/torch/dtype.h` | CUDA-free Torch dtype names, checkpoint-name matching, supported-type metadata and FP8/int8 ranges |
| `api/torch/dispatch.cuh` | CUDA element/ScalarType mapping, type tags and shared scalar/pair type-list visitors |
| `kernel/quantize/kernel.cuh` | CUDA cast, transpose and optional fused history update; derives ring pointers from its base and shared slot offsets |
| `launcher/quantize.cuh` | Native quantize grid/block configuration, launch submission and error checking |
| `datatype/dequant.cuh` | in-register dequantization functors (`DequantPair<SrcT, MmaT>`): the exact int8→bf16 expansion quantized-GEMM operands fold between the smem read and the mma |
| `api/native/gemm.h` | GEMM layout tags and unchanged `GemmParams` POD; no Torch or CUDA instruction dependency |
| `policy/gemm/traits.cuh` | Compute promotion (`gemm_mma_traits`), promoted MMA traits and shared-memory policy (`GemmTraits`, `GemmSmem`) |
| `policy/gemm/geometry.h` | Pure C++ geometry, layout tests and ring-budget formulas, shared by host planning and CUDA policy |
| `policy/gemm/manifest.h` | One native tile vocabulary and staging manifest, with no Torch or CUDA instruction dependency |
| `policy/gemm.cuh` | `GemmPolicy`: the kernel's composed dtype, layout, tile, staging, and output policy |
| `policy/gemm/config.h` | Internal runtime configuration owner and helpers; imports the native planner contract |
| `memory/load_async.cuh` / `load_crosswise.cuh` / `load_crosswise_packed.cuh` | Operand staging by access pattern: cp.async (congruous and 16-bit transposed) with `PrefetchCarry`; direct 8-bit crosswise LDG+PRMT with `CrosswiseCarry`; packed k-pair crosswise with `PairPackCarry` |
| `scheduler.cuh` | CTA id → (block_m, block_n) grouped/plain raster (runtime `raster` knob) |
| `kernel/gemm/mainloop.cuh` | `GemmCollectiveMainloop`: stage rings, stage loads, fragment addressing (ldmatrix + dequantized scalar paths), pipelined mma.sync loop |
| `epilogue/writer.cuh` | `GemmCollectiveEpilogue`: fused bias + per-row/per-channel scale folding + bf16/fp32 smem scatter + coalesced copy-out |
| `kernel/gemm/kernel.cuh` | Device entry kernels for cp.async and TMA staging; they compose the mainloop and epilogue |
| `kernel/gemm/gemv.cuh` | Single-row (decode) row GEMV: one warp per output row, 16B weight chunks, scale/bias folded in lane 0 |
| `launcher/gemv.cuh` | `m == 1` capability gate and launch; the tile dispatch calls it first and keeps its path when it declines |
| `launcher/gemm_launch.cuh` | Typed CUDA launch and TMA descriptor setup |
| `launcher/gemm_tiles.cuh` | Manifest tile selection, output-reclaim fallback, and launch from the resolved staging decision |
| `launcher/gemm_dispatch.cuh` | Heavy definitions: typed query construction, layout canonicalization and routing shared by launch and probe |
| `api/native/gemm_plan.h` | Pure C++ query/decision, staging, probe results, capabilities, configuration and vocabulary interfaces |
| `gemm/plan/table.h` / `gemm/plan/table.cpp` | Private host row contract, parsing, row sources, configuration and row selection |
| `gemm/plan/builtin.cpp` | Generated measured rows, compiled in a separate host TU |
| `gemm/plan/planner.cpp` | Recipe vocabulary, model ranking and raster selection |
| `api/torch/gemm.h` | Tensor-based `quant_gemm_impl` and `ScalarType`-based `plan_probe`; imports native result types and contains no Python interfaces or kernel instantiations |
| `gemm/entry.cu` | Typed host dispatch using the launcher header: one dtype-pair visitor feeds launch and probe; the schedule visitor selects the compiled architecture variant. Holds no `py::` type |
| `gemm/fp8/linear.cu` | Composed fp8 training linear (forward and backward) in one C++ `autograd::Function` |
| `gemm/fp8/runtime.cu` / `gemm/fp8/linear.h` | Single `State` owner, checkpoint/debug and pybind interface; internal declarations connecting the runtime to the autograd entry |
| `gemm/fp8/ring.h` / `gemm/fp8/cache.h` / `gemm/fp8/state.h` | Scale-ring lifecycle; reusable weight/activation casts; typed metadata registry and checkpoint snapshots |
| `gemm/bindings.cu` | Quantized GEMM and planner pybind surface: None-tolerant argument marshalling, state/config dict contracts and `PYBIND11_MODULE`; calls `bind_fp8` from `gemm/fp8/runtime.cu` |
| `csrc/quantize/bindings.cu` | pybind surface only → module `quantize` |
| `csrc/quantize/entry.cu` | the entry implementation: quantize / quantize_scaled validation, allocation and dtype dispatch (compiled once as the `quantize_entry` object target and linked into both modules) |

`api/torch/` may include `api/native/`; native contracts and shared device
headers must never include the Torch adapters. The planner implementations
compile and link with ordinary C++17 without Torch or CUDA runtime libraries.
`DeviceFacts` is declared in `api/native/device.h`, while `utils/device.cuh`
queries CUDA. `GemmConfigPatch` uses `std::optional` (the same underlying type
as the supported Torch version's `c10::optional`); configuration semantics and
ownership remain unchanged. FP8 training still shares the one GEMM module and
its planner/state instances.

## Scales

`quantize` takes the current quantization *multiplier*; the
strategy layer passes `scale.reciprocal()` and the kernel multiplies by it.
`quant_gemm` takes per-operand dequant scales (`a_scale`, `b_scale`;
the fp8 training path passes `sa` / `sb` separately). `amax` is produced
only by the delayed-scaling ring fold (the in-kernel fused reduction) or
measured by the caller — the plain no-ring quantize runs a pure scale+cast
and returns `amax=None`.

Both FP8 and int8 use `quantize`; there is no separate int8 helper. The
multiplier is float32 on the input device and broadcasts over its logical
`[rows, cols]` view: one element is scalar, `[cols]` or `[1, cols]` is per
column, `[rows, 1]` is per row, and `[rows, cols]` is per element. Shape
declares these axes and Tensor strides address storage, including expanded
or non-contiguous views. `quantize_scaled` requires a scalar multiplier
and publishes a global next scale using `quant_max`; its default range is 448, 57344 or 127
for E4M3FN, E5M2 or int8, selecting `dtype` before `tdtype`.

GEMM scales are contiguous CUDA float32 on the operand device. `a_scale`
has one element or M elements, and `b_scale` has one or N. Each int8 operand
requires its corresponding scale; FP8 scales are optional and BF16 operands
do not accept scales. These factors run along output dimensions, never K.
For W8A8 linear with x `[M, K]` and w `[N, K]`, caller-computed row scales
map directly to `a_scale` and `b_scale` with `trans_b=True`. Pass their
reciprocals to quantize with shapes `[M, 1]` and `[N, 1]`; keep the GEMM
scale tensors flat. A multiplier stays tied to its original input element
when a transposed copy is produced. Quantize also accepts column and matrix
multipliers, while GEMM's scale inputs still vary only along M/N; scales
varying along K cannot be folded into its epilogue.

Python layer (two levels): `astrai/extension/kernel/quantize.py` and
`kernel/gemm.py` are the stateless kernel adapters (`quantize` /
`quantize_scaled` / `quant_gemm` wrappers, one adapter file per compiled kernel
module). `astrai/extension/policy/quantization/autocast.py` owns the FP8
strategy: recipes, history / dynamic scaling and `fp8_autocast`. The same quantize
adapter executes FP8 and int8 casts with caller-provided multipliers.
The composed fp8 linear —
quantize, ring fold/advance, the weight-cast cache and all three GEMMs — lives in C++
(`csrc/gemm/fp8/linear.cu`, compiled into the `gemm` module
where the GEMM dispatch state lives) behind a single Python->C++ crossing;
the strategy layer routes `aten::linear` to it on CUDA. The aten override
installs lazily on the first fp8 activation (autocast enter or the global
enable), so importing the module is dispatcher-neutral.

The shared quantization interfaces and scale-publication math are described
in [quantize.md](quantize.md). Their execution layer depends only on
`QuantParams`, `QuantizeOutputs` and `RingLayout`; cached-cast invalidation,
metadata registration and checkpoint restoration belong to FP8 linear.

## FP8 training composition

Master weights, gradients and optimizer state stay BF16/FP32. FP8 is used
for GEMM operands. Forward casts the original operands and, when backward
needs them, transposed copies in the same pass:

```text
out = bf16((x8 @ w8^T) * sx * sw + bias)
grad_x = bf16((g8 @ w8T) * sg * sw)
grad_w = bf16((g8T @ x8T) * sg * sx)
grad_b = bf16(sum_rows(g))
```

The GEMMs accumulate in float32 and fold dequantization scales into the
epilogue. Backward quantizes the gradient once for both GEMMs and reuses
the saved transposed forward casts. Hybrid recipes request E4M3 original
outputs and E5M2 transposes explicitly.

Dequantization scales vary only along the GEMM output dimension: `a_scale`
has one or M values and `b_scale` one or N. The training path uses per-tensor
scales for every operand because the weight's output dimension changes in
backward; per-channel weight scaling is a forward-only choice.

Dynamic scaling measures the current input before calling `quantize`.
History scaling calls `quantize_scaled` with the current reciprocal and
the other publication pair. `ScaleRing` owns seeding, the history cursor
and double buffering; linear controls their forward/backward timing.
Versioned weight casts retain a scale snapshot and become invalid after an
in-place weight update. Shared activation casts retain their associated
ring. Cache hits skip both quantization and ring advancement, and no-grad
calls do not update history. These caches and registry snapshots are not
dependencies of either quantization entry.

The runtime owns registration, reset and restoration. Python checkpoint
dictionaries are converted to typed snapshots at its boundary; checkpoint
v2, legacy defaults and named/FIFO matching keep their existing behavior.

## Entry and parameter ownership

The public Tensor signature stays in `api/torch/gemm.h`; the kernel POD stays
in `api/native/gemm.h`. Private packing types never enter the native API.

| Location | Responsibility |
|---|---|
| `gemm/bindings.cu` | Python arguments and planner dictionaries |
| `gemm/entry.cu` | Device guard, metadata validation, operand/output ownership, dtype/schedule selection and probe |
| `gemm/plan/` | Ordinary C++ planner, row table and generated builtin rows |
| `gemm/fp8/` | Training autograd, scale rings, cast caches and runtime state |
| `gemm/instantiations/` | One heavy CUDA compilation unit per dtype pair; CMake compiles each schedule |
| `policy/gemm/` | Tile manifest, traits, geometry, cost model and shared runtime switches |
| `launcher/gemm_*.cuh` | Native layout lowering, recipe selection and CUDA launch |

The Tensor function arguments retain the original operands, scales and bias.
`resolve_operand` returns a `GemmOperand` with its Tensor storage, leading
dimension, batch stride and effective transpose flag, with no output-reference
arguments. `PreparedGemm` keeps both resolved operands and the output alive
alongside `GemmParams` until the asynchronous launch has been submitted.
The CUDA device guard encloses packing, allocation and launch. Problem extents
and leading strides are checked before narrowing to native `int`; batch
must be in `[1, 65535]` for both launch and probe; empty operand batches
are rejected before broadcast resolution. Shape/scale/bias errors
are rejected before contiguous fallback copies and output allocation.
Grid ceil division uses quotient and remainder to avoid signed overflow.

`select_gemm_variant` selects the dtype pair and compiled schedule once,
binding launch and probe functions to the same `DeviceFacts` snapshot. The
native probe accepts `GemmParams`, using the launch's layout-lowering route;
the public probe fills a contiguous hypothetical problem before calling it.
The native launch passes the POD by const reference until layout lowering or
device launch needs a copy. Kernel parameters and layout remain unchanged.

FP8 forward passes its recipe parameters explicitly to the composed
implementation. Torch dtype names, checkpoint-name matching and
`quant_max_of` are shared in CUDA-free `api/torch/dtype.h`. Bindings and
`ScaleRing` include that metadata without depending on CUDA dispatch.
FP8 state owns its registry and restore logic.
CUDA element mappings and type-list visitors live in
`api/torch/dispatch.cuh`, shared by GEMM, attention and quantize. Each family
declares its supported list locally, for example:

```cpp
using AttentionTypes = TypeList<bf16, fp16>;
using GemmBasePairs = TypeList<DTypePair<bf16, bf16>,
                               DTypePair<bf16, int8_t>,
                               DTypePair<int8_t, int8_t>>;
```

GEMM concatenates its base pairs with the FP8 list enabled by the compiled
capability manifest. `visit_dtype_pair<GemmPairs>` selects a legal pair;
names and support queries use that same list. The schedule callback retains
the device/build gates and existing MX, TMA, cp.async priority. Quantize uses
`visit_dtype<QuantInputTypes>`; attention uses `visit_dtype<AttentionTypes>`.
Callback returns may be void, values or references, with one common return
type across all candidates; duplicate keys and empty visitors fail compilation.

The Torch entry includes `launcher/gemm_dispatch.cuh` for native launch
and probe, without local forward declarations. Definitions and template
defaults live in that header. When the extension's `gemm_build.h` manifest
is present, the launcher declares external instantiations for the compiled
pairs and schedules. Their definitions remain in the per-pair CUDA units,
so the entry does not regenerate kernels or broaden architecture images.
Standalone native callers instantiate directly from the same header.
The shared build macro spells each explicit-instantiation signature once;
dtype support and runtime visitors use templates.

The native launch is a single `gemm_dispatch` template: its optional
`const DeviceFacts*` defaults to null and resolves device facts inside the
function when omitted. The Torch entry passes its selected device snapshot.
`plan_query` and `plan_dispatch_for` use the same optional device pointer;
the query resolves an omitted snapshot once into `PlanQuery::dev`.
`launch_policy_tma` has one implementation. Its shared-memory budget
defaults to -1; the function queries the device limit only when the
budget was not supplied.

Layout resolution: the user flag names the math (0 = last two dims are
`[rows][contract]`, 1 = transposed); a col-major *view* of a contiguous
buffer folds into the returned dispatch flag at zero copy — the kernel's
`LayoutA`/`LayoutB` tags cover both storages, and m/n/k derive from the
user flag only.

Geometry requires **M > 0, N > 0, K > 0**, including batched calls.
The operator entry and planner probe reject zero dimensions before dispatch;
probe also rejects negative dimensions. There is no empty-output or empty-sum
special case. Tests cover every planner mode, batches, and bias.

## Device design notes

**Staging layouts and swizzle.** Staging layouts are CUTLASS-style *types*:
`utils/swizzle.cuh` provides the `Swizzle<Bits, Shift>` /
`Layout<Shape<Rows, Chunks>, Stride<Chunks, 1>>` /
`composition(Swizzle, Layout)` vocabulary (16B-chunk units,
dtype-independent `(Bits, Shift)` pairs), and each collective declares its
tile's layout once — `SmemLayoutA/B` and the trans mirrors in the mainloop,
`OutLayout` in the epilogue. The family instances: congruous 2B staging is
the TMA 128B mode `<3,3>`, 1B is `<2,3>`, 16-bit trans staging swizzles
chunks by the k-row bits (custom `<log2(min(chunks,8)), log2(chunks)>`),
and the epilogue output keeps its row-width mode. The tensor layer
dispatches to `ComposedLayout::operator()` (the closed two-coordinate
form) — the 16B chunk index XORed with row bits at
`[3, 3+log2(kChunks))` — so a warp's ldmatrix fragment load (8 consecutive
rows × 16B) hits all 32 banks exactly once (the unswizzled row word-stride
is `kK/4` words, so rows `r` and `r + 8/kChunks` collide mod 32). Chunks
stay contiguous, so cp.async staging is unaffected. The helper derives the
XOR term from the row coordinate alone (the layout's `kRowShift`/`kMask`)
rather than the linearized index — keep the two-coordinate form in the hot
paths.

**Tensor vocabulary** (`utils/tensor.cuh`, cute's `Tensor<Engine, Layout>`).
ONE tensor type — storage and addressing are its two template parameters,
and every operation dispatches to a layout op; use sites spell
`Tensor<...>` directly, with no second names. Engines: `PtrEngine<T>`
(smem/gmem) and `ArrayEngine<T, N>` (the mma fragment cells; `MmaOp` names
them `AFrag`/`BFrag`/`CFrag`, and the typed `fma`/`ldmatrix` overloads take
them by reference so the registers stay in place). Layouts: the
`ComposedLayout` instances of `utils/swizzle.cuh` (16B chunk grids,
dtype-blind; `chunk_of` is the swizzled-chunk op, and the tensor scales the
row/chunk terms separately in 32-bit), `RingLayout` (slot rotation over a
per-stage grid) and `CellLayout` (element-unit (m, n) grid). A staged tile
is `Tensor<PtrEngine<Elem>, ComposedLayout>`; the stage ring adds the slot
dimension (`make_ring` / `stage_of`); the warp's accumulator is
`Tensor<ArrayEngine<CFrag>, CellLayout>` — `*acc(mt, nt)` at the fma seam,
the kPairB x4 fold is `BFragPair::cell`. Every method folds away at -O3.

**Boundary predication.** Predicated staging rides cp.async's runtime
source size (CUTLASS 2.x's zfill iterators): `cp.async.cg [dst], [src],
16, src_size` with `src_size` derived per chunk from the remaining
contract extent — 0 reads nothing and the hardware zero-fills the 16B
chunk, a partial size covers the k tail, and only a misaligned base
(non-16B `ld`) keeps the scalar-copy fallback.

**Fragment addressing (base-pair scheme).** One base register per operand
per k_seg, every fragment offset an LDSM immediate: for A,
`addr(s, mt) = lane_base + mt*(16*kK) ^ (s<<5)`; for B,
`addr(s, nt) = lane_base + nt*(8*kK) ^ (s<<5)`. The closure holds because
the XOR swizzle's source bits come only from the lane's row-within-matrix
`r7` — the 8/16-row fragment steps never reach them. Steady-state read
pointers advance one stage per iteration with an equality wrap instead of a
per-k-tile `(tile % ring) * stage_bytes` recomputation.

**Pipeline and barriers (cp.async).** Every operand ring holds
`kStages+1` buffers: the load for tile `i+kStages` targets slot
`(i-1)%(kStages+1)`, which compute(i-1) finished reading before this
iteration's barrier — no post-compute barrier, one `__syncthreads` per
k-tile. Prologue and tail commits are unconditional so the group sequence
stays tile-indexed and the fixed `wait_group<kStages-1>` is
iteration-invariant. The final iteration carries no trailing barrier of its
own, so the epilogue transition pays one explicit drain
(`PipelineSync<Stages>::drain()`, which empties only the calling thread's
groups) followed by `__syncthreads()` — without it a thread racing into the
epilogue scatters the output tile over peers' still-in-flight staging
writes and final fragment reads. The ring discipline runs through the
`PipelineSync` stage-pipeline type (`memory/pipeline.cuh`), with the raw
mbarrier PTX sites (init / arrive_expect_tx / wait_parity) shared with the
TMA path below.

**TMA staging** (sm_90+, dual-congruous operands, `memory/tma.cuh`). The
staging swizzles ARE the TMA hardware modes (`<3,3>` = SWIZZLE_128B for
2-byte elements, `<2,3>` = SWIZZLE_64B for 1-byte), so fragment addressing,
ring slots and the epilogue reclaim are untouched — only the load/wait
discipline changes. `TmaSwizzleOf<Staged>` derives the swizzle width and
the box's inner extent from the declared `ComposedLayout` (the encoder
decodes the `CUtensorMap` swizzle enum from that width), and
`tma_spec<Elem, Staged, BoxRows>` fills only the runtime geometry — the map
cannot drift from what the fragments read. Each operand's rank is a
template bit on `GemmTmaContext` / `gemm_kernel_tma` (strided batch = 3D
emitter, broadcast = shared 2D map), so the per-stage 2D/3D issue pick
compiles away; the launcher dispatches the four rank combinations. One
elected thread arms a per-slot mbarrier with `arrive.expect_tx` and issues
the boxes; OOB coordinates zero-fill, which absorbs the k tail and edge
tiles the cp.async zfill iterators predicated per chunk. Two TMA-specific
invariants: the swizzle applies to the ABSOLUTE shared address, so the ring
base rounds up to 1024B (budgeted in `Policy::kSmemBytes` together with the
barrier array); and the pipeline is the CUTLASS `PipelineTmaAsync`
handshake — `full[slot]` (count 1, expect_tx) for arrival, `empty[slot]`
(count = CTA threads, every consumer arrives after its last fragment read)
for release — which replaces the per-k-tile `__syncthreads`: warps skew
freely across slots and the producer's overwrite gate is the empty phase
alone. Descriptors are host-encoded (`cuTensorMapEncodeTiled` via dlsym —
no link-line changes) and cached exact-match, so steady-state calls pay the
few-microsecond encode once. The planner gates TMA on the device's compute
capability, dual-congruous layouts, 1-/2-byte dtypes and descriptor
encodability (misaligned base/ld falls back to the cp.async twin, which
stays compiled); `set_staging(tma=False)` forces the fallback.

**Stage depth.** The manifest carries s3 deep-ring siblings of every class
(thinner operand pairs leave more smem headroom under the budget). A row
names its ring depth directly and is smem-gated like any row; candidates
are measured as one-row tables (`set_table`, see
`csrc/bench/tune_plan_table.py sweep`).

**MMA cells and dtype promotion.** The same mainloop serves every dtype
pairing through one promotion rule (`gemm_mma_traits`): the mma runs on the
**MmaT** — symmetric fp8 keeps its native `m16n8k32`, symmetric bf16
(W16A16) passes through untouched, symmetric int8 (W8A8) keeps its native
`m16n8k32.s8.s8.s32` (int32 accumulators; `.satfinite` clamps the wrap
all-max-magnitude K≈16k inputs could reach), and a *lone* int8 (W8A16
weight-only or the mirrored A8W16) promotes to bf16 `m16n8k16` with
per-operand in-register dequant (`kDequantA`/`kDequantB` — W8A16 only B).
Staging never changes: int8 operands ride the existing congruous cp.async /
crosswise PRMT paths into the canonical swizzled tiles, and `kMmaK` follows
the mma cell (the `MmaShapeFor` trait) so the tile geometry is shared — the
native s8 pair reuses the fp8 k32 fragment layouts verbatim (1-byte dtypes
share the packed two-per-b16-slot layout, so ldmatrix addressing is
identical).

**Mma trait layer** (`mma/mma.cuh`). The instruction vocabulary assembles
from two specializable traits: `MmaShapeFor<Dtype>` maps an input dtype to
its instruction `Shape<M, N, K>` (primary template undefined — a dtype with
no MMA is a compile error; K follows the 256-bit A-fragment invariant: bf16
k16, 1-byte dtypes k32; `kMinArch` encodes each instruction's hardware
floor as a build-time assert), and `MmaOp<A, B, Shape>` — one
specialization per instantiated pairing cell carrying the accumulator type,
register counts and the dedicated asm block. `mma_sync<InT>` remains as the
fp32-family convenience view (attention + tests); the gemm mainloop calls
`Traits::MmaOp::fma` directly so the accumulator type rides the cell (fp32
for float families, s32 for s8). `Shape` itself lives in `utils/shape.cuh`.

**Dequant** (`datatype/dequant.cuh`). Each fragment register pair costs one
`LDS.16` + four LOP3-class instructions, exact for the full int8 range
including −128. bf16 carries only 7 mantissa bits, so the naive
"OR the byte into a bf16 base" trick (0x6400-style, exact for fp16) breaks
linearity — bit 7 spills into the exponent. Instead the magnitude bits
(0-6) and the sign bit (7) take separate LOP3s:
`h = (u & 0x7F) | 0x4300` → exactly 128+u7; `s = (u & 0x80) | 0x4300` →
128 or 256 as the sign picks; `v = h - s` is the exact int8 value, and
every intermediate is bf16-exact.

**MMA cell (sm_120a block_scale).** The plain warp-level fp8 mma
(`m16n8k32.e4m3/e5m2`) decodes at HALF rate on sm_120. The
`kind::mxf8f6f4` block_scale variant
(`mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X…ue8m0`) runs
at the full rate with an IDENTICAL A/B/C/D register contract, so the
mainloop/staging/epilogue layers are untouched: `MxMmaOp` (mma/mma.cuh)
swaps only the cell, carrying a constant unit scale (every ue8m0 scale byte
0x7f = 2^0, selectors inert — the scale-factored product IS the plain
product; quant_gemm_test output stays byte-identical). Warp-level
block_scale is an sm_120-FAMILY instruction (CUDA 13.0 ptxas: 120a/121a/
120f accepted, 100a/103a/110a rejected — datacenter Blackwell does MX
through tcgen05, which needs 13.1+), so the cell gates on the family pass
(`ASTRAI_ARCH_FAMILY >= 1200`, mma/mma.cuh's value-macro twin of
`ASTRAI_DEVICE_ARCH`) and the gemm fatbin carries the sm_120a image (the
arch token names it — `ASTRAI_CUDA_ARCH=120a`; CMake's native `a` grammar
emits one image per token, and a plain `120` list warns at configure time)
which the driver picks on sm_120; every other pass/device falls back to the
plain cell inside the same tree, so routing can only trade speed, never
correctness. `launch_plan` routes the symmetric-fp8 pair through the mx
tree on sm_120 unless `set_staging(mx=False)` knocks it out (the A/B knob;
read once per process — separate processes to compare).

**Crosswise loads.** Crosswise operands (A `[K][M]` / B `[N][K]` storage)
cannot cp.async into the canonical tile; they take the direct LDG.128×4 +
in-register PRMT transpose + STS.32 path. 8-bit crosswise operands with an
admissible tile take the k-pair packed grid instead: the operand staged as
16-bit (row, k-pair) units so the 16-bit reader's ldmatrix.trans contract
applies unchanged.

**Interior copy.** When both operands are congruous, the whole CTA is
interior, base|ld is 16B-aligned and K has no tail, the mainloop switches
to a predication-free copy with loop-carried prefetch state.

**NN swap.** The dual-N-contiguous problem runs as its transpose
`E = B^T @ A^T` over swapped operands with an out-transposed epilogue
scatter (CUTLASS-sm90 `is_swapAB`): one instantiation fewer per tile
config, at the cost of a scalar-store scatter on a path no LLM-linear
operand pair hits.

**Tile vocabulary (CUTLASS-style).** Tile geometry is expressed as types,
not positional ints: `Shape<M, N, K>` (CTA tile; K = the per-stage k-tile)
and `Shape<M, N>` (warp tile) compose into a `GemmTileConfig` — one named
recipe bundling shapes + stage depth + loop mode. `Shape` itself is the
shared vocabulary type of `utils/shape.cuh`: the same `Shape<...>` spells
both the CTA tile here and the staging layouts' chunk grids, so tile
geometry and smem layout read in one notation. The production manifest in
`policy/gemm/manifest.h` — the named `Tile_*` recipes; read the list there, not here,
because copied enumerations rot — is the `TileManifest` type list the
launch ladders dispatch over (CUTLASS builder-table style: `dispatch_tile`
in `launcher/gemm_tiles.cuh` indexes the manifest by the plan's `TileClass` and depth bit,
both ladder-agnostic); a new geometry is one alias plus one manifest entry
and one planner branch, never a re-spelled per-site ladder. Device
collectives only read the derived `Traits::kBlockM/kBlockN/...` constants,
so this is purely a configuration surface — the generated SASS is
unchanged.

## Planning

`gemm/plan/planner.cpp` ranks recipes; `gemm/plan/table.cpp` owns row parsing,
configuration and lookup. Ties keep manifest order. Decisions are cached
per thread/query; row lookup precedes heuristic ranking, so newly installed rows
take effect immediately. Empty tables avoid their mutex.

| Mode | Search order (first match wins) |
| --- | --- |
| `table` | override → injected → builtin |
| `hybrid` | override → injected → builtin → geom_cta |
| `model` (default) | legacy model with RTX 5090 calibrated cost coefficients |
| `heuristic` | geom_cta |

Builtin rows are device-signature gated. `table_off=True` disables all row tiers;
`planner=""` restores the default. `ASTR_GEMM_*` variables seed configuration
once per process; explicit API calls override them. There is no fixed M-band
fallback: an exhausted planner raises `no eligible recipe`. In `table` mode,
a matching usable row is required. Nonpositive M/N/K are rejected before
dispatch. The autotuner injects only measured cache rows; it does not seed synthetic M-band rows.

**Rows.** One line selects a dtype class, crosswise count, CTA, k_stages and
raster. Shape bands are `(min,max]`, with zero meaning open:

```text
m_min m_max n_min n_max perf_class crosswise cta k_stages raster
    [k [k_min k_max [min_ctas_per_sm [min_wave_permille]]]]
```

The display wraps for readability; each actual row occupies one line.
Optional `k_tile` is the recipe's ring K (default 64); `k_min/k_max` bound the
problem K. Trailing gates require enough grid CTAs per SM or machine-fill
per-mille. Raster zero invokes `plan_raster`. Unsupported recipes or rings
above the device's shared-memory budget fall through. Builtin rows live
between generated markers in `gemm/plan/builtin.cpp`; changing them
requires a rebuild.

**Existing fitted model.** Available only through `planner="model"` for
comparison; the default chain no longer uses it. It prices work as well as
waves, since a large CTA performs more work than a small one:

- cp.async: `(K-padded operand bytes + output bytes) * waves`, using raw
  shared-memory-floor residency in the wave denominator.
- TMA: `max(operand + output + mainloop, MMA_arm) * W_eff`. Mainloop cost
  is `8*BM*BN*ceil(K/KK)` for non-byte pairs, zero for byte pairs; MMA cost
  is 64 times the warp MMA instruction count. `W_eff` is resident-scaled
  waves for two-byte pairs, otherwise `ceil(blocks/SMs)`. The coefficients
  are RTX 5090 fitted equivalent-byte weights, not hardware rates.

TMA residency includes ring padding/barriers. Pointer or stride alignment
failures use cp.async. A smaller KK can improve residency but also adds loop
iterations; neither occupancy nor wave count alone predicts performance.

**Python API.** `plan.py` owns configuration and probe; `autotune.py` owns
measurements, cache rows and persistence. FP8 slots and autocast routing are
owned separately by `fp8_slots.py` and `autocast.py`.

```python
from astrai.extension import plan

plan.configure(planner="heuristic")
plan.configure(rows="rows.txt", tier="override")  # file or inline rows
plan.configure(rows="", tier="injected")         # clear one tier
plan.configure(log=True, tma=False)
plan.probe(512, 11008, 4096)  # decision and source
plan.config; plan.facts; plan.tiles()
with plan.override(planner="model", rows="", tier="override"):
    ...  # restores configuration and rows, including on exceptions
```

Unspecified configuration fields stay unchanged. Raw binding names such as
`set_table`, `set_planner`, `set_staging`, `state`, `probe`, `facts` and
`tile_vocabulary` retain their dict/list contracts for benchmark tooling.

**Autotuning.** `kernel.gemm.enable()` enables candidate timing on caller
tensors and persists winning rows under
`~/.astrai/cache/gemm_plans/<device-sig>.rows`; an override row suppresses
tuning. Offline `csrc/bench/tune_plan_table.py run` uses interleaved sweeps,
a holdout gate (reject ≥2% per-shape regression), and row-band merging
(`--min-gain` defaults to 1%). The sweep covers fused-linear NT layout;
other layouts need separate validation. Generated K/batch conflicts at the
same `(M,N)` currently resolve to the best-TFLOPS point.

## Not implemented

Grouped-along-K / 2-D block scales (GPTQ/AWQ import — needs mainloop scale
application; the epilogue cannot fold them, and the `mxf8f6f4.block_scale`
cell runs identity scales today), asymmetric quantization with zero-points
(offline folding at repack time), sub-int8 dtypes (int4 and 3/5/6/7-bit
need packed staging + a second dequant family), the offline weight
interleave (one `LDS.32` feeding both registers of a dequant pair),
stream-K (wave quantization), warp specialization / cluster / PDL (the TMA
producer is an elected thread of the math CTA, not a dedicated warp),
NVRTC JIT, and MoE gather/grouped GEMM. Supported beyond the base design:
strided-batch operands with broadcast, fp32 output, and the transposed
output orientation (out_layout Dual) produced from one read.
