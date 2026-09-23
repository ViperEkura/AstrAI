# GEMM / Linear Kernel

> Part of the [operator docs](README.md); required reading before changing
> the tile vocabulary, the planners, or the fp8 recipes.

The `quantize` family accelerates bf16 linear layers by
quantizing to FP8 and running tensor-core GEMMs (**requires sm_89+**; fp8
`mma.sync.m16n8k32` only exists on Ada/Hopper). Same three-layer style as
attention; the GEMM device code is split humming/CUTLASS-style into one
layered directory:

| File | Role |
|------|------|
| `quantize/common.h` | capability helpers (`sm_at_least`, `kMinSmForFp8`) + `QuantLayout` + `QuantParams` POD — raw `__nv_fp8_*` element types, no format enum, no torch |
| `quantize/quantize.cuh` | pure-CUDA device code: vectorized `fp8_quantize_kernel` + 64×32-tile transpose kernel (out_layout 0/1/2, Dual orientation a template param), `fp8_cvt_traits<Fp8T>` convert + `quant_in_traits<InT>` unpack (primary templates undefined — one specialization per dtype/format) — no torch |
| `quantize/dequant.cuh` | in-register dequantization functors (`DequantPair<SrcT, MmaT>`): the exact int8→bf16 expansion quantized-GEMM operands fold between the smem read and the mma |
| `gemm/common.h` | dtype-neutral GEMM family declarations: layout tags, `gemm_elem_traits<T>` (kBytes — the smem ring budgets; the MMA K extent rides `MmaShapeFor<MmaT>`), `gemm_mma_traits<ElemA, ElemB>` (MmaT promotion + per-operand kDequantA/B), `GemmParams` POD |
| `gemm/policy.cuh` | dtype-generic `GemmTraits<ElemA, ElemB, CtaShape, WarpShape, Stages>` (tile geometry via the promoted MmaT) + `GemmTileConfig` (CUTLASS-style tile recipe: CTA/warp `Shape` types + stages, with the named production manifest `Tile_128x128x64_W64x32_S2` / `Tile_128x128x64_W64x32_S3` / `Tile_128x64x64_W32x32_S2` / `Tile_128x64x64_W32x32_S3` / `Tile_64x64x64_W16x32_S2` / `Tile_64x64x64_W16x32_S3`) + smem budget (`GemmSmem`) + `GemmPolicy` (dtypes × layouts × one tile config — the kernel's single template parameter) + the `TileClass` dispatch key, the class→CTA-geometry table `kTileClassCta` (static_assert'd against the tiles' CTA shapes) and the `TileManifest` / `TileManifestByte` / `TileManifestCross` ladders the launch ladders index — the congruous and byte ladders compose the crosswise one through `tuple_cat_t`, so their shared six-tile prefix is structural rather than a copy |
| `gemm/load.cuh` | operand loaders: typed staged tiles (`Tensor<PtrEngine<Elem>, StagedLayout>`, `common/tensor.cuh`) over the tile's declared layout (`common/swizzle.cuh`), congruous cp.async staging (predicated via runtime src-size zfill + interior), `PrefetchCarry`, crosswise LDG+PRMT direct load, async trans staging |
| `gemm/scheduler.cuh` | CTA id → (block_m, block_n) grouped/plain raster (runtime `raster` knob) |
| `gemm/mainloop.cuh` | `GemmCollectiveMainloop`: stage rings, stage loads, fragment addressing (ldmatrix + dequantized scalar paths), pipelined mma.sync loop |
| `gemm/epilogue.cuh` | `GemmCollectiveEpilogue`: fused bias + per-row/per-channel scale folding + bf16/fp32 smem scatter + coalesced copy-out |
| `gemm/gemm.cuh` | umbrella: `gemm_kernel<Policy>` orchestrator + device-parameterized host planning (`plan_gemm` / `plan_raster` over `DeviceFacts`; 64×64 / 128×64 / 128×128 CTA) + manifest-driven tile dispatch (`dispatch_tile` over `TileManifest`, one ladder per staging discipline) + entry `gemm_dispatch<ElemA, ElemB, OutT>` = `canonicalize_gemm` → `plan_gemm` → `launch_plan` |
| `gemm/plan_table.h` | The row vocabulary and its chain: `TableRow` (band + recipe + optional gates), the `RowSource` containers (each remembering the spec it was installed from — what makes the config state re-installable), the `GemmConfig` runtime state, the empty-by-default per-class tables (`kBuiltinPlanW16A16`/`W8A16`/`W8A8`/`F8A8` — a device-specific build pastes rows in) and the degraded ladder that ends every chain. The planners themselves (`RowSetPlanner`, `ModelPlanner`) live in `gemm.cuh` |
| `gemm/api.h` | The family's C++ surface — declarations only, and template-free so including it instantiates no dtype-pair kernel: `quant_gemm_impl` (the one GEMM entry), the planner face (`PlanProbe` + `plan_probe`, `GemmConfigPatch` — `rows` + `tier` (`RowTier`) + `table_off` + the mode/log/staging knobs — with the re-installable `GemmConfigState`, through `configure` / `config_state`) and the vocabulary (`tile_vocabulary` / `tile_class_names`). No Python type in a signature — the composed fp8 linear and the bindings TU call the same functions |
| `gemm/gemm.cu` | The typed host layer: the dtype-pair registry (`ASTRAI_GEMM_PAIRS`, one entry feeding both the `gemm_dispatch` and the `plan_probe_for` lookup; one extern-template declaration per pair, which is what keeps this TU from re-instantiating them) plus the `api.h` implementations. Holds no `py::` type |
| `gemm/fp8_linear.cu` | The composed fp8 training linear (forward *and* backward) in one C++ `autograd::Function`, so only one entry call stays in Python; its per-call state machine (rings, weight cast cache, checkpoint snapshot) is `fp8_state.cuh` |
| `gemm/bindings.cu` | The pybind surface of the module: None-tolerant argument marshalling, the dict shapes of both directions (the state report's keys and the config patch's `kPatchKeys` table — each key set spelled once, here, as the contract the Python tooling reads) and `PYBIND11_MODULE` → module `gemm`. `configure` takes one patch dict, so a knob's name has two homes total: this table and the Python signature |
| `quantize/quantize.cu` | binding only: entry checks (`checks.h` device gate + scale validation), param packing, launch dispatch, pybind → module `quantize` |

Scale semantics: `quantize` takes the quantization *multiplier*; the
strategy layer passes `scale.reciprocal()` and the kernel multiplies by it.
`quant_gemm` takes per-operand dequant scales (`a_scale`, `b_scale`;
the fp8 training path passes `sa` / `sb` separately). `amax` is produced
only by the delayed-scaling ring fold (the in-kernel fused reduction) or
measured by the caller — the plain no-ring quantize runs a pure scale+cast
and returns `amax=None`.

Python layer (two levels): `astrai/extension/ops/quantize.py` and
`ops/gemm.py` are the stateless kernel adapters (plain `quantize` /
`quantize_dual` / `quant_gemm` wrappers, one adapter file per compiled kernel
module), and `astrai/extension/quantize.py` is the strategy layer (fp8
recipes, delayed / dynamic scaling, `fp8_autocast`, plus the int8
quantizers). The composed fp8 linear — quantize, ring fold/advance, the
weight-cast cache and all three GEMMs — lives in C++
(`csrc/kernels/gemm/fp8_linear.cu`, compiled into the `gemm` module
where the GEMM dispatch state lives) behind a single Python->C++ crossing;
the strategy layer routes `aten::linear` to it on CUDA. The aten override
installs lazily on the first fp8 activation (autocast enter or the global
enable), so importing the module is dispatcher-neutral.

The fp8 training math contract — the scaled-cast formula, the delayed vs
dynamic scaling recipes, `quantize_dual`, and the forward/backward dataflow
formulas that consume these operands — lives in
[quantize.md](quantize.md); the sm_120 block_scale mma cell is covered in
the design notes below.

## FP8 GEMM design notes

The load-bearing invariants behind the kernel code (all measurements on
L20/sm_89 unless noted):

**Swizzle.** Staging layouts are CUTLASS-style *types*: `common/swizzle.cuh`
provides the `Swizzle<Bits, Shift>` / `Layout<Shape<Rows, Chunks>,
Stride<Chunks, 1>>` / `composition(Swizzle, Layout)` vocabulary (16B-chunk
units, dtype-independent `(Bits, Shift)` pairs), and each collective
declares its tile's layout once — `SmemLayoutA/B` and the trans mirrors in
the mainloop, `OutLayout` in the epilogue. The family instances: congruous
2B staging is the TMA 128B mode `<3,3>`, 1B is `<2,3>`, 16-bit trans
staging swizzles chunks by the k-row bits (custom `<log2(min(chunks,8)),
log2(chunks)>`), and the epilogue output keeps its row-width mode.
The tensor layer dispatches to `ComposedLayout::operator()` (the closed
two-coordinate form) — the 16B chunk index
XORed with row bits at `[3, 3+log2(kChunks))` — so a warp's ldmatrix
fragment load (8 consecutive rows × 16B) hits all 32 banks exactly once
(the unswizzled row word-stride is `kK/4` words, so rows `r` and
`r + 8/kChunks` collide mod 32). Chunks stay contiguous, so cp.async
staging is unaffected. The helper derives the XOR term from the row
coordinate alone (the layout's `kRowShift`/`kMask`) instead of the
linearized index: a linear form serialized the XOR behind the row×stride
multiply in the hot dequant fragment readers and regressed W8A8 up to
+29% (RTX 5090) — keep the two-coordinate form in the hot paths.

**Boundary predication.** Predicated staging rides cp.async's runtime
source size (CUTLASS 2.x's zfill iterators): `cp.async.cg [dst], [src],
16, src_size` with `src_size` derived per chunk from the remaining
contract extent — 0 reads nothing and the hardware zero-fills the 16B
chunk, a partial size covers the k tail, and only a misaligned base
(non-16B `ld`) keeps the scalar-copy fallback.

**Tensor vocabulary (common/tensor.cuh, cute's Tensor<Engine, Layout>).**
ONE tensor type — storage and addressing are its two template parameters,
and every operation dispatches to a layout op; use sites spell
`Tensor<...>` directly, with no second names. Engines: `PtrEngine<T>`
(smem/gmem) and `ArrayEngine<T, N>` (cute's Array — the mma fragment
cells; `MmaOp` names them `AFrag`/`BFrag`/`CFrag`, and the typed
`fma`/`ldmatrix` overloads take them by reference so the registers stay
in place). Layouts: the `ComposedLayout` instances of `common/swizzle.cuh`
(16B chunk grids, dtype-blind; `chunk_of` is the swizzled-chunk op, and
the tensor scales the row/chunk terms separately in 32-bit — a 64-bit
multiply on the address chain regressed the crosswise readers' registers)
plus `RingLayout` (slot rotation over a per-stage grid) and `CellLayout`
(element-unit (m, n) grid). A staged tile is
`Tensor<PtrEngine<Elem>, ComposedLayout>`; the stage ring adds the slot
dimension (`make_ring` / `stage_of`, cute's make_tensor / slicing); the
warp's accumulator is `Tensor<ArrayEngine<CFrag>, CellLayout>` —
`*acc(mt, nt)` at the fma seam, the kPairB x4 fold is `BFragPair::cell`,
so `+ (nt & 1) * 2`-style pointer arithmetic has no spelling left. Every
method folds away at -O3.

**Fragment addressing (base-pair scheme).** One base register per operand
per k_seg, every fragment offset an LDSM immediate. The closure works
because the XOR swizzle's source bits come only from the lane's
row-within-matrix `r7`: the 8/16-row fragment steps never reach them, so
`addr(s, mt) = lane_base + mt*(16*kK) ^ (s<<5)` for A and
`addr(s, nt) = lane_base + nt*(8*kK) ^ (s<<5)` for B. This replaced
runtime offset tables that spilled at 131 registers (~55 of 146 hot-loop
instructions were address math; cuBLAS's inner loop has ~0). Steady-state
read pointers advance one stage per iteration with an equality wrap,
replacing the per-k-tile `(tile % ring) * stage_bytes` recomputation
(UIMAD.WIDE magic-division ladder).

**Pipeline depth and barriers.** Every operand ring holds `kStages+1`
buffers: the load for tile `i+kStages` targets slot `(i-1)%(kStages+1)`,
which compute(i-1) finished reading before this iteration's barrier — no
post-compute barrier, one `__syncthreads` per k-tile. Prologue and tail
commits are unconditional so the group sequence stays tile-indexed and
the fixed `wait_group<kStages-1>` is iteration-invariant (a runtime
wait-count dispatch ladder cost 16 instructions/k-tile). A lean
`kStages`-deep ring trading the barrier for a 4th resident CTA measured
+5..9% slower at 1280³ and was removed. The final iteration carries no
trailing barrier of its own, so the epilogue transition pays one explicit
drain (`PipelineSync<Stages>::drain()`, which empties only the calling
thread's groups) followed by `__syncthreads()` — without it a thread racing
into the epilogue scatters the output tile over peers' still-in-flight
staging writes and final fragment reads (caught by racecheck + a W8A8
stress case: zeroed 32-row output bands on multi-wave grids). The ring
discipline itself (prologue commit, steady wait, tail commit) runs through
the `PipelineSync` stage-pipeline type of `common/pipeline.cuh` — the
sm_80/89 backing of the cp.async rings, with the raw mbarrier PTX sites
(init / arrive_expect_tx / wait_parity) shared with the TMA path below.

**TMA staging (sm_90+, dual-congruous operands).** The congruous rings
can also be fed by TMA (`common/tma.cuh`): the staging swizzles already
ARE the TMA hardware modes (`<3,3>` = SWIZZLE_128B for 2-byte elements,
`<2,3>` = SWIZZLE_64B for 1-byte), so fragment addressing, ring slots
and the epilogue reclaim are untouched — only the load/wait discipline
changes. The templating follows the staging types: `TmaSwizzleOf<Staged>`
derives the swizzle width and the box's inner extent (the mode's span)
from the declared `ComposedLayout` (the encoder decodes the `CUtensorMap`
swizzle enum from that width), and `tma_spec<Elem, Staged, BoxRows>`
fills only the runtime geometry — the map cannot drift from what the
fragments read. Each operand's rank is a template bit on
`GemmTmaContext` / `gemm_kernel_tma` (strided batch = 3D emitter,
broadcast = shared 2D map), so the per-stage 2D/3D issue pick compiles
away; the launcher dispatches the four rank combinations. One elected
thread arms a per-slot mbarrier with `arrive.expect_tx` and issues the
boxes; OOB coordinates zero-fill, which absorbs the
k tail and edge tiles the cp.async zfill iterators predicated per chunk.
Two TMA-specific invariants: the swizzle applies to the ABSOLUTE shared
address, so the ring base rounds up to 1024B (budgeted in
`Policy::kSmemBytes` together with the barrier array); and the pipeline
is the CUTLASS `PipelineTmaAsync` handshake — `full[slot]` (count 1,
expect_tx) for arrival, `empty[slot]` (count = CTA threads, every
consumer arrives after its last fragment read) for release — which
REPLACES the per-k-tile `__syncthreads`: warps skew freely across slots
and the producer's overwrite gate is the empty phase alone. A phase-1
variant that kept the `__syncthreads` (mbarrier only replacing
`wait_group`) measured 2-4% SLOWER than cp.async on RTX 5090 — the
CTA-join elimination is the win, not the TMA issue itself; the full
handshake measures +3..7% over cp.async across the llama shapes (largest
on the bf16 pair, whose fat operands pay the most issue slots).
Descriptors are host-encoded (`cuTensorMapEncodeTiled` via dlsym — no
link-line changes) and cached exact-match, so steady-state calls pay the
few-microsecond encode once. The planner gates TMA on the device's
compute capability, dual-congruous layouts, 1-/2-byte dtypes and
descriptor encodability (misaligned base/ld falls back to the cp.async
twin, which stays compiled); `set_staging(tma=False)` forces the fallback
for experiments.

**Stage depth.** The manifest carries s3 deep-ring siblings of every
class (`Tile_128x128x64_W64x32_S3` / `Tile_128x64x64_W32x32_S3` / `Tile_64x64x64_W16x32_S3`,
humming's `_fit_num_stages` rule — the thinner the operand pair, the
more smem headroom under the 96KB budget). RTX 5090 measured them a
wash to -1.3% on the cp.async rings (three buffers already hide the
LDGSTS latency), so table rows keep s2 wherever a sweep point did not
measure s3 ahead. Candidates are measured as one-row tables (`set_table`)
files (see `csrc/bench/tune_plan_table.py sweep`), which name the ring depth
directly and are smem-gated like a real row.

**MMA cell (sm_120a block_scale).** The plain warp-level fp8 mma
(`m16n8k32.e4m3/e5m2`) decodes at HALF rate on sm_120 — measured pure issue
rate 506 TFLOPS vs 1011 for the shape- and width-identical s8 instruction
(RTX 5090). The `kind::mxf8f6f4` block_scale variant
(`mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X…ue8m0`,
humming's MXMMA route) runs at the full 1007 TFLOPS with an IDENTICAL
A/B/C/D register contract, so the mainloop/staging/epilogue layers are
untouched: `MxMmaOp` (common/mma.cuh) swaps only the cell, carrying a
constant unit scale (every ue8m0 scale byte 0x7f = 2^0, selectors inert —
the scale-factored product IS the plain product; quant_gemm_test output stays
byte-identical). Warp-level block_scale is an sm_120-FAMILY instruction
(CUDA 13.0 ptxas: 120a/121a/120f accepted, 100a/103a/110a rejected —
datacenter Blackwell does MX through tcgen05, which needs 13.1+), so the
cell gates on the family pass (`ASTRAI_ARCH_FAMILY >= 1200`,
common/mma.cuh's value-macro twin of `ASTRAI_DEVICE_ARCH`) and the gemm
fatbin carries the sm_120a image (the arch token names it —
`ASTRAI_CUDA_ARCH=120a`; CMake's native `a` grammar emits one image per
token, and a plain `120` list warns at configure time) which the driver
picks on sm_120; every other pass/device falls back to the plain cell
inside the same tree, so routing can only trade speed, never correctness.
`launch_plan` routes the symmetric-fp8 pair through the mx tree on sm_120
unless `set_staging(mx=False)` knocks it out (the A/B knob; read
once per process — separate processes to compare). End to end the fp8
pair's benchmark geomean went 356 → 508 TFLOPS (+43%, peak 619; non-fp8
classes unchanged — their kernels are SASS-identical in both images). **Crosswise loads.** Crosswise operands (A `[K][M]` / B `[N][K]` storage)
cannot cp.async into the canonical tile; they take the direct LDG.128×4 +
in-register PRMT transpose + STS.32 path. A staged variant (cp.async into
K-major staging + per-tile smem→smem transpose) measured 15-20% slower
across every probed shape including DRAM-streaming B (git history 5745c2f).

**Fast-loop peel.** When both operands are congruous, the whole CTA is
interior, base|ld is 16B-aligned and K has no tail, the mainloop switches
to a predication-free copy with loop-carried prefetch state: +4.5..10% on
the issue-bound 64×64 CTA (256³..1024³), −3% on the 128×128 CTA, so only
the small CTA opts in.

**Tile vocabulary (CUTLASS-style).** Tile geometry is expressed as types,
not positional ints: `Shape<M, N, K>` (CTA tile; K = the per-stage k-tile)
and `Shape<M, N>` (warp tile) compose into a `GemmTileConfig` — one named
recipe bundling shapes + stage depth + loop mode. `Shape` itself is the
shared vocabulary type of `common/shape.cuh`: the same `Shape<...>`
spells both the CTA tile here and the staging layouts' chunk grids, so
tile geometry and smem layout read in one notation. The production
manifest in `policy.cuh` (`Tile_128x128x64_W64x32_S2`, `Tile_128x128x64_W64x32_S2`,
`Tile_128x64x64_W32x32_S2`, `Tile_64x64x64_W16x32_S2/s3`) is the `TileManifest` type list
the launch ladders dispatch over (CUTLASS builder-table style: `dispatch_tile`
in `gemm.cuh` indexes the manifest by the plan's `TileClass` and depth bit,
both ladder-agnostic); a new geometry is one alias plus one manifest entry
and one planner branch, never a re-spelled per-site ladder. Device
collectives only read the derived `Traits::kBlockM/kBlockN/...` constants,
so this is purely a configuration surface — the generated SASS is unchanged.

**Launch planning (retired 2026-09-09).** The original wave-count cost
model was deleted from `gemm.cuh` — dispatch is table-first (AOT dispatch
table below), the degraded band rows are the last resort, and the
analytical `ModelPlanner` (Runtime plan autotuning) fills table misses.
What survives from that era as measured rules: the padding gate and the
crosswise ladder (crosswise loads price differently — the small CTA hides
their latency, the big CTA's reuse wins once its grid fills ~1.5 waves),
and the raster L2-budget rule (B already L2-resident means plain raster;
otherwise reserve a B-stream fraction of L2 and size the M-group so its A
tiles stay resident across the N sweep). Persistent schedules measured
worse on L20 (−4..−8%; the ticket variant recovers L2 locality but its
loop-head barrier costs what the CTA-restart overlap saves). The retired
model's calibration record (per-regime `kPlanEff` fit over an offline
sweep+holdout; on the RTX 5090 the refit cut holdout regret 12.7% → 2.9%
at p90 and the llama W16A16 mean 184 → 194 TF, and replacing the L20
ladder's picks totaled −5.3% W16A16 / −1.3% W8A16 / −1.5% W8A8 with no
case >3% worse) is the reference the shipped sweep tables were validated
against.

**AOT dispatch table** (`plan_table.h`): the production planner —
`plan_gemm` consults a measured row table: (M, N, K) bands (min exclusive,
max inclusive, 0 = open), keyed per dtype class / crosswise count, each row
naming a recipe (CTA class + ring depth; raster 0 = `plan_raster` with the
row's geometry). The compiled-in rows are one table per dtype class
(`kBuiltinPlanW16A16` / `W8A16` / `W8A8` / `F8A8`, selected by
`builtin_plan_table`): the class *is* the table — a row tuned for one
operand pair cannot fire on another (`gemm_perf_class` tests the int8 pair
before the "not bf16 → fp8 pair" arm; an earlier `perf_class` field once
made every W8A8 row unreachable while int8 dispatched on F8A8 rows).
`plan_from_row` is the single interpreter: geometry from the CTA class,
ring depth from the row, raster 0 = `plan_raster`, and one smem gate — a
row whose ring exceeds the smem opt-in ceiling is a stale tuning artifact
and falls through to the degraded bands instead of a failed launch. A miss
falls to the degraded band rows (last-resort M-band geometry: small ≤ 512,
narrow ≤ 3072, big beyond), always matching, so planning is a total
function; the wave-count cost model is deleted from the codebase.
Full-coverage tables end each class with a catch-all row, so a miss means
the table is empty or stale, not a shape the planner should infer.

Rows come from two sources, override first: `set_table(path, inline text,
or "-" to turn AOT off entirely — degraded bands only)` — one row per
line, `m_min m_max n_min n_max perf_class crosswise cta stages raster
[k [k_min k_max [min_ctas_per_sm]]]`. The optional `k` is the row's ring
K (omitted keeps 64; a kK=32 row only survives a dual-2-byte pair); the
optional trailing pair is the row's K band, same (min, max] rule, omitted
keeps it open — the winning recipe can flip with K below ~512, where the
wide CTA loses the epilogue race and the kK=32 twin wins hardest, and the
W8A8 wide rows carry the band that holds under both scale granularities.
The last optional field is the row's WAVE GATE `min_ctas_per_sm`: a row
matches only while its own grid covers that many CTAs per SM, priced
against `DeviceFacts` at lookup time — "past M 3072" is usually wave
arithmetic wearing a literal M calibrated to one SM count, while measured
latency crossovers keep literal bounds and re-measure on a new device.
Compiled-in rows are pasted manually between the GENERATED markers (the
measurement script only emits the row file; a rebuild picks it up). Every
sweep candidate is probed for the planner's decision tag first, so a
candidate some gate demotes is dropped rather than recorded under the
fallback's numbers. The sweep times every combo × recipe over the M ×
shape grid, each candidate as a one-row table toggled per launch —
interleaved at each shape so the comparison shares one clock/thermal state
(sweeping whole recipe batches in separate processes measured the big CTA
first and the small CTA 30 minutes later and picked systematically wrong
winners); winners per point become rows, adjacent M runs with the same
winner band-merge at mid-point edges. `--min-gain` (default 1%) rows only
leads above a minimum gain; `--full-coverage` rows every band, ties broken
by the stable big>narrow>small preference, plus the per-class catch-all.
K is not a row key (the ring K is fixed at 64): a K/batch conflict at one
(M, N) resolves to the best-TFLOPS point. The sweep times the fused-linear
(NT) layout, so generated rows carry crosswise 0 — non-NT shapes (TT, TN,
the mixed dual-row-major NN case) miss into the degraded bands. The
production route runs through the hooked planner, so the correctness suite
exercises the row format indirectly, and the smem gate turns a stale row
into a degraded-band launch instead of a launch failure.

**NN swap.** The dual-N-contiguous problem runs as its transpose
`E = B^T @ A^T` over swapped operands with an out-transposed epilogue
scatter (CUTLASS-sm90 `is_swapAB`): one instantiation fewer per tile
config, at the cost of a scalar-store scatter on a path no LLM-linear
operand pair hits.

## Quantized GEMM: W8A16 / W8A8 / W16A16 (humming-style)

The same mainloop serves every dtype pairing through one promotion rule
(`gemm_mma_traits`): the mma runs on the **MmaT** — symmetric fp8 keeps its
native `m16n8k32`, symmetric bf16 (W16A16) passes through untouched,
**symmetric int8 (W8A8) keeps its native `m16n8k32.s8.s8.s32`** (int32
accumulators; `.satfinite` clamps the wrap all-max-magnitude K≈16k inputs
could reach — the standard production int8-GEMM semantics), and a *lone*
int8 (W8A16 weight-only or the mirrored A8W16) promotes to bf16
`m16n8k16` with per-operand in-register dequant (`kDequantA` /
`kDequantB` — W8A16 only B). Staging never changes: int8 operands ride
the existing congruous cp.async / crosswise PRMT paths into the canonical
swizzled tiles, and `kMmaK` follows the mma cell (the `MmaShapeFor`
trait) so the tile geometry is shared — the native s8 pair reuses the
fp8 k32 fragment layouts verbatim (1-byte dtypes share the packed
two-per-b16-slot layout, so ldmatrix addressing is identical).

**Mma trait layer (common/mma.cuh, humming's compile-time format).** The
instruction vocabulary assembles from two specializable traits:
`MmaShapeFor<Dtype>` maps an input dtype to its instruction `Shape<M, N,
K>` (primary template undefined — a dtype with no MMA is a compile error;
K follows the 256-bit A-fragment invariant: bf16 k16, 1-byte dtypes k32;
`kMinArch` encodes each instruction's hardware floor as a build-time
assert), and `MmaOp<A, B, Shape>` — one specialization per
instantiated pairing cell carrying the accumulator type, register counts
and the dedicated asm block. `mma_sync<InT>` remains as the fp32-family
convenience view (attention + tests); the gemm mainloop calls
`Traits::MmaOp::fma` directly so the accumulator type rides the cell
(fp32 for float families, s32 for s8). `Shape` itself lives in
`common/shape.cuh` — the shared static-geometry vocabulary the policy,
staging and mma layers all spell.

The switch from the dequant-promoted W8A8 (measured at 162-198 TF,
0.92× cuBLAS bf16 on the RTX 5090) to the native s8 mma delivers
520-650 TF — 2.6-3.0× cuBLAS bf16 end-to-end, ~3× the old path at
identical staging and tile choices (kPlanEff's kW8A8 scalars held; the
relative tile efficiencies barely moved with the mma no longer the
bottleneck).

**Dequant (quantize/dequant.cuh).** Each fragment register pair costs one
`LDS.16` + four LOP3-class instructions, exact for the full int8 range
including −128. bf16 carries only 7 mantissa bits, so the naive
"OR the byte into a bf16 base" trick (0x6400-style, exact for fp16) breaks
linearity — bit 7 spills into the exponent. Instead the magnitude bits
(0-6) and the sign bit (7) take separate LOP3s:
`h = (u & 0x7F) | 0x4300` → exactly 128+u7; `s = (u & 0x80) | 0x4300` →
128 or 256 as the sign picks; `v = h - s` is the exact int8 value, and
every intermediate is bf16-exact. A future humming-style offline byte
interleave (per k16 slice, u32 word c holding `(k₂c, k₂c₊₈, k₂c₊₁, k₂c₉)`)
would let one `LDS.32` feed both registers of a pair and drop the spread
PRMT; deferred until measurement justifies a repack pass.

**Scales.** `GemmParams` carries per-operand dequant scales folded
multiplicatively into the epilogue (the mma accumulates the raw quantized
product): per-tensor device scalar, per-row activation `a_scale[m]`, or
per-channel weight `b_scale[n]`. The epilogue applies them after the
accumulator's int→float conversion, so the s32 path shares one scatter
code. Grouped-along-K scales belong in the mainloop and are not
implemented. The transposed-output epilogue branch applies
`b_scale`/bias per kernel row — including the +8 accumulator half
(its own row factor), a pre-existing mixup the first scale-carrying
NN-swap test exposed.

**Python surface (two layers).** `astrai/extension/ops/gemm.py` is the
compiled `gemm` module's adapter — the single `quant_gemm` entry (the
operand dtype pair picks the kernel, per-side scale arity is validated:
int8 requires its scale, fp8 takes one optionally, bf16 takes none); `astrai/extension/quantize.py` carries the int8
policy (symmetric per-channel weight quantization, per-row dynamic
activation quantization). There is deliberately no nn.Module layer on the
int8 path: the only model-facing quantization integration is the fp8
autocast (same module, routing `aten::linear`), and quantized callers
compose the primitives directly.

Benchmark (L20, llama weight shapes, `csrc/bench/benchmark_quant_gemm.py`,
M=2048/4096): W8A16 reaches 0.83–1.08× of cuBLAS bf16 `F.linear`
(28–42 TFLOPS, faster than bf16 on the wide up_gate shapes where halved
weight traffic pays), W8A8 0.73–0.86×, W16A16 0.83–0.94× — the dequant
insert is not the bottleneck at these shapes; the modes track the
W16A16 baseline within a few percent.

**Humming parity (what we deliberately have and have not).** Adopted from
humming: in-register LOP3 dequant, the dtype-promotion unified mainloop,
per-operand epilogue scale placement, the CUTLASS-style
`Shape`/`GemmTileConfig` vocabulary, the device-parameterized launch
planning (the `DeviceFacts` wave-count model plus the L2-budget raster
rule from humming's tune heuristics), and — since the TMA staging — the
sm90+ load/PDMA vocabulary (TMA descriptors + the mbarrier full/empty
handshake humming's sm120 heuristics enable for WnA16). Not
adopted, in rough priority order
for future work: grouped-along-K / 2-D block scales (GPTQ/AWQ import —
needs mainloop scale application, the epilogue cannot fold them),
asymmetric quantization with zero-points (offline folding at repack time),
sub-int8 dtypes (int4 and 3/5/6/7-bit need packed staging + a second
dequant family), the offline weight interleave (documented deferred
above), stream-K (wave-quantization; persistent scheduling alone measured
worse here), and warp specialization / cluster / PDL (the TMA producer is
currently an elected thread of the math CTA, not a dedicated warp). Out
of scope by design: NVRTC JIT and MoE
gather/grouped GEMM (the JIT-per-SM-heuristic idea survives AOT as the
per-dtype-class `kPlanEff` table, calibrated by the offline sweep). We keep
two things humming lacks: strided-batch operands with broadcast, and fp32
output.


## Runtime plan autotuning (ops.gemm)

The planner the launch path uses is configured at runtime from Python —
**no rebuild, no environment variable**. The surface is
`astrai.extension.plan` (the gemm family's policy module) and it is
re-exported from `astrai.extension`; the flat `set_*` / `state` / `probe` /
`facts` / `tile_vocabulary` views over the same bindings stay in
`astrai.extension.ops.gemm` (their raw dict/list shapes are what the
`csrc/bench` tools parse). It is one value, one writer and one scope:

```python
from astrai.extension import plan

plan.config                            # the whole configuration, as a value
plan.configure(planner="hybrid")       # "table" | "hybrid" | "model" | "" (unset)
plan.configure(rows="rows.txt", tier="override")   # a row file or inline text
plan.configure(rows="", tier="injected")           # clear that tier
plan.configure(table_off=True)         # every row tier off at once
plan.configure(log=True, tma=False)    # the decision log; the A/B staging switches
plan.probe(512, 11008, 4096)           # the decision + who made it
plan.facts                             # the DeviceFacts geometry
plan.tiles()                           # the recipe vocabulary, with class names

with plan.override(planner="model", rows="", tier="override"):
    ...                                # restored on exit — knobs *and* rows
```

`configure` leaves every argument it is not given alone and returns the
resulting value, so a saved `config` is re-installable: feeding its fields
back restores exactly that state (each row tier carries the source spec it
was installed from, and a plain `override(...)` block does this for you —
including when the block raises). The records answer to the old idioms too
(`cfg["staging"]["tma"]`, `tile[3]`, 12-tuple unpacking), and the flat
`set_table` / `set_planner` / `set_log` / `set_staging` / `state` / `probe` /
`facts` / `tile_vocabulary` names remain as the same bindings in their raw
dict/list shapes — the four `csrc/bench` tools parse those keys, so those
spellings are contract.

`plan.probe` returns the decision `gemm_dispatch` would make, with the
planner that made it: `"override"` (rows from the override tier),
`"injected"` (rows from the injected tier, ranked below override and above
the compiled-in table), `"builtin"`, `"model"` (the analytical planner), or
`"degraded"` (the band ladder).

The planners compose as a chain — override rows, injected rows, the
compiled-in table, the analytical model, the degraded bands — and the mode
above only picks which chain runs; `configure(table_off=True)` disables every
row tier at once. The shipped default is **hybrid with the compiled-in tables
empty**, so a fresh process answers with the analytical model and takes
measured recipes from the override tier or the autotuner cache; a measured row is
only shipped when a device-specific build pastes one in (measured 2026-09-14
on sm_120: the previous W16A16 rows were +4.3% behind the degraded ladder
across a holdout, six shapes past +2%). `configure(planner="")` restores the
default instead of pinning a mode. The retired `ASTR_GEMM_*` environment
variables are still read once per process as a migration seed; explicit
API calls win.

The analytical model is a port of DeepGEMM's config search
(`get_best_configs`) that keeps only what survives measurement. DeepGEMM
ranks candidates by **wave count**; that is a valid proxy only when every
candidate's wave costs the same, which holds for it because its blocks are
pinned to instruction shapes. Here a 64x64 CTA's wave carries a quarter of
a 128x128's work, so counting waves prefers the coarse tile — measured over
the production cells (9 shapes, sm_89), wave count ranks the shapes at
**rho -0.90** against the measurement and picks the *worst* cell on the
narrow-N band. Pricing the wave instead does not rescue it: written with a
fractional last wave the makespan is `(blocks/slots) * concurrency * solo`,
`solo` grows with `bm*bn` while `blocks` shrinks with it, and the product is
the same `MN` for every candidate of a problem. That matches what the sweep
measures (every production cell within 1.13-1.38x of every other on a given
shape), so the model keeps no FLOP or traffic term at all and ranks on the
one axis that is left:

```
better = (resident desc, stages desc, manifest order)   # resident = CTAs/SM the ring allows
```

More CTAs resident per SM, then deeper prefetch. That picks the 16-warp
64x64 cell on the kK=32 ring, which is the measured best or second-best cell
on **9 of 9** swept shapes. `kK` is not a model axis — DeepGEMM fixes
`block_k` — and falls out of the same residency rule: the kK=32 twin's 48KB
ring holds two CTAs where kK=64's 96KB holds one.

Dropping the L1/L2/FLOP rates also drops the last per-architecture
constant, so the model carries **none** and behaves the same on every part.
What it cannot express is the per-CTA efficiency that separates the classes
at a given `(M, N)` — the measured axis the row tables own, and the reason
the hybrid chain keeps rows first.

`ops.gemm.enable()` installs the runtime autotuner (shapes no row serves
tune once: candidates from `tile_vocabulary` filtered to the staging pair
and smem ceiling, forced as one-row tables, interleaved CUDA-event medians
over the caller's own tensors; the winner persists under
`~/.astrai/cache/gemm_plans/<device-sig>.rows` so a new process or a
different part re-derives nothing measured). The hook costs one flag check
when disabled and idles while an override table owns the source. The
offline whole-table recalibration stays a separate command:
`csrc/bench/tune_plan_table.py run` sweeps (`sweep --full-coverage`),
gates on the holdout validator (a ≥2% per-shape regression rejects), and
installs under the same device signature.
