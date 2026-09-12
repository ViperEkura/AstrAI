# CUDA Kernels

AstrAI includes optional custom CUDA kernels for attention, rotary embedding,
and FP8 GEMM. These are built when `nvcc` is available and CUDA is detected.

## Overview

| Kernel | File | Description |
|--------|------|-------------|
| `attn_decode` | `attention/decode.cu` | GQA decode attention (split-KV) |
| `attn_prefill` | `attention/prefill.cu` | GQA prefill attention (split-Q) |
| `attn_paged_decode` | `attention/paged_decode.cu` | Paged KV cache decode attention |
| `attn_paged_prefill` | `attention/paged_prefill.cu` | Paged KV cache prefill attention (ragged batch) |
| `rotary_emb` | `rotary_emb.cu` | Fused rotary embedding (cos/sin lookup + rotation) |
| `quantize` | `quantize/quantize.cu` | FP8 quantization kernels (sm_89+) |
| `gemm` | `gemm/gemm.cu` + per-dtype-pair `gemm_*.cu` | dtype-generic tensor-core GEMM binding + one explicit `gemm_dispatch` instantiation per dtype pair (fp8 / W8A16 / W8A8 / W16A16, sm_89+) |

Additionally, optimized `.cuh` variants with tensor-core MMA (Matrix Multiply-Accumulate) exist:

| Variant | File | Optimization |
|---------|------|--------------|
| Split-KV MMA decode | `attention/decode_split_kv_mma.cuh` | Split KV across warps + MMA (sm_80+) |
| Split-Q MMA prefill | `attention/prefill_split_q_mma.cuh` | Split Q across warps + MMA (sm_80+) |

> The paged and non-paged paths share one kernel body. Prefill is templated on
> an independent Q schedule (`DenseQSchedule` / `PackedQSchedule`) and KV
> source (`ContigKV` / `PagedKV`); decode only needs the KV source. There are
> no separate `attn_paged_*.cuh` files.

### Rotary Embedding Kernel

The `rotary_emb` kernel (`csrc/kernels/rotary_emb.cu`) fuses cos/sin lookup and rotation into a single kernel:

- One thread per (head, dim-pair), vectorized `__nv_bfloat162` load/store
- f32 cos/sin input, bf16 compute and output
- 256-thread blocks, grid-stride loop
- Auto-dispatched via `apply_rotary_emb` in `astrai/extension/backend/rotary.py` (CUDA when available + inference mode, else torch complex-multiply fallback)
- No context-manager backend needed — rotary is backend-agnostic, both attention backends benefit

Standalone benchmark vs torch complex-multiply (48 calls = 24 layers × q+k): 6-9x faster, max diff 0 (decode) to 3e-2 (large prefill, bf16).

### FP8 GEMM / Linear Kernel

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
| `gemm/policy.cuh` | dtype-generic `GemmTraits<ElemA, ElemB, CtaShape, WarpShape, Stages>` (tile geometry via the promoted MmaT) + `GemmTileConfig` (CUTLASS-style tile recipe: CTA/warp `Shape` types + stages + loop mode, with the named production manifest `Tile_128x128x64_W64x32_S2_Fast` / `Tile_128x128x64_W64x32_S3_Fast` / `Tile_128x64x64_W32x32_S2_Fast` / `Tile_128x64x64_W32x32_S3_Fast` / `Tile_64x64x64_W16x32_S2_Fast` / `Tile_64x64x64_W16x32_S3_Fast`) + smem budget (`GemmSmem`) + `GemmPolicy` (dtypes × layouts × one tile config — the kernel's single template parameter) + the `TileClass` dispatch key, the class→CTA-geometry table `kTileClassCta` (static_assert'd against the tiles' CTA shapes) and the `TileManifest` / `TileManifestByte` / `TileManifestCross` ladders the launch ladders index — the congruous and byte ladders compose the crosswise one through `tuple_cat_t`, so their shared six-tile prefix is structural rather than a copy |
| `gemm/load.cuh` | operand loaders: typed staged tiles (`Tensor<PtrEngine<Elem>, StagedLayout>`, `common/tensor.cuh`) over the tile's declared layout (`common/swizzle.cuh`), congruous cp.async staging (predicated via runtime src-size zfill + interior), `PrefetchCarry`, crosswise LDG+PRMT direct load, async trans staging |
| `gemm/scheduler.cuh` | CTA id → (block_m, block_n) grouped/plain raster (runtime `raster` knob) |
| `gemm/mainloop.cuh` | `GemmCollectiveMainloop`: stage rings, stage loads, fragment addressing (ldmatrix + dequantized scalar paths), pipelined mma.sync loop |
| `gemm/epilogue.cuh` | `GemmCollectiveEpilogue`: fused bias + per-row/per-channel scale folding + bf16/fp32 smem scatter + coalesced copy-out |
| `gemm/gemm.cuh` | umbrella: `gemm_kernel<Policy>` orchestrator + device-parameterized host planning (`plan_gemm` / `plan_raster` over `DeviceFacts`; 64×64 / 128×64 / 128×128 CTA) + manifest-driven tile dispatch (`dispatch_tile` over `TileManifest`, one ladder per staging discipline) + entry `gemm_dispatch<ElemA, ElemB, OutT>` = `canonicalize_gemm` → `plan_gemm` → `launch_plan` |
| `gemm/plan_table.h` | AOT dispatch rows: (M, N, K) band × dtype class → measured recipe winner (`TableRow`; band-parse + lookup, `ASTR_GEMM_TABLE` override or the per-class compiled-in tables `kBuiltinPlanW16A16`/`W8A16`/`W8A8`/`F8A8`) — `plan_gemm` is table-only (planning section below); a row's CTA geometry is read from `policy.cuh`'s `kTileClassCta`, not re-spelled here |
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
recipes, delayed / dynamic scaling, `fp8_autocast`,
`fp8_linear_forward/backward` wiring `aten::linear` on CUDA, plus the int8
quantizers). The aten override installs lazily on the first fp8 activation
(autocast enter or the global enable), so importing the module is
dispatcher-neutral.

#### FP8 GEMM design notes

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
twin, which stays compiled); `ASTR_GEMM_NO_TMA=1` forces the fallback
for experiments.

**Stage depth.** The manifest carries s3 deep-ring siblings of every
class (`Tile_128x128x64_W64x32_S3_Fast` / `Tile_128x64x64_W32x32_S3_Fast` / `Tile_64x64x64_W16x32_S3_Fast`,
humming's `_fit_num_stages` rule — the thinner the operand pair, the
more smem headroom under the 96KB budget). RTX 5090 measured them a
wash to -1.3% on the cp.async rings (three buffers already hide the
LDGSTS latency), so table rows keep s2 wherever a sweep point did not
measure s3 ahead. Candidates are measured as one-row `ASTR_GEMM_TABLE`
files (see `csrc/bench/gen_plan_table.py`), which name the ring depth
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
fatbin carries a sm_120a image next to the plain ones (CMake appends the
`-gencode` — the 3.22 `CUDA_ARCHITECTURES` grammar rejects the `a` suffix)
which the driver picks on sm_120; every other pass/device falls back to the
plain cell inside the same tree, so routing can only trade speed, never
correctness.
`launch_plan` routes the symmetric-fp8 pair through the mx tree on sm_120
unless `ASTR_GEMM_NO_MX=1` knocks it out (the A/B knob; the env is read
once per process — separate processes to compare). End to end the fp8
pair's benchmark geomean went 356 → 508 TFLOPS (+43%, peak 619; non-fp8
classes unchanged — their kernels are SASS-identical in both images). The
`kPlanEff` F8A8 row re-derived to {0.82, 0.52} (was 0.85/0.56): the
doubled mma rate leaves the finer tiles staging-bound; both values since
superseded by the wave-regime recalibration (Launch planning, below).

**Crosswise loads.** Crosswise operands (A `[K][M]` / B `[N][K]` storage)
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
manifest in `policy.cuh` (`Tile_128x128x64_W64x32_S2`, `Tile_128x128x64_W64x32_S2_Fast`,
`Tile_128x64x64_W32x32_S2_Fast`, `Tile_64x64x64_W16x32_S2_Fast/s3`) is the `TileManifest` type list
the launch ladders dispatch over (CUTLASS builder-table style: `dispatch_tile`
in `gemm.cuh` indexes the manifest by the plan's `TileClass` and depth bit,
both ladder-agnostic); a new geometry is one alias plus one manifest entry
and one planner branch, never a re-spelled per-site ladder. Device
collectives only read the derived `Traits::kBlockM/kBlockN/...` constants,
so this is purely a configuration surface — the generated SASS is unchanged.

**Launch planning (retired 2026-09-09).** The wave-count cost model
described here was deleted from `gemm.cuh` — production dispatch is
table-only (AOT dispatch table below), with the degraded band rows as
the last resort; kept below as the record of the approach and its
calibration (the reference planner the sweep tables were validated
against). The congruous (NT) band picked its recipe by the wave-count
model over the manifest —
`cost = ceil(tiles / sms) · bm·bn / eff · padding-waste` per recipe —
instead of measured crossover thresholds, so the bands follow the device
arithmetic (`DeviceFacts`: SM count + L2 size + the per-block smem opt-in
ceiling, `common/device.cuh`) rather than one GPU's calibration.
Recipes over that smem ceiling are pruned before scoring (humming's
candidate filter; every manifest recipe fits on the production archs —
96KB max vs 99KB optin — so the gate only guards ports), and the small
recipe's 3-stage variant requires the same headroom. `eff` is the per-SM
throughput scalar of a recipe relative to the big CTA, one row per dtype
class and one cell per wave regime of the recipe's own grid ({1, 2, ≥3}
waves — `kPlanEff[4][2][3]`; a flat scalar spans the regimes badly: the
finer tiles ride even with big only once the grid saturates). It also
absorbs smem residency, which is why the model needs
no separate residency term. The scan prefers the bigger tile and a
challenger needs a >2% lead (hysteresis); the small recipe's ring depth
follows its wave count (3-stage below ~2 waves — lighter smem keeps a
second CTA resident — 4-stage past it). The padding gate and the
crosswise ladder stay measured rules (crosswise loads price differently;
the small CTA hides their latency, the big CTA's reuse wins once its grid
fills ~1.5 waves). Raster group width along M is humming's L2-budget
rule: B already L2-resident means plain raster; otherwise reserve a
B-stream fraction of L2 (fatter B than A reserves more) and size the
group so its A tiles stay resident across the N sweep, floored at enough
M rows to keep every SM busy in one sweep; the N-side mirror keeps the
measured width 8. Persistent schedules (static round-robin and atomic
ticket) both measured worse on L20 (−4..−8%; the ticket variant recovers
L2 locality but its loop-head barrier costs what the CTA-restart overlap
saves).

Calibration: an offline sweep (standalone-nvcc harness, the C-test
convention; kept out of tree) times every manifest recipe × dtype combo
over an M×(N,K) shape grid through the production launch route (TMA
first, production raster; infeasible recipes excluded rather than timed
zero), and the fit takes the per-regime eff medians that make the cost
model reproduce the measured time ratios, plus a shape-holdout report
(winner-class accuracy, measured-time regret) and a cuBLAS gap list.
Porting to a new GPU is re-running that sweep+fit. On the
RTX 5090 the wave-regime refit (7 combos × 81 shapes) cut the holdout
regret from 12.7% to 2.9% at p90 (winner-class accuracy 69% → 79%): the
llama-shape W16A16 mean went 184 → 194 TF with the weak bands
(+10..27%) landing at 0.99-1.00× cuBLAS while the saturated bands and
the quantized-class means held within ±1.2% (single shapes trade up to
−7% inside the fitted regret envelope).
`ASTR_GEMM_PLAN=1` adds a read-only launch log (shape →
recipe/stages/grid/raster) from `launch_policy`. On RTX 5090 (170 SM, 96 MB L2) the model replaced the
L20 ladder's mid/large-M narrow picks with the big CTA and its sub-wave
small picks with s2: benchmark_w8 over the llama shapes totals
−5.3% (W16A16) / −1.3% (W8A16) / −1.5% (W8A8) vs the retired ladder
with no case regressing >3%.

**AOT dispatch table** (`plan_table.h`): the table is the production
planner — `plan_gemm` consults a measured row table: (M, N, K) bands
(min exclusive, max inclusive, 0 = open), keyed per dtype class /
crosswise count, each row naming a recipe (CTA class + ring depth;
raster 0 = `plan_raster` with the row's geometry). The compiled-in rows are
one table per dtype class (`kBuiltinPlanW16A16` / `W8A16` / `W8A8` /
`F8A8`, selected by `builtin_plan_table`): the class *is* the table, so a
row tuned for one operand pair cannot fire on another, and a static_assert
rejects a row keyed for a different class. (The earlier single mixed table
matched rows on a `perf_class` field, which made a row's reach a property
of class ids alone — and int8 shared an id with fp8 until
`gemm_perf_class` stopped bucketing it there, so every W8A8 row was
unreachable while int8 dispatched on the F8A8 rows. `gemm_perf_class` now
tests the int8 pair before the "not bf16 → fp8 pair" arm, with
static_asserts pinning all four pairings.) `plan_from_row` is the
single interpreter: geometry from the CTA class, ring depth from the row,
raster 0 = `plan_raster` at the row's geometry, and one smem gate (a row
whose ring exceeds the smem opt-in ceiling is a stale tuning artifact — it
falls through to the degraded bands instead of a failed launch). A miss
falls to the degraded
band rows — last-resort M-band geometry (small ≤ 512, narrow ≤ 3072,
big beyond), always matching, so planning is a total function. The
original cost model is deleted from the codebase (it was retired from
production in this revision and never called again); full-coverage
tables end each class with a
catch-all row, so a miss means the table is empty or stale, not a shape
the planner should infer. Rows come
from two sources, override first: `ASTR_GEMM_TABLE=/path/to/plan_table.txt`
(one row per line, `m_min m_max n_min n_max perf_class crosswise cta
stages raster [k [k_min k_max [min_ctas_per_sm]]]`, where the optional `k` is
the row's ring K
— omitted keeps 64, and a kK=32 row only survives a dual-2-byte pair — and
the optional trailing pair is the row's contract-depth band, same (min, max]
rule as M and N, omitted keeps it open so older row files and sweeps keep
their meaning; the band exists because the winning recipe can flip with K —
the wide CTA loses the short-K epilogue race below ~512 while the kK=32 twin
wins short K hardest — so one (M, N) band would have to lose one of them;
note the crossover is epilogue dependent, which is why the W8A8 wide rows
carry the band that holds under both per-tensor and per-channel scales;
the last optional field is the row's WAVE GATE: `min_ctas_per_sm` makes a row
match only while its own tile's grid covers that many CTAs per SM, priced
against `DeviceFacts` at lookup time. A bound like "past M 3072" is usually
wave arithmetic wearing a literal M — a tile wins once its grid fills the
slots its ring leaves resident, which is `sms * resident_ctas` on the part
actually running — and a literal is calibrated to one SM count, so the gated
form states the requirement once and every device supplies its own count (the
M 3072 row means 2 CTAs/SM, i.e. M 2816 on a 128-SM part and M 3712 on a
170-SM one). Bands that are not wave arithmetic — a measured latency
crossover, or a table property like "a row above already answers this" — keep
literal bounds and are re-measured on a new device, which is what porting to
one means below;
re-parsed only
when the env path changes; the special
value `-` turns AOT off entirely — neither override nor builtin rows —
so the degraded bands run for dev and
bench) and the
compiled-in rows pasted manually between the GENERATED markers (the
measurement script only emits the row file; a rebuild picks the rows up).
Every candidate the sweep measures is first probed for the planner's own
decision tag, so a candidate that some gate demotes (ring, operand width,
k-tile depth) is dropped rather than recorded under its own name with the
fallback's numbers.
The sweep times every
combo × recipe at the M × shape grid, each candidate as a one-row
`ASTR_GEMM_TABLE` file toggled per launch: the row source re-reads that env
on every call, so the generator interleaves the candidates at each shape
and the comparison shares one GPU clock/thermal state — sweeping whole
recipe batches in separate processes measured the big CTA first and the
small CTA 30 minutes later, under different boost states, and picked
systematically wrong winners — winners per point become rows,
adjacent M runs with the same winner band-merge at mid-point edges. The
model-retention mode (`--min-gain`, default 1%) rows only leads above a
minimum gain and leaves the band to the degraded rows elsewhere; the
production mode (`--full-coverage`) rows every band, resolves ties by
the stable big>narrow>small preference and appends the per-class
catch-all. K is not a row key (the ring K is fixed at 64): a K/batch
conflict at one (M, N) resolves to the best-tflops point.
`ASTR_GEMM_PLAN` logs the decision source (`table` /
`degraded (no table)`); the row-format details
are exercised indirectly by the correctness suite
(the production route runs through the hooked planner) and by the
smem gate, which turns a stale row into a degraded-band launch instead of
a launch failure. The sweep times the fused-linear (NT) layout, so
generated rows carry crosswise 0 — the table covers the NT path;
non-NT shapes (TT, TN, the mixed dual-row-major NN case) miss into the
degraded bands.

**NN swap.** The dual-N-contiguous problem runs as its transpose
`E = B^T @ A^T` over swapped operands with an out-transposed epilogue
scatter (CUTLASS-sm90 `is_swapAB`): one instantiation fewer per tile
config, at the cost of a scalar-store scatter on a path no LLM-linear
operand pair hits.

#### Quantized GEMM: W8A16 / W8A8 / W16A16 (humming-style)

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

Benchmark (L20, llama weight shapes, `csrc/bench/benchmark_w8.py`,
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

#### Plan table tuning log

The AOT rows in `csrc/kernels/gemm/plan_table.h` are measurements, and this
is where the measurements live: the header keeps each row's rule and the
numbers its dispatch depends on, and points here for the sweeps behind
them. Two things every entry shares — the timings are interleaved A/B in
one process (absolute TFLOPS drift with clock/power, ratios do not), and a
band is only widened where the direction held at **every** swept point
inside it.

**2026-09-10, power-of-2 grid sweep.** Full-coverage sweep over M, N, K in
32..4096 powers of two, all seven dtype combos / four perf classes
(`gen_plan_table.py --full-coverage`), emitted as 42 rows and merged down
to 14 (abutting same-recipe rectangles joined; verified decision-identical
over 80656 probe points x 4 classes x 2 crosswise counts, so the merge
costs nothing at dispatch). The distillate these rows replaced keyed the
recipe on a single N split at 1280, and the measured recipe depends on N
far more strongly than that for large M: it sent M>2560, N>1280 to the
narrow CTA where the big CTA is 1.40x faster (4096x4096x4096 w16a16
101 -> 142 TFLOPS), and M>2560, N<=1280 to the big CTA where the small CTA
is up to 3.6x faster (4096x64x4096 16.5 -> 57 — a 128-wide N tile wastes
half its mma on a 64-column problem). The quantized classes had no rows at
all and fell to the degraded bands. Measured on the 42-row form
(`validate_plan_table.py`, 26 holdout shapes x 6 combos): grid 1.199x, LLM
shape list 1.097x, combined 1.109x, worst per-shape regression 0.84x.
These rows are a **floor, not an optimum**: every one is kK=64 with
`cta<=2` while the manifest carries more. On the narrow-N bands the kK=32
twin of the class they name measures ~1.85x (RTX 4090, 4096x256x4096
w16a16 71 -> 131 TFLOPS), and the warp tiling is not addressable from a
row at all (the dispatch key is class + stages + kK).

**2026-09-10, v2 dense sweep + rectangle search.** The first eight W16A16
rows are measured additions in front of that floor, each required to beat
the row it shadows at every sweep grid point inside its rectangle (min gain
>= 1.00x, geomean >= 1.26x), so a row can only be an improvement.
Rows 1-2 (pc2 wide CTA on the large-N rectangles the class-2 catch-all sent
to the 64x64 tile): 1.34-1.45x (w8a8 4096x11008x4096 304 -> 450 TFLOPS).
Rows 3-7 (pc0 kK=32 on the narrow-N and mid-N bands): 1.31-1.78x geomean,
up to 1.85x (w16a16 2048x512x4096 73.6 -> 136.3 TFLOPS) — the kK=32 ring
is 32KB at s3, under the 48KB two-CTA watermark, which is where the win
comes from; s2 and s3 measure within noise. Row 8 (pc1 big CTA on the
large-M/small-N band the class-1 rows sent to the 64x64 tile): 1.27x.
Validated interleaved against the pre-v2 table (12 holdout shapes x 7
combos): w16a16 -36.9/-39.2/-45.9% on three narrow-N holdouts, no
regression above the ~4% per-shape noise floor.

**2026-09-11, ring-residency rows (RTX 4090).** Two effects, both
properties of the 128x128 kK=32 s2 ring. *Residency*: the kK=64 s2 ring is
96KB, so exactly one CTA is resident per SM and the epilogue — which
scatters the fp32 accumulators through the reclaimed rings and copies the
tile out — is fully exposed; the kK=32 twin's 48KB ring keeps two CTAs
overlapping. *Warps*: that tile runs 16 warps of 32x32 (512 threads,
REG:64 = exactly the two-CTA register budget), against the 8 warps of 64x32
the 128x128 entries carried; doubling the warps per partition is worth
7-14% on large shapes on its own. Interleaved A/B (median of 5 rounds,
event-timed) over the six fused-linear families x M in {512,2048,4096}:
2048x4096x4096 -12.7%, 4096x4096x4096 -14.7%, 512x4096x11008 -11.6%,
4096x4096x11008 -10.7%, 2048x4096x11008 -10.2%, 2048x6144x1536 -8.0%,
512x28672x8192 -8.0%, 512x11008x4096 -8.0%, 512x4096x4096 -6.4%,
4096x1536x1536 -5.3%; every other probed point within +-1.5%. The wide
bands open at M 512 rather than the 1024 the first cut used — a band min is
exclusive, so 1024 left M=1024 (and 768) on the rows below, which the 64x64
CTA served at 0.81-0.89x cuBLAS while their M=2048 neighbours sat at
0.96-1.00x. Extending those rows one octave down measured -12.5 to -15.7%
on the M in {768,1024} points of the six families, with every
already-covered point back within +-0.2%.

**2026-09-11, the narrow-N wave bound.** The bound that says when the
128x128 kK=32 tile beats the 64x64 one was a hand-calibrated M literal
twice, and wrong twice: "past M 3072" (read off M 3072 below the line and
M 4096 above it, assuming the crossover sat at the 256-slot mark), then
"past M 3712" (29 M-tiles, from the N=1536 sweep). Both are the same
arithmetic frozen at one N — the grid is m_tiles * n_tiles, so the M that
fills the machine moves with N, and no single literal holds it. Measured
against the 64x64 kK=32 row underneath, on a 128-SM part (256 slots at this
ring's resident 2), as CTAs / 256:

| shape | grid | waves | winner | TFLOPS |
|---|---|---|---|---|
| 4096x1152 | 32x 9 = 288 | 1.13 | 64x64 | 132.2 vs 107.6 |
| 3328x1536 | 26x12 = 312 | 1.22 | 64x64 | 132.7 vs 116.6 |
| 3584x1536 | 28x12 = 336 | 1.31 | 64x64 | 132.1 vs 125.9 |
| 5120x1152 | 40x 9 = 360 | 1.41 | 128x128 | 133.6 vs 131.0 |
| 3840x1536 | 30x12 = 360 | 1.41 | 128x128 | 133.9 vs 131.1 |
| 4096x1536 | 32x12 = 384 | 1.50 | 128x128 | 141.6 vs 133.8 |

Every N<=1536 point on either side of 1.36 waves agrees, including the one
where the two rules disagree by more than noise (4096x1152: the literal sent
it to the big tile and lost 23%). The 64x64 rate is flat (~132) across the
span, so the crossover is the big tile climbing out of its own
wave-quantization hole: a stub wave of 56-92 CTAs costs more than the tile's
reuse advantage is worth. Hence the row states 1360 permille and drops the M
literal — N=1152 shapes now need 39 M-tiles instead of 29, the dependence
the literal could not express.

**Open: the same rule does not hold across N in (1536,3072].** Four mid-N
points split two against two on it:
3072x2048 (24x16 = 384 = 1.50 waves) -> big wins 141.6 vs 133.5 (+6%);
2688x3072 (21x24 = 504 = 1.97) -> big wins 141.5 vs 137.8;
**2560x2560 (20x20 = 400 = 1.56) -> big loses 113.3 vs 133.6 (-15%)**;
2048x2816 (16x22 = 352 = 1.38) -> big loses 131.2 vs 133.5.
The two rejections are the two *best*-filled grids of the four, so wave
fill is not what decides this band, and nothing about 2560 or 2816 is
special (both divide 128). Until something that does decide it is measured,
that half of the band keeps the N=1536 literal plus a bare CTAs-per-SM
gate, and the 3072x2048 improvement is left on the table. Widening on a 2-2
split would have been a 15% regression on a shape no row was measured on.

**2026-09-11, the two kK=32 band swaps in the small CTA.** N in (768,1536]
resolved to the kK=64 small tile (or, past M 768, to the 128x128 kK=64 one),
whose ring is 64KB/96KB — one resident CTA — while the kK=32 twin is 32KB
and leaves the epilogue overlapped. Above M 1536 the band is left alone
(4096x1536x1536 is served by the big-tile row and the big CTA is 5.3%
ahead), and M<=128 too (the grid is too thin for residency to pay; the
deeper ring amortizes better — 128x1536x4096 is 4% slower on the kK=32
twin). The kK=64 defaults of the bands N in (1536,3072] and N<=768 pay for
their extra ring exactly where the grid is already thin: 1024x3072x1536
-27% (94.7 -> 129.8), 768x768x3072 -37%, 768x512x2048 -23%, 512x768x3072
-20%, 1024x2048x1536 -5%; the M bounds keep those rows off the shapes where
the deeper ring measured ahead (3072x2048x1536 +2% for kK=64, 256x768x3072
+4%).

**The W8A8 wide CTA's win is a grid-fill property, not a shape one.** At
128x256 per CTA the tile needs >= 1 wave to pay off (measured: <0.5 wave
0.79x, >=1 wave 1.08x, >=4 waves 1.42x against the small CTA), so those
bands start where M/128 * N/256 >= 128 — one wave on a 128-SM part, which
is what the resident-1 72KB ring makes a wave. Every wide row also carries
the K > 512 band: the crossover is epilogue dependent, so the band takes
the value that holds under both per-tensor and per-row/per-channel scales
and the row is a strict win rather than one a cheap-scale harness sees as a
regression.

**Open: the M=512 family.** Its grid is thin in a way no tile fixes —
512x1536x1536 is 192 CTAs of 64x64 against 384 slots, and the 128x128 tile
is worse (48 CTAs, 1.13 waves). Split-K would add more reduction traffic
than the lost parallel slack costs (the fp32 partials are larger than the
bf16 output), so the shapes stay on the 64x64 tile at 0.87-0.95x cuBLAS
until a persistent or stream-K kernel makes the grid a free variable.

**2026-09-12, the small-M half of the wide band (N > 3072, K > 2048).**
The band carried one row — big CTA, kK=32 — across its whole M in (0,512]
span, and below M ~384 that was the wrong tile by a wide margin: at M <= 128
its grid is ceil(M/128) x n_tiles, 32 CTAs at N=4096, and a thin grid
streaming B outweighs everything the 128x128 tile is good at. Graph-timed
(host enqueue is 8.4us and the kernel is 30-130us here, so event-timed
Python measures the host) A/B of the shipped row against the 64x64 tile:
M=1 131.6 -> 43.3us, M=64 121.9 -> 33.9 (3.6x), M=128 106.4 -> 36.0,
M=256 107.2 -> 61.8, M=384 108.5 -> 95.8, M=512 109.9 -> 122.3 (big tile
kept). Same ordering on the 28672x8192 family at M=16..256 (1.9x down to
1.04x). The split replaced the single row with three 64x64 rows (s3:
kK=64 above K 4096, kK=32 below) and restored the big tile on (384,512].
Two lessons recorded. The crossover is a **shape** property, not a wave
one — at M 512 the big tile's grid is 128 CTAs, half a wave, and it still
wins, so the wave-permille rule deliberately does not reach this band; the
M bounds are measured. And the kK=64/32 split inside it is calibrated on
one family per side (kk=64 wins the K=8192 small-M points, kk=32 the
K=4096 ones), so the K > 4096 boundary wants a re-measure if another
large-K family shows up. Post-fix, M <= 64 sits at 0.30-0.46x cuBLAS on
the K=1536 families — that residue is **split-K territory, not tile
choice** (cuBLAS splits K there; at M <= 16 the fp32 partial workspace is
< 1MB and the 8x-output traffic argument above does not bind), while
M=64..512 landed at or above parity on the large-N families.

**2026-09-12, what does NOT move the decode band (M=1..128, N>3072).**
Two negative probes before building split-K, both CUDA-graph timed,
interleaved in one process on the post-split table. *Fill*: a 3-D `a`
broadcasts `b` (0 stride), so batch=B puts B copies of the same CTA on the
machine — batch 2 (128 CTAs, one per SM) timed within 8% of batch 1
(64 CTAs) on 4096x4096 and *slower* from there (batch 4 = 63us, batch 8 =
125us vs 48.5us), on every family including the L2-resident ones. Extra
CTAs are free up to one per SM but buy no B throughput: the per-CTA
k-chain is what makes the time. *Ring depth*: s4/s5 and kK=64 variants of
the small-M wide rows (via `ASTR_GEMM_TABLE`) timed flat on all 24
probed shapes — deeper rings add no DRAM in-flight the compiler wasn't
already tracking. Together: the decode deficit vs cuBLAS (46.7 vs 16.9us
at 1x4096x4096) is per-CTA k-chain latency, so the lever is cutting
k-iters per CTA — split-K's actual mechanism — not grid fill or tile
depth. Expectation this sets: split=4 should land ~2.3x (128 iters x
365ns/iter / 4, plus a reduce pass), well short of cuBLAS's gemv-class
kernel; that residue is a skinny-kernel question, not a scheduling one.

**2026-09-12 (later), the k-chain story was wrong: it is the B-read
pattern, and the memory system has two regimes.** A standalone
cp.async-ring microbench at the exact 64x64 staging geometry
(`kstream_bench.cu`, workspace root) swept kK x stages x threads x
CTAs/SM x issue order x per-row run length, plus a linear control with
the identical ring/barrier/wait structure. Three findings. (1) The RTX
4090's DRAM peak is 1008 GB/s — every cold-config (512MB set) hits
0.88-0.96 of it with the shipped wait-first discipline, flat across the
whole sweep; there is nothing to win on DRAM-bound shapes and no
pipeline bug. (2) The decode shapes are NOT DRAM-bound: 4096x4096 bf16
weights are 32MB against 72MB of L2, and cuBLAS's 16.9us (1.9TB/s) is an
L2-resident number. From a warm 64MB set, the tile's interleaved B walk
— 64 rows advancing 128B per round, row pitch K*2 — delivers only
~0.6TB/s, while the linear control reaches 1.7TB/s at 64 CTAs and
3.4TB/s at 128: the multi-stream interleave itself caps L2->SM at a
fifth of what one stream per CTA gets. The shipped kernel's 0.69TB/s at
1x4096x4096 is exactly this cap; issue order (wait-first vs issue-first,
ring one slot deeper) moves it +5%, stage depth and thread count nothing.
(3) Lengthening each row's per-round run (two/four back-to-back kK
windows, 256B/512B) makes it WORSE, monotonically (0.63 -> 0.31 -> 0.08
TB/s from L2, 0.94 -> 0.48 from DRAM) — so no load_stage reshaping inside
the tile pattern recovers it, and with it die: the issue-reorder fix
(+5%), kK=128-deeper-runs (run length is anti-correlated), and split-K /
stream-K (more CTAs on the same interleave — consistent with the flat
batch-fill probe). What survives: for M <= 8, a gemv-class kernel whose
warps walk whole B rows linearly (the 3.4TB/s pattern) with A staged
once per CTA; M in 9..64 is the open gap where tensor cores still matter
but no measured in-tile fix exists. (The per-CTA 11GB/s constant that
motivated the k-chain theory is real but it is the L2-regime interleave
cap, not a latency chain: at DRAM the same code does 7.4GB/s/CTA with
128 CTAs = 0.95TB/s aggregate.)

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

# Or invoke CMake directly
cmake -S csrc -B build/cmake \
  -DTORCH_HOME=<site-packages>/torch \
  -DPYTHON_INCLUDE_DIR=<python include> \
  -DPY_SOABI=cpython-312-x86_64-linux-gnu
cmake --build build/cmake -j 16
```

### Architecture flags

`setup.py` passes the GPU compute capability to CMake via `ASTRAI_CUDA_ARCH`
(a semicolon list, e.g. `"80;89;120"`, produces one multi-arch fatbin). When
unset, `setup.py` auto-detects the real GPU capability through
`torch.cuda.get_device_capability()`; the CMake fallback default is `80` (sm_80):

- **sm_80+** (Ampere and later): enables the tensor-core MMA path
  (`mma.sync.m16n8k16.bf16` for bf16 attention, `mma.sync.m16n8k32` for FP8).
- **sm_89+**: required for the FP8 family (`quantize`) — FP8 tensor-core
  instructions only exist on Ada/Hopper and newer. On older architectures,
  CMake emits a warning and skips the `quantize` target so the remaining CUDA
  kernels still build successfully.
- **`-DASTRAI_NO_MMA`** is a manual escape hatch only — the build never defines
  it automatically. To disable the MMA path, add it to `NVCC_FLAGS` yourself;
  all supported build targets are sm_80+.

### Build configuration

`csrc/CMakeLists.txt` defines the CUDA extension build:

```
NVCC_FLAGS = -O3 --expt-relaxed-constexpr --use_fast_math
             --ptxas-options=-O3,-v --extra-device-vectorization --threads=16
```

Each kernel in `astrai/extension/lib` is compiled as an independent pybind11 module (one `.so` per kernel, named `<kernel>.cpython-*-x86_64-linux-gnu.so`). CMake builds all registered kernel targets in parallel via `cmake --build -j N` (the five base targets always; `quantize` additionally on sm_89+). The target list is the **single source of truth**: `KERNEL_NAMES` and the parallel `KERNEL_SRCS` list in `csrc/CMakeLists.txt`; `astrai/extension/loader.py` auto-discovers the compiled `.so` files.

## Python Extension Architecture

The Python extension package separates low-level kernel bindings from execution
policy:

```text
astrai/extension/
├── __init__.py             # Stable public API
├── loader.py               # Optional compiled-module discovery and loading
├── ops/
│   ├── attention.py        # Stateless attention kernel wrappers
│   ├── rotary.py           # Stateless rotary kernel wrapper
│   └── fp8.py              # Stateless FP8 primitives (custom_op)
├── fp8.py                  # FP8 strategy layer (fp8_autocast, recipes)
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

## Attention Backend

`astrai/extension/backend/attention.py` provides the backend abstraction:

- **`AttentionBackend`** (ABC): single abstract `forward`; each subclass branches on `fwd` ("decode" / "prefill" / None) internally, `_check_fwd` guards unknown modes
- **`CudaBackend`**: CUDA kernel dispatch — decode via `attn_paged_decode` (page_size=1), prefill via `attn_paged_prefill` (ragged batch, `qo_indptr` + `kv_indptr`). Default on GPU.
- **`FlashAttnBackend`**: Optional flash-attn dispatch via `flash_attn_varlen_func` over gathered flat K/V.
- **`TorchNativeBackend`**: SDPA with indirect KV cache gather (always-available fallback)

Default priority: cuda > flash > torch. Set ``ASTR_BACKEND=cuda|torch_native|flash``
to override the default.

Select a backend via context manager (mirrors `torch.nn.attention.sdpa_kernel`):

```python
from astrai.extension import attn_backend, ATTN_BACKEND

with attn_backend(ATTN_BACKEND.CUDA):
    engine.generate("hello")
```

The `attention(...)` policy entry point falls back to `FlashAttnBackend` (when
flash-attn is installed and supports the call) or `TorchNativeBackend` when the
automatically selected CUDA backend cannot handle an input. Resolution
precedence is: explicit `attn_backend(...)` context > `ASTR_BACKEND` env >
default. An explicit `attn_backend(...)` selection is strict and raises instead
of silently switching implementations; the env override (and the implicit
default) fall back to the first compatible backend when incapable. Training
calls (`fwd=None`, no KV cache) resolve by capability: the CUDA cache kernels
cannot run without a cache, so they fall back to flash (mask-free/causal calls
only) and finally to torch SDPA.

### Rotary Backend

`astrai/extension/backend/rotary.py` provides `apply_rotary_emb(x, (cos, sin))` with auto-dispatch:

- **CUDA path**: calls `rotary_emb` kernel directly when available, input is bf16 on CUDA, and `torch.is_grad_enabled()` is `False` (inference)
- **Torch fallback**: complex multiply (`torch.view_as_complex` → `torch.complex` multiply → `torch.view_as_real`), used during training (supports autograd) or when kernel unavailable

No context-manager switching needed — the dispatch is automatic per call.

## Python Wrappers

`astrai/extension/ops/attention.py` provides Python wrappers for each compiled attention kernel. Each wrapper calls its CUDA kernel directly and raises `RuntimeError` if the `.so` is not available. Fallback to torch SDPA is handled by the attention backend, not the wrapper functions.

`astrai/extension/ops/rotary.py` provides the wrapper for the rotary embedding kernel. Fallback to torch complex multiply is handled by `backend/rotary.py`.

Interface (all functions):
```
is_causal: True = causal mask; False = non-causal
mask:      2D [batch, kv_len] or 3D [batch, q_len, kv_len] (bool, True=keep)
```

Layout convention: all q/k/v are `[batch, seq_len, n_heads, head_dim]` (blhd). Scale is always `1/sqrt(head_dim)`.

### Q Scheduling and KV Addressing

Prefill separates Q work scheduling from KV storage:

- `DenseQSchedule` maps a rectangular grid directly with
  `batch = blockIdx.z` and `q_tile = blockIdx.x`.
- `PackedQSchedule` consumes a compact work map for a packed
  `[total_q, q_heads, head_dim]` tensor.
- `ContigKV` and `PagedKV` only provide KV lengths and translate logical KV
  positions into physical addresses. They do not schedule Q blocks.

For ragged Q lengths `[70, 10, 130]` and 64 rows per Q tile, cache binding
builds:

```text
qo_indptr       = [0, 70, 80, 210]
q_tile_to_batch = [0, 0, 1, 2, 2, 2]
q_tile_to_index = [0, 1, 0, 0, 1, 2]
```

Paged prefill launches (MMA path, GQA head packing):

```text
grid.x = num_q_tiles * HB   # HB = min(G, WARPS): q heads packed per block
grid.y = kv_heads * ceil(G / HB)
grid.z = 1
```

The tensor-core prefill kernel packs `HB = min(G, WARPS)` query heads of one
kv-head group into a block, so K/V tiles stream once per block instead of once
per q head (~HB× less global K/V traffic). Warp `w` handles head slot `w / WPH`
and 16-row chunk `w % WPH`, where `WPH = WARPS / HB`; `G = q_heads / kv_heads`
and `G = 1` (MHA) degenerates to the historical one-head-per-block layout.
Each host Q tile (64 rows, `Q_TILE_ROWS`) splits into `HB` packed blocks along
`grid.x`. Each block resolves its request and request-local row range in O(1):

```cpp
host_tile = blockIdx.x / HB;
batch = q_tile_to_batch[host_tile];
row_base = q_tile_to_index[host_tile] * 64 + (blockIdx.x % HB) * (64 / HB);
```

The kernel then uses `qo_indptr[batch]` for the packed Q base and adjacent
`qo_indptr` / `kv_indptr` entries for that request's Q and KV lengths. This
avoids the previous per-block linear scan over the batch, shared-memory
broadcast, mapping barrier, and upper-bound grid with potentially invalid
blocks.

## Standalone Testing

Each `csrc/tests/*.cu` file has the `nvcc` compile command in its header comment. Example:

```bash
nvcc -I csrc/kernels -arch=sm_89 -O3 --use_fast_math \
     --ptxas-options=-O3,-v --extra-device-vectorization \
     -Xcompiler -fopenmp csrc/tests/attn_test.cu -o /tmp/test && /tmp/test
```

Test files:
- `attn_test.cu` — decode + prefill kernels (correctness tables + benchmarks)
- `attn_paged_test.cu` — paged decode/prefill kernels
- `quant_gemm_test.cu` — quantized GEMM correctness: every dtype pair (fp8/int8/bf16 × layouts/K tiles/ragged shapes/scales/fp32 out) + a per-combo TFLOPS bench (sm_89+)
- `plan_table_test.cu` — planner checks: dtype-class keying, per-class table selection (a row cannot leak across classes), the K band's edge cases, and row-file parsing for the 9/10/12-field forms

## Benchmarks

Hardware: NVIDIA L20 (sm_89, 46 GB), CUDA 12.8, driver 570.86.

Reproduce (decode + prefill in `attn_test.cu`, paged in `attn_paged_test.cu`):
```bash
nvcc -I csrc/kernels -arch=sm_89 -O3 --use_fast_math \
     --ptxas-options=-O3,-v --extra-device-vectorization \
     -Xcompiler -fopenmp csrc/tests/attn_test.cu -o /tmp/test && /tmp/test
```

## Known Optimization Targets

- **Decode D=256**: spill eliminated (BC=16 + STAGES=2), but still 248 regs — further tiling could help.
- **Prefill single-batch**: bandwidth low (22 GB/s at q=kv=2048) — compute-bound at ~94 TFLOP/s (near L20 bf16 ceiling ~193 TFLOP/s for non-causal).
- **Decode single-batch**: bandwidth low (113 GB/s at kv=512, 13% of 864 GB/s theoretical) — small kv underutilizes SMs despite split-KV; scales to 757 GB/s (88%) at B=16+.

## File Layout

```
csrc/
├── CMakeLists.txt                    # CMake build: kernel registry (KERNEL_NAMES / KERNEL_SRCS), torch/pybind11 linking
├── kernels/
│   ├── common/                       # cross-family pure-CUDA helpers (no torch)
│   │   ├── device.cuh                #   DeviceFacts geometry query (sms / smem opt-in / L2); fp8 capability helpers live in quantize/common.h, the torch-bound gate in quantize/checks.h
│   │   ├── mma.cuh                   #   shared mma_sync<InT> + mma_shape<InT> (bf16 m16n8k16 / fp8 m16n8k32) + ldmatrix_x2<T> and the per-lane ldmatrix cores + typed fragment cells (AFrag/BFrag/CFrag, by-reference fma/ldmatrix overloads)
│   │   ├── pipeline.cuh              #   async data-movement vocabulary, one header: raw cp.async 16B emitters (fixed + runtime-src-size zfill) and the mbarrier PTX sites, plus the PipelineSync (sm_80/89 wait_group+syncthreads) stage pipeline
│   │   ├── swizzle.cuh               #   staging-layout vocabulary: Swizzle/Shape/Stride/Layout + composition(Swizzle, Layout) in 16B-chunk units; per-tile SmemLayout types declared by the gemm collectives
│   │   ├── tensor.cuh                #   tensor vocabulary, cute's Tensor<Engine, Layout>: PtrEngine/ArrayEngine, RingLayout/CellLayout, one Tensor type spelled directly (make_ring constructs the stage ring, stage_of slices a slot)
│   │   └── reduce.cuh                #   warp_reduce_max, atomic_max_float
│   ├── attention/                    # attention family (module names keep the attn_* prefix)
│   │   ├── common.h                  #   AttentionParams POD, TensorLayout enum (BHLD/BLHD)
│   │   ├── warp_utils.cuh            #   warp reduction helpers
│   │   ├── layout_policies.cuh       #   KV addressing policies: DenseQSchedule/PackedQSchedule, ContigKV/PagedKV
│   │   ├── mma_utils.cuh             #   ldmatrix/pack helpers + online-softmax (bf16 mma via common/mma.cuh)
│   │   ├── entry_utils.cuh           #   torch binding helpers: DISPATCH_HEAD_DIM, pack_*_params
│   │   ├── dispatchers.cuh           #   pure-CUDA launchers: dispatch_decode/prefill(_impl) funnel (+paged), split-K math
│   │   ├── decode_split_kv.cuh       #   decode kernel, scalar (split-KV)
│   │   ├── decode_split_kv_mma.cuh   #   decode kernel, MMA + split-K
│   │   ├── prefill_split_q.cuh       #   prefill kernel, scalar (split-Q)
│   │   ├── prefill_split_q_mma.cuh   #   prefill kernel, MMA (split-Q, GQA head packing, packed/ragged Q schedule)
│   │   ├── decode.cu                 #   → module attn_decode
│   │   ├── prefill.cu                #   → module attn_prefill
│   │   ├── paged_decode.cu           #   → module attn_paged_decode
│   │   └── paged_prefill.cu          #   → module attn_paged_prefill
│   ├── rotary_emb.cu                  # rotary embedding (kernel + binding in one file) → module rotary_emb
│   ├── quantize/                        # quantize family (pure CUDA; checks.h is the torch-bound gate)
│   │   ├── common.h                  #   sm_at_least + kMinSmForFp8 capability helpers, QuantLayout, QuantParams POD (raw __nv_fp8_* types)
│   │   ├── checks.h                  #   torch-bound entry validation (check_fp8_device over ATen-cached properties)
│   │   ├── dequant.cuh               #   in-register dequant functors (DequantPair<SrcT, MmaT>: exact int8→bf16)
│   │   └── quantize.cuh              #   quantize kernels: vectorized + 64×32-tile transpose (out_layout 0/1/2, Dual as a template param)
│   ├── gemm/                         # GEMM family, dtype-neutral (→ module gemm)
│   │   ├── common.h                  #   layout tags, gemm_elem_traits<T>, gemm_mma_traits (MmaT promotion), GemmParams POD (no torch)
│   │   ├── gemm.cuh                  #   GEMM umbrella: kernel orchestrator + host launch planning (no torch)
│   │   ├── policy.cuh                #     Shape/TileConfig tile recipes + smem budget + GemmPolicy + TileManifest (+ kTileClassCta)
│   │   ├── plan_table.h              #     AOT dispatch rows (TableRow): override/per-class builtin/degraded row sources
│   │   ├── load.cuh                  #     operand loaders (typed staged tiles over declared layouts, congruous cp.async + zfill, crosswise direct, trans staging, PrefetchCarry)
│   │   ├── scheduler.cuh             #     grouped/plain raster mapping
│   │   ├── mainloop.cuh              #     stage rings + pipelined mma.sync mainloop (+ dequantized fragment paths)
│   │   ├── epilogue.cuh              #     fused bias + scale folding + bf16/fp32 smem scatter + copy-out
│   │   ├── gemm_bf16_* / gemm_*.cu   #     per-pair explicit gemm_dispatch instantiation units (one nvcc job each)
│   │   └── gemm.cu                   #   quant_gemm binding + dtype-pair switch (module gemm; extern template decls)
│   └── quantize/quantize.cu                  #   binding only (module quantize): validation, param packing, launch dispatch, pybind
└── tests/
    ├── test_utils.cuh                # Shared test utilities (now_ms, f2bf, bf2f, randf)
    ├── attn_test.cu                  # Decode + prefill kernels
    ├── attn_paged_test.cu            # Paged decode/prefill kernels
    ├── plan_table_test.cu            # Planner: class keying, per-class tables, K band, row-file forms
    └── quant_gemm_test.cu           # GEMM correctness: fp8/bf16/int8 pairs across layouts/K tiles/ragged shapes + dtype-combo TFLOPS bench
```

Compiled `.so` files are placed in `astrai/extension/lib/`, separate from Python source files.

> Document Update Time: 2026-09-10
