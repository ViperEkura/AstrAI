# Quantize (FP8)

> Kernel modules `csrc/kernels/quantize/`; python adapters in
> `astrai/extension/ops/quantize.py`, strategy layer (recipes,
> `fp8_autocast`, aten::linear override) in
> `astrai/extension/quantize.py`.

## FP8 training dataflow (math contract)

One linear layer, one optimizer step. Master weights, gradients and
optimizer state stay bf16/fp32 — FP8 exists only inside the GEMM operands
and is never persisted. `FP8_max` is 448 for e4m3, 57344 for e5m2.

**Scaled cast** (`quantize` / `quantize_dual`). With `amax = max|x_i|`
(clamped >= 1e-12) and the tensor scale `s = amax / FP8_max / 2^margin`:

    x8_i = satfinite_rn(x_i * s^-1)      # e4m3 forward, e5m2 gradients (hybrid)

The kernel is handed the *multiplier* `s^-1` and saturates at the format's
range. Delayed scaling derives `s` from the amax history window (max over
the last `history_len` entries, default 16): the quantize kernel folds the
current amax into the ring (`[hist n | scale | recip | amax | done | 32-slot
scratch]`, fp32, plus the composed ring's trailing double-buffered
`scale | recip` pair) and publishes the next step's scale *and its
reciprocal* (`__frcp_rn`, bit-identical to the host's `1/x`) in its own last
block — one tensor read per step, and the policy hands the published recip
straight to the next call instead of materializing one. The same block
reports that round's amax on the ring's `amax` slot, and the window length is
an explicit argument (`hist_len`): it is not recoverable from the buffer's
`numel`. Dynamic scaling measures the current amax instead (two reads, no
window, exact on the first step).
`quantize_dual(g)` yields `(g8, g8T)` in a single read: the training path
consumes the gradient in both orientations (dgrad takes `g`, wgrad takes
`g^T`). No-grad calls read the rings without folding or advancing
(checkpointing recompute stays numerically identical).

**Forward** (layout NT — both operands K-major, the only orientation that
reaches full tensor-core rate; crosswise layouts land at 230-380 TFLOPS):

    out = bf16( (x8 @ w8^T) * sx*sw + bias )

fp32 accumulation in the mma; the dequant scales and the bias fold into the
epilogue (fp32 add, one bf16 rounding at the end). `w8`/`w8T` are cached per
weight version: the optimizer's in-place update bumps the version counter, so
one cast serves a whole gradient-accumulation loop and the ring folds once
per optimizer step rather than once per micro-batch (an unchanged weight has
an unchanged amax — the fold would republish the same scale).

**Backward** (both GEMMs NT via transposed quantized copies; the gradient
is quantized once and shared):

    grad_x = bf16( (g8  @ w8T) * sg*sw )   # dgrad, B = transposed weight copy
    grad_w = bf16( (g8T @ x8T) * sg*sx )   # wgrad, both operands transposed
    grad_b = bf16( sum_rows(g) )           # plain reduce, never fp8

The transposed operands are cast by the *forward* (the weight pair rides the
version cache; under a symmetric format pair one dual pass yields `x8T` too)
and saved on the autograd node — the backward re-reads neither `x` nor `w`.
Both sides of a backward GEMM must share one format, so under hybrid
(E4M3 fwd / E5M2 bwd) the casts are E5M2 and `x8T` is left to the backward.

**Scale extents** (`resolve_quant_scale`): `a_scale` holds 1 or `m` floats
and `b_scale` 1 or `n` — per-row/per-channel factors run along the *output*
dimension only. Per-channel weight scaling is therefore a forward-only
option (in dgrad the weight plays B and its output dim is K); the training
path uses per-tensor for every operand.

**sm_120 mma cell.** Production symmetric-fp8 GEMMs dispatch through
`mma.sync...kind::mxf8f6f4.block_scale.scale_vec::1X` with constant unit
scales (ue8m0 bytes 0x7f = 2^0, selectors inert — the scale-factored product
is the plain product): the plain fp8 `mma.sync` decodes at half rate on
consumer Blackwell (measured 506 vs 1011 TFLOPS issue rate), while the
block_scale form runs full rate with the same register contract. The cell
gates on cc == 120 (kill switch `ops.gemm.set_staging(mx=False)`); its inert scale
operands are also the seam where real per-block (MX) scaling would attach.

