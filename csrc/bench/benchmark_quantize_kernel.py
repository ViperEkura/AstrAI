"""ncu target for the fp8 quantize kernels (versioned home of /tmp/qmerged_target.py).

Emits the three production modes at the canonical shapes so ncu can judge
them under the only trustworthy metric on this box (see
notes/quantize-kernel-ncu-2026-09-19.md): cold-cache duration with pinned
clocks. Round 1 warms; ncu profiles round 2 via --launch-skip.

    CUDA_VISIBLE_DEVICES=0 ncu --clock-control base --cache-control all \
        --kernel-name regex:fp8_quantize --launch-skip 3 --launch-count 3 \
        --metrics gpu__time_duration.sum,launch__registers_per_thread \
        .venv/bin/python csrc/bench/benchmark_quantize_kernel.py --mode all

Single-mode A/B: --mode rm|t|dual (one launch per round, --launch-skip 1).
Wall-clock timings here are sanity only — L2 residency makes hot loops lie.
"""

import argparse

import torch

from astrai.extension.ops.quantize import quantize, quantize_dual
from astrai.extension.quantize import fp8_autocast

# astrai_1b pretrain projections: x is the m=16384 activation against
# hidden=1536; g is the backward gradient against qkv's 6144 outputs.
X_SHAPE = (16384, 1536)
G_SHAPE = (16384, 6144)
HIST_LEN = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["rm", "t", "dual", "all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x = torch.randn(*X_SHAPE, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(*G_SHAPE, device="cuda", dtype=torch.bfloat16)
    s4 = torch.tensor(1.0, device="cuda")
    state = torch.zeros(HIST_LEN + 4 + 32, device="cuda")
    fold = dict(ring_state=state, hist_idx=0, fp8_max=448.0, pow2_margin=1.0)

    def run_round() -> None:
        if args.mode in ("rm", "all"):
            quantize(x, s4, torch.float8_e4m3fn, **fold)
        if args.mode in ("t", "all"):
            quantize(x, s4, torch.float8_e5m2, transposed=True)
        if args.mode in ("dual", "all"):
            quantize_dual(g, s4, torch.float8_e5m2, **fold)

    with fp8_autocast(enabled=True):
        for _ in range(args.rounds):
            run_round()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
