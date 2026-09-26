"""Post-relayout verification of the include-resolution snapshot.

Companion to scripts/csrc_include_resolve.py. The snapshot was keyed by the
OLD path of each source file; after the stage-tree move the same file lives
elsewhere. This script remaps the keys through the relayout's move table,
re-resolves each site (angle-bracket spellings, root csrc/include), and
asserts every site still resolves to the same concrete file it did before
the move — the guard against a spelling change silently retargeting an
include through a same-named header.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc"
INCLUDE = CSRC / "include"
TESTS = CSRC / "tests"

INCLUDE_RE = re.compile(r'^\s*#\s*include\s+([<"])(?P<name>[^">]+)[">]')

# Old csrc/-relative path -> new csrc/-relative path (the relayout's map).
MOVES: dict[str, str] = {
    "kernels/common/device.cuh": "include/utils/device.cuh",
    "kernels/common/dtype.cuh": "include/utils/dtype.cuh",
    "kernels/common/launch.cuh": "include/utils/launch.cuh",
    "kernels/common/shape.cuh": "include/utils/shape.cuh",
    "kernels/common/swizzle.cuh": "include/utils/swizzle.cuh",
    "kernels/common/tensor.cuh": "include/utils/tensor.cuh",
    "kernels/common/mma.cuh": "include/mma/mma.cuh",
    "kernels/common/pipeline.cuh": "include/memory/pipeline.cuh",
    "kernels/common/reduce.cuh": "include/arith/reduce.cuh",
    "kernels/common/tma.cuh": "include/memory/tma.cuh",
    "kernels/attention/common.h": "include/utils/attention_common.h",
    "kernels/attention/dispatchers.cuh": "include/kernel/attention_dispatch.cuh",
    "kernels/attention/dtype_list.cuh": "include/launcher/dtype_list.h",
    "kernels/attention/entry_utils.cuh": "include/launcher/entry_utils.h",
    "kernels/attention/layout_policies.cuh": "include/memory/layout_policies.cuh",
    "kernels/attention/mma_utils.cuh": "include/mma/utils.cuh",
    "kernels/attention/softmax.cuh": "include/arith/softmax.cuh",
    "kernels/attention/decode_split_kv.cuh": "include/kernel/attention_decode_split_kv.cuh",
    "kernels/attention/decode_split_kv_mma.cuh": "include/kernel/attention_decode_split_kv_mma.cuh",
    "kernels/attention/prefill_split_q.cuh": "include/kernel/attention_prefill_split_q.cuh",
    "kernels/attention/prefill_split_q_mma.cuh": "include/kernel/attention_prefill_split_q_mma.cuh",
    "kernels/attention/decode.cu": "attention/decode.cu",
    "kernels/attention/prefill.cu": "attention/prefill.cu",
    "kernels/attention/paged_decode.cu": "attention/paged_decode.cu",
    "kernels/attention/paged_prefill.cu": "attention/paged_prefill.cu",
    "kernels/gemm/common.h": "include/utils/gemm_common.h",
    "kernels/gemm/api.h": "include/launcher/api.h",
    "kernels/gemm/gemm.cuh": "include/kernel/gemm.cuh",
    "kernels/gemm/policy.cuh": "include/policy.cuh",
    "kernels/gemm/plan_table.h": "include/launcher/plan_table.h",
    "kernels/gemm/load.cuh": "include/memory/load.cuh",
    "kernels/gemm/scheduler.cuh": "include/scheduler.cuh",
    "kernels/gemm/mainloop.cuh": "include/kernel/gemm_mainloop.cuh",
    "kernels/gemm/epilogue.cuh": "include/epilogue/writer.cuh",
    "kernels/gemm/fp8_state.cuh": "include/launcher/fp8_state.h",
    "kernels/gemm/gemm.cu": "gemm/gemm.cu",
    "kernels/gemm/bindings.cu": "gemm/bindings.cu",
    "kernels/gemm/fp8_linear.cu": "gemm/fp8_linear.cu",
    "kernels/gemm/gemm_bf16_bf16.cu": "gemm/gemm_bf16_bf16.cu",
    "kernels/gemm/gemm_bf16_int8.cu": "gemm/gemm_bf16_int8.cu",
    "kernels/gemm/gemm_int8_int8.cu": "gemm/gemm_int8_int8.cu",
    "kernels/gemm/gemm_fp8_e4m3_fp8_e4m3.cu": "gemm/gemm_fp8_e4m3_fp8_e4m3.cu",
    "kernels/gemm/gemm_fp8_e5m2_fp8_e5m2.cu": "gemm/gemm_fp8_e5m2_fp8_e5m2.cu",
    "kernels/quantize/common.h": "include/utils/quantize_common.h",
    "kernels/quantize/checks.h": "include/launcher/checks.h",
    "kernels/quantize/dequant.cuh": "include/datatype/dequant.cuh",
    "kernels/quantize/launch.cuh": "include/launcher/launch.h",
    "kernels/quantize/quantize.cuh": "include/kernel/quantize.cuh",
    "kernels/quantize/quantize.cu": "quantize/quantize.cu",
    "kernels/gated_deltanet/gated_deltanet.h": "include/launcher/gated_deltanet.h",
    "kernels/gated_deltanet/gated_deltanet_fwd.cu": "gated_deltanet/gated_deltanet_fwd.cu",
    "kernels/gated_deltanet/gated_deltanet_bwd.cu": "gated_deltanet/gated_deltanet_bwd.cu",
    "kernels/gated_deltanet/bindings.cu": "gated_deltanet/bindings.cu",
    "kernels/rotary_emb.cu": "rotary_emb.cu",
}

# old spelling -> new spelling (what the relayout rewrote the line to)
SPELLINGS: dict[str, str] = {
    "common/device.cuh": "utils/device.cuh",
    "common/dtype.cuh": "utils/dtype.cuh",
    "common/launch.cuh": "utils/launch.cuh",
    "common/shape.cuh": "utils/shape.cuh",
    "common/swizzle.cuh": "utils/swizzle.cuh",
    "common/tensor.cuh": "utils/tensor.cuh",
    "common/mma.cuh": "mma/mma.cuh",
    "common/pipeline.cuh": "memory/pipeline.cuh",
    "common/reduce.cuh": "arith/reduce.cuh",
    "common/tma.cuh": "memory/tma.cuh",
    "attention/common.h": "utils/attention_common.h",
    "attention/dispatchers.cuh": "kernel/attention_dispatch.cuh",
    "attention/dtype_list.cuh": "launcher/dtype_list.h",
    "attention/entry_utils.cuh": "launcher/entry_utils.h",
    "attention/layout_policies.cuh": "memory/layout_policies.cuh",
    "attention/mma_utils.cuh": "mma/utils.cuh",
    "attention/softmax.cuh": "arith/softmax.cuh",
    "attention/decode_split_kv.cuh": "kernel/attention_decode_split_kv.cuh",
    "attention/decode_split_kv_mma.cuh": "kernel/attention_decode_split_kv_mma.cuh",
    "attention/prefill_split_q.cuh": "kernel/attention_prefill_split_q.cuh",
    "attention/prefill_split_q_mma.cuh": "kernel/attention_prefill_split_q_mma.cuh",
    "gemm/common.h": "utils/gemm_common.h",
    "gemm/api.h": "launcher/api.h",
    "gemm/gemm.cuh": "kernel/gemm.cuh",
    "gemm/policy.cuh": "policy.cuh",
    "gemm/plan_table.h": "launcher/plan_table.h",
    "gemm/load.cuh": "memory/load.cuh",
    "gemm/scheduler.cuh": "scheduler.cuh",
    "gemm/mainloop.cuh": "kernel/gemm_mainloop.cuh",
    "gemm/epilogue.cuh": "epilogue/writer.cuh",
    "gemm/fp8_state.cuh": "launcher/fp8_state.h",
    "quantize/common.h": "utils/quantize_common.h",
    "quantize/checks.h": "launcher/checks.h",
    "quantize/dequant.cuh": "datatype/dequant.cuh",
    "quantize/launch.cuh": "launcher/launch.h",
    "quantize/quantize.cuh": "kernel/quantize.cuh",
    "gated_deltanet/gated_deltanet.h": "launcher/gated_deltanet.h",
}

HARNESS_LOCAL = {"test_utils.cuh"}


def remap_rel(path_rel: str) -> str:
    if path_rel in MOVES:
        return MOVES[path_rel]
    # tests/ and bench/ never moved
    return path_rel


def resolve_new(name: str, src_dir: Path, in_bench: bool) -> list[Path]:
    """Search order after the relayout: own dir (quoted only, but we keep it
    for test_utils.cuh), then csrc/include, then csrc/tests (bench only)."""
    roots = [src_dir, INCLUDE]
    if in_bench:
        roots.append(TESTS)
    found = []
    for root in roots:
        candidate = root / name
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved not in found:
                found.append(resolved)
    return found


def main() -> int:
    snapshot_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/inc-snapshot.json")
    old = json.loads(snapshot_path.read_text())
    bad = []
    checked = 0
    for key, entry in old.items():
        src_rel, lineno = key.rsplit(":", 1)
        src_rel = src_rel.removeprefix("csrc/")
        new_src_rel = remap_rel(src_rel)
        src = CSRC / new_src_rel
        if not src.exists():
            bad.append(f"{key}: moved source missing: {new_src_rel}")
            continue
        new_spelling = SPELLINGS.get(entry["spelling"], entry["spelling"])
        in_bench = "bench" in src.relative_to(CSRC).parts
        if new_spelling in HARNESS_LOCAL:
            # harness-local: quoted, own dir first, then csrc/tests (the
            # bench harness reaches it through its extra -I)
            targets = [
                str(p.relative_to(ROOT))
                for p in resolve_new(new_spelling, src.parent, in_bench)
            ]
        else:
            targets = [
                str(p.relative_to(ROOT))
                for p in resolve_new(new_spelling, src.parent, in_bench)
            ]
        # Normalize the old targets through the move map too.
        old_targets = []
        for t in entry["targets"]:
            t_rel = t.removeprefix("csrc/")
            old_targets.append(
                str((CSRC / MOVES.get(t_rel, t_rel)).resolve().relative_to(ROOT))
            )
        checked += 1
        if targets != old_targets:
            bad.append(
                f"{key}: {entry['spelling']!r} resolved to {old_targets} before, "
                f"now {new_spelling!r} -> {targets}"
            )
    if bad:
        print("RESOLUTION DRIFT:", file=sys.stderr)
        for line in bad:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"verify: {checked}/{len(old)} sites remapped, all resolve to the same files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
