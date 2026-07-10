from pathlib import Path


def _arch_flags() -> list[str]:
    import torch

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
    else:
        cap = (8, 0)
    ver = f"{cap[0]}{cap[1]}"
    flags = [f"-gencode=arch=compute_{ver},code=sm_{ver}"]
    # tensor-core mma path (mma.sync.m16n8k16.bf16) requires sm_80+; decide the
    # kernel dispatch at build time via this define rather than at runtime.
    if cap[0] < 8:
        flags.append("-DASTRAI_NO_MMA")
    return flags


_kernels_dir = Path("csrc/kernels")
REGISTRY: dict[str, dict] = {}

CXX_FLAGS = ["-O3", "-march=native", "-funroll-loops"]
NVCC_FLAGS = [
    "-O3",
    "--expt-relaxed-constexpr",
    "--use_fast_math",
    "--ptxas-options=-O3,-v",
    "--extra-device-vectorization",
]


def register(name: str, sources: list[str] | None = None, **kwargs):
    if sources is None:
        sources = [str(_kernels_dir / f"{name}.cu")]
    REGISTRY[name] = {
        "sources": sources,
        "cxx_flags": [*CXX_FLAGS],
        "nvcc_flags": [*NVCC_FLAGS, *_arch_flags()],
        "extra_link_args": kwargs.pop("extra_link_args", []),
        **kwargs,
    }


register("attn_decode")
register("attn_prefill")
