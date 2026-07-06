from pathlib import Path


def _arch_flag():
    import torch

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        ver = f"{cap[0]}{cap[1]}"
        return f"-gencode=arch=compute_{ver},code=sm_{ver}"
    return "-gencode=arch=compute_80,code=sm_80"


_kernels_dir = Path("csrc/kernels")
REGISTRY: dict[str, dict] = {}


def register(name: str, sources: list[str] | None = None, **kwargs):
    if sources is None:
        sources = [str(_kernels_dir / f"{name}.cu")]
    REGISTRY[name] = {
        "sources": sources,
        "nvcc_flags": ["-O3", "--expt-relaxed-constexpr", _arch_flag()],
        "extra_link_args": kwargs.pop("extra_link_args", []),
        **kwargs,
    }


register("gqa_decode_attn")
register("gqa_prefill_attn")
