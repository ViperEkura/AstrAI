import os
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

sys.path.insert(0, str(Path(__file__).parent))
os.makedirs("astrai/extension", exist_ok=True)


def _should_build():
    force = os.environ.get("CSRC_KERNELS", "").strip().lower()
    if force == "true":
        return True
    if force == "false":
        return False
    try:
        import shutil

        import torch

        return shutil.which("nvcc") is not None and torch.cuda.is_available()
    except Exception:
        return False


ext_modules = []
cmdclass = {}

if _should_build():
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    from csrc.build import REGISTRY

    _torch_lib = torch.utils.cpp_extension.library_paths()[0]

    for name, info in REGISTRY.items():
        ext_modules.append(
            CUDAExtension(
                f"astrai.extension.{name}",
                info["sources"],
                extra_compile_args={
                    "cxx": info["cxx_flags"],
                    "nvcc": info["nvcc_flags"],
                },
                extra_link_args=[f"-Wl,-rpath,{_torch_lib}"],
            )
        )
    cmdclass["build_ext"] = BuildExtension

if not cmdclass:

    class _NullBuildExt(_build_ext):
        def build_extensions(self):
            pass

    cmdclass["build_ext"] = _NullBuildExt

setup(ext_modules=ext_modules, cmdclass=cmdclass)
