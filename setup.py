import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

from setuptools import setup
from setuptools.command.build import build as _build
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.editable_wheel import editable_wheel as _editable_wheel

sys.path.insert(0, str(Path(__file__).parent))
os.makedirs("astrai/extension/lib", exist_ok=True)


def _should_build():
    force = os.environ.get("CSRC_KERNELS", "").strip().lower()
    if force == "true":
        return True
    if force == "false":
        return False
    try:
        import torch

        return shutil.which("nvcc") is not None and torch.cuda.is_available()
    except Exception:
        return False


def _torch_prefix():
    """Return the torch install dir (site-packages/torch) used for headers/libs."""
    try:
        import torch

        return str(Path(torch.__file__).parent.resolve())
    except Exception:
        return os.environ.get("TORCH_HOME", "")


def _python_include():
    import sysconfig

    return sysconfig.get_path("include")


def _python_soabi():
    import sysconfig

    ext = sysconfig.get_config_var("EXT_SUFFIX").lstrip(".")
    return ext[: -len(".so")]


class _CMakeBuildExt(_build_ext):
    def run(self):
        src = Path(__file__).parent
        build_dir = src / "build" / "cmake"
        torch_home = _torch_prefix()
        if not torch_home:
            raise RuntimeError(
                "torch not found; cannot build kernels. "
                "Activate the environment or set TORCH_HOME."
            )

        nvcc_ver = _cuda_toolkit_version()
        torch_cuda = _torch_cuda_version()
        if (
            nvcc_ver is not None
            and torch_cuda is not None
            and nvcc_ver[0] != int(torch_cuda.split(".")[0])
        ):
            warnings.warn(
                f"CUDA version mismatch: nvcc is {nvcc_ver[0]}.{nvcc_ver[1]} "
                f"but torch was built with CUDA {torch_cuda}. "
                f"Install a matching torch wheel.",
                stacklevel=2,
            )

        cmake = shutil.which("cmake")
        if cmake is None:
            raise RuntimeError("cmake not found on PATH; install it to build kernels")

        # Job-level parallelism: one nvcc job per compile unit x arch pass;
        # each job already pools its own ptxas via --threads. Default fills
        # the box (capped), BUILD_PARALLEL overrides.
        parallel = os.environ.get("BUILD_PARALLEL", str(min(os.cpu_count() or 4, 32)))
        cfg = [
            cmake,
            "-S",
            str(src / "csrc"),
            "-B",
            str(build_dir),
            f"-DTORCH_HOME={torch_home}",
            f"-DPYTHON_INCLUDE_DIR={_python_include()}",
            f"-DPY_SOABI={_python_soabi()}",
        ]
        arch = os.environ.get("ASTRAI_CUDA_ARCH")
        max_arch = None
        try:
            if arch:
                # Accept a semicolon list ("80;89;120a"); the FP8 gate keys
                # on the maximum, matching the CMake-side validation. The
                # arch-specific 'a' suffix selects a compile pass, never a
                # CC number, so the comparison strips it.
                max_arch = max(
                    int(re.sub(r"[^0-9]", "", a))
                    for a in arch.split(";")
                    if re.sub(r"[^0-9]", "", a)
                )
            else:
                # No env: follow the local GPU (dev iteration — one arch,
                # arch-specific where it matters). The mixed fleet is the
                # explicit release opt-in (ASTRAI_CUDA_ARCH="80;89;120a").
                # With no GPU either, CMake builds no kernel targets at all
                # (there is no arch fallback), so this path may pass None.
                arch = _detect_cuda_arch()
                max_arch = int(re.sub(r"[^0-9]", "", arch)) if arch else None
        except ValueError:
            warnings.warn(
                f"Could not parse ASTRAI_CUDA_ARCH={arch!r}; "
                "FP8 capability will be decided by CMake.",
                stacklevel=2,
            )
        if arch:
            if max_arch is not None and max_arch < 89:
                warnings.warn(
                    f"FP8 operator disabled: CUDA compute capability {arch} "
                    "requires 89 or newer.",
                    stacklevel=2,
                )
            cfg.append(f"-DASTRAI_CUDA_ARCH={arch}")
        # CUDACXX: a cold-cache configure cannot find nvcc when it is off
        # PATH — and caches the failure.
        env = dict(os.environ)
        if (nvcc := _find_nvcc()) and not env.get("CUDACXX"):
            env["CUDACXX"] = nvcc
        subprocess.run(cfg, check=True, env=env)
        subprocess.run(
            [cmake, "--build", str(build_dir), "-j", parallel], check=True, env=env
        )

        # After compilation finishes, verify mandatory CUDA kernels to confirm build succeeded.
        # CMake may report partial‑target success even if some architecture‑specific kernels are skipped.
        # Prevent editable install from reporting success when critical kernel shared objects are missing.
        lib_dir = src / "astrai" / "extension" / "lib"
        required = (
            "attn_decode",
            "attn_prefill",
            "attn_paged_decode",
            "attn_paged_prefill",
            "rotary_emb",
        )
        missing = [name for name in required if not any(lib_dir.glob(f"{name}.*.so"))]
        if missing:
            raise RuntimeError(
                "CUDA build completed without some required kernel modules!"
            )


def _find_nvcc():
    """nvcc path: PATH first, then the standard toolkit locations."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    for root in (
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
    ):
        if root and (Path(root) / "bin" / "nvcc").is_file():
            return str(Path(root) / "bin" / "nvcc")
    return None


def _cuda_toolkit_version():
    import subprocess

    nvcc = _find_nvcc()
    if nvcc is None:
        return None
    try:
        out = subprocess.check_output(
            [nvcc, "--version"], stderr=subprocess.STDOUT, text=True
        )
        for line in out.splitlines():
            if "release" in line:
                ver = line.split("release")[1].split(",")[0].strip()
                return tuple(int(x) for x in ver.split("."))
    except Exception:
        pass
    return None


def _detect_cuda_arch():
    """Detect real GPU compute capability via torch (nvidia-smi may be spoofed).

    Returns a token like ``"89"``, ``"103"`` or ``"120a"`` (``None`` if
    unavailable). CC 12.0 gains the arch-specific suffix: the gemm's fp8
    block_scale cell lives only in that pass, and an sm_120 device selects
    the arch-specific image anyway.
    """
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}{minor}"
            if arch == "120":
                arch += "a"
            return arch
    except Exception:
        pass
    return None


def _torch_cuda_version():
    try:
        import torch

        return torch.version.cuda
    except Exception:
        return None


class _NullBuildExt(_build_ext):
    def build_extensions(self):
        pass


class _Build(_build):
    """Run the CMake kernel build as part of setuptools' build lifecycle."""

    def run(self):
        if _should_build():
            self.run_command("build_ext")
        super().run()


class _EditableWheel(_editable_wheel):
    """Run the CMake kernel build for PEP 660 editable installations."""

    def run(self):
        if _should_build():
            self.run_command("build_ext")
        super().run()


cmdclass = {}

if _should_build():
    cmdclass["build_ext"] = _CMakeBuildExt
else:
    cmdclass["build_ext"] = _NullBuildExt

cmdclass["build"] = _Build
cmdclass["editable_wheel"] = _EditableWheel

setup(
    ext_modules=[],
    cmdclass=cmdclass,
)
