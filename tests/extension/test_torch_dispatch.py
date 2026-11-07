"""Compile and exercise shared dtype visitors without CUDA launches."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def cpp_command():
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a C++17 compiler")
    candidates = []
    if os.environ.get("CUDA_HOME"):
        candidates.append(Path(os.environ["CUDA_HOME"]))
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(Path(nvcc).resolve().parents[1])
    candidates.extend([Path("/usr/local/cuda"), Path("/usr/local/cuda-13.0")])
    cuda_include = next(
        (
            home / "include"
            for home in candidates
            if (home / "include/cuda_fp8.h").is_file()
        ),
        None,
    )
    if cuda_include is None:
        pytest.skip("requires CUDA element headers, but no GPU or CUDA libraries")
    return [
        compiler,
        "-std=c++17",
        "-O0",
        "-I",
        str(ROOT / "csrc/include"),
        "-I",
        str(Path(torch.__file__).parent / "include"),
        "-I",
        str(cuda_include),
    ]


@pytest.mark.parametrize(
    "body,diagnostic",
    [
        (
            "using L = TypeList<bf16, bf16>; static_assert(dtype_supported<L>(at::kBFloat16));",
            "duplicate dtype dispatch key",
        ),
        (
            "using P = DTypePair<bf16, int8_t>; using L = TypeList<P, P>; "
            "static_assert(dtype_pair_supported<L>(at::kBFloat16, at::kChar));",
            "duplicate dtype dispatch key",
        ),
        (
            'visit_dtype<TypeList<>>(at::kFloat, [](auto) {}, [](auto) { return "missing"; });',
            "dtype visitor requires a nonempty type list",
        ),
        (
            "visit_dtype<TypeList<bf16, float>>(at::kFloat, [](auto tag) { "
            "using T = typename decltype(tag)::Type; "
            "if constexpr (std::is_same_v<T, bf16>) return 1; else return 1.0; "
            '}, [](auto) { return "missing"; });',
            "dtype visitor callbacks must have the same return type",
        ),
    ],
    ids=["duplicate-scalar", "duplicate-pair", "empty-visitor", "return-mismatch"],
)
def test_invalid_dtype_lists_fail_compilation(cpp_command, tmp_path, body, diagnostic):
    source = tmp_path / "invalid.cpp"
    source.write_text(
        "#include <api/torch/dispatch.cuh>\nusing namespace astrai;\n"
        "int main() { " + body + " }\n"
    )
    result = subprocess.run(
        cpp_command + ["-fsyntax-only", str(source)], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert diagnostic in result.stderr, result.stderr
