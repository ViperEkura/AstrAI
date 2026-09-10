cmake_minimum_required(VERSION 3.18)
project(astrai_kernels LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED TORCH_HOME)
    set(TORCH_HOME "$ENV{TORCH_HOME}")
endif()
if(NOT TORCH_HOME)
    message(FATAL_ERROR "TORCH_HOME must point at the torch install dir (site-packages/torch)")
endif()

if(NOT DEFINED PYTHON_INCLUDE_DIR)
    set(PYTHON_INCLUDE_DIR "/usr/include/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
endif()

if(NOT DEFINED ASTRAI_CUDA_ARCH)
    if(DEFINED ENV{ASTRAI_CUDA_ARCH})
        set(ASTRAI_CUDA_ARCH "$ENV{ASTRAI_CUDA_ARCH}")
    else()
        # Mixed-fleet fallback for GPU-less builds (setup.py passes the
        # local GPU's arch when one is present): native SASS for every
        # supported generation; interim GPUs JIT the nearest compute_* PTX.
        set(ASTRAI_CUDA_ARCH "80;89;120")
    endif()
endif()

# ASTRAI_CUDA_ARCH accepts a single arch or a semicolon list ("80;89;120"):
# every listed arch lands in the fatbin, so one .so serves a mixed fleet —
# the driver picks the SASS slice per device. Validation and the FP8 gate
# below run on the list's maximum.
#
# Arch-level conventions, one per layer (keep them in lockstep):
#   build (this file / setup.py env): CC tens        "120"  = CC 12.0
#   device compile (mma.cuh): __CUDA_ARCH__ hundreds  1200  (kMinArch compares
#     against ASTRAI_DEVICE_ARCH, the raw __CUDA_ARCH__)
#   runtime (device_facts().cc): CC tens again        120   (gemm.cuh's
#     route gates — cc >= 90 TMA, cc == 120 the sm_120a mx cell)
# The sm_120a image below is emitted exactly when "120" is listed, and the
# runtime mx route keys on cc == 120: the two conditions are one contract.
separate_arguments(ASTRAI_CUDA_ARCH_LIST NATIVE_COMMAND "${ASTRAI_CUDA_ARCH}")
list(TRANSFORM ASTRAI_CUDA_ARCH_LIST STRIP)
list(REMOVE_DUPLICATES ASTRAI_CUDA_ARCH_LIST)
if(NOT ASTRAI_CUDA_ARCH_LIST)
    message(FATAL_ERROR "ASTRAI_CUDA_ARCH resolved to an empty arch list")
endif()
list(SORT ASTRAI_CUDA_ARCH_LIST COMPARE NATURAL)
list(GET ASTRAI_CUDA_ARCH_LIST -1 ASTRAI_CUDA_ARCH_MAX)

set(TORCH_LIB_DIR "${TORCH_HOME}/lib")
set(CUDA_LIB_DIR "/usr/local/cuda/lib64")

set(CXX_FLAGS -O3 -funroll-loops)
set(NVCC_FLAGS -O3
    --expt-relaxed-constexpr
    --use_fast_math
    "--ptxas-options=-O3,-v"
    --extra-device-vectorization
    --threads=16)

set(TORCH_LIBS
    "${TORCH_LIB_DIR}/libtorch_python.so"
    "${TORCH_LIB_DIR}/libtorch_cuda.so"
    "${TORCH_LIB_DIR}/libc10_cuda.so"
    "${TORCH_LIB_DIR}/libtorch_cpu.so"
    "${TORCH_LIB_DIR}/libtorch.so"
    "${TORCH_LIB_DIR}/libc10.so"
    CUDA::cudart)

# One authoritative arch knob: force the cache entry too, so a stale
# toolkit-default value (e.g. 75, seeded by project() before the env is
# parsed) never lingers in CMakeCache.txt next to the effective list.
set(CMAKE_CUDA_ARCHITECTURES "${ASTRAI_CUDA_ARCH_LIST}"
    CACHE STRING "CUDA architectures (derived from ASTRAI_CUDA_ARCH)" FORCE)

# Kernel registry — one module per entry: name (.so / pybind name, globally
# unique across families), then its source paths under kernels/ (space-
# separated: a module may span several TUs; gemm is split into per-dtype-pair
# instantiation units (one explicit gemm_dispatch instantiation each), so
# the heavy template work parallelizes). The '|' separator keeps the entry
# a single list element — CMake lists cannot nest semicolons. `loader.py`
# auto-discovers the .so files in astrai/extension/lib/, so this CMake
# registry is the single place to register a new kernel.
#
# FP8 MMA instructions require sm_89+. Keep the target out of the build
# unless at least one listed arch is 89+ instead of instantiating templates
# that cannot compile. The remaining kernels are still useful on sm_80+
# (including sm_86).
set(KERNEL_MODULES
    "attn_decode|attention/decode.cu"
    "attn_prefill|attention/prefill.cu"
    "attn_paged_decode|attention/paged_decode.cu"
    "attn_paged_prefill|attention/paged_prefill.cu"
    "rotary_emb|rotary_emb.cu"
)

if(ASTRAI_CUDA_ARCH_MAX GREATER_EQUAL 89)
    list(APPEND KERNEL_MODULES
        "quantize|quantize/quantize.cu"
        "gemm|gemm/gemm.cu gemm/gemm_bf16_bf16.cu gemm/gemm_bf16_int8.cu gemm/gemm_bf16_fp8_e4m3.cu gemm/gemm_bf16_fp8_e5m2.cu gemm/gemm_int8_int8.cu gemm/gemm_fp8_e4m3_fp8_e4m3.cu gemm/gemm_fp8_e5m2_fp8_e5m2.cu")
else()
    message(WARNING
        "FP8 operator disabled: max ASTRAI_CUDA_ARCH=${ASTRAI_CUDA_ARCH} "
        "requires compute capability 89 or newer")
endif()

foreach(entry IN LISTS KERNEL_MODULES)
    string(REPLACE "|" ";" _parts "${entry}")
    list(GET _parts 0 name)
    list(REMOVE_AT _parts 0)
    string(REPLACE " " ";" _srcs "${_parts}")
    list(TRANSFORM _srcs PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/kernels/")
    add_library(${name} MODULE ${_srcs})

    target_compile_definitions(${name} PRIVATE TORCH_EXTENSION_NAME=${name})

    # Per-family arch lists: the fp8 mma cells static_assert below sm_89
    # (MmaOp's kMinArch = 890), so the fp8 modules compile only the 89+
    # subset of the requested arches — a compute_80 device pass would trip
    # the assert at template instantiation. Every other family carries the
    # full list (its oldest cell is sm_80). The 120a image below keys on
    # the shared list, so it rides along whenever 120 is requested.
    set(_target_arches "${ASTRAI_CUDA_ARCH_LIST}")
    if(name STREQUAL "gemm" OR name STREQUAL "quantize")
        set(_target_arches "")
        foreach(a IN LISTS ASTRAI_CUDA_ARCH_LIST)
            if(a GREATER_EQUAL 89)
                list(APPEND _target_arches "${a}")
            endif()
        endforeach()
    endif()
    set_target_properties(${name} PROPERTIES
        CUDA_ARCHITECTURES "${_target_arches}")

    target_include_directories(${name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
        "${TORCH_HOME}/include"
        "${TORCH_HOME}/include/torch/csrc/api/include"
        "${PYTHON_INCLUDE_DIR}")

    target_link_libraries(${name} PRIVATE ${TORCH_LIBS})
    target_link_options(${name} PRIVATE "-Wl,-rpath,${TORCH_LIB_DIR}")

    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>)

    set_target_properties(${name} PROPERTIES
        PREFIX ""
        SUFFIX ".${PY_SOABI}.so"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../astrai/extension/lib")

    # Extra sm_120a image for gemm when 120 is built: the arch-specific pass
    # that activates the block_scale cell (MxMmaOp; CMake <3.24 rejects the
    # 'a' suffix in CUDA_ARCHITECTURES, hence the raw -gencode).
    if(name STREQUAL "gemm" AND "120" IN_LIST ASTRAI_CUDA_ARCH_LIST)
        target_compile_options(${name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_120a,code=sm_120a>)
    endif()
endforeach()
