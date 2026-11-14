cmake_minimum_required(VERSION 3.18)
project(astrai_kernels LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
find_package(CUDAToolkit REQUIRED)
include(cmake/arch.cmake)
include(cmake/parallel.cmake)

if(NOT DEFINED TORCH_HOME)
    set(TORCH_HOME "$ENV{TORCH_HOME}")
endif()
if(NOT TORCH_HOME)
    message(FATAL_ERROR "TORCH_HOME must point at site-packages/torch")
endif()
if(NOT DEFINED PYTHON_INCLUDE_DIR)
    set(PYTHON_INCLUDE_DIR "/usr/include/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
endif()
set(ASTRAI_KERNEL_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../astrai/extension"
    CACHE PATH "Directory for compiled AstrAI extension modules")
set(TORCH_LIB_DIR "${TORCH_HOME}/lib")
if(WIN32)
    include(cmake/windows.cmake)
else()
    include(cmake/linux.cmake)
endif()

function(astrai_compile_target target)
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    # The pool caps concurrent TUs; --threads caps the threads inside one
    # nvcc. Both come from cmake/parallel.cmake and multiply into the load.
    if(ASTRAI_JOB_POOL)
        set_target_properties(${target} PROPERTIES JOB_POOL_COMPILE ${ASTRAI_JOB_POOL})
    endif()
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
        "${TORCH_HOME}/include"
        "${TORCH_HOME}/include/torch/csrc/api/include"
        "${PYTHON_INCLUDE_DIR}")
    target_link_libraries(${target} PRIVATE ${TORCH_LIBS})
    astrai_platform_compile_options(${target})
    target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-O3,-v>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--extra-device-vectorization>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--compress-all>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--threads=${ASTRAI_NVCC_THREADS}>")
endfunction()

if(NOT ASTRAI_CUDA_ARCH_LIST)
    message(STATUS "ASTRAI_CUDA_ARCH is unset: no CUDA kernel targets are built")
    return()
endif()

# Target name, library kind, and sources; modules declare shared object dependencies here.
set(KERNEL_TARGETS
    "quantize_entry|OBJECT|quantize/entry.cu"
    "attention|MODULE|attention/decode.cu attention/prefill.cu attention/paged_decode.cu
        attention/paged_prefill.cu attention/bindings.cu"
    "rotary_emb|MODULE|rotary_emb.cu"
    "swiglu|MODULE|swiglu.cu"
    "cross_entropy|MODULE|cross_entropy.cu"
    "newton_schulz|MODULE|newton_schulz/entry.cu newton_schulz/kernels.cu
        newton_schulz/bindings.cu"
    "gated_deltanet|MODULE|gated_deltanet/gated_deltanet_fwd.cu
        gated_deltanet/gated_deltanet_bwd.cu gated_deltanet/bindings.cu"
    "quantize|MODULE|quantize/bindings.cu $<TARGET_OBJECTS:quantize_entry>"
    "gemm|MODULE|gemm/entry.cu gemm/bindings.cu gemm/plan/planner.cpp
        gemm/plan/table.cpp gemm/plan/builtin.cpp
        gemm/fp8/linear.cu gemm/fp8/runtime.cu $<TARGET_OBJECTS:quantize_entry>")

foreach(entry IN LISTS KERNEL_TARGETS)
    string(REPLACE "|" ";" parts "${entry}")
    list(GET parts 0 name)
    list(GET parts 1 kind)
    list(GET parts 2 sources)
    separate_arguments(sources NATIVE_COMMAND "${sources}")
    add_library(${name} ${kind} ${sources})
    astrai_compile_target(${name})
    astrai_target_arches(${name} ${ASTRAI_CUDA_ARCH_LIST})
    if(kind STREQUAL "MODULE")
        target_compile_definitions(${name} PRIVATE TORCH_EXTENSION_NAME=_C_${name})
        astrai_platform_link_options(${name})
        if(name STREQUAL "cross_entropy")
            target_link_libraries(${name} PRIVATE CUDA::cublas)
        endif()
        set_target_properties(${name} PROPERTIES PREFIX "" OUTPUT_NAME "_C_${name}"
            SUFFIX ".${PY_SOABI}${ASTRAI_EXTENSION_SUFFIX}"
            LIBRARY_OUTPUT_DIRECTORY "${ASTRAI_KERNEL_OUTPUT_DIRECTORY}"
            RUNTIME_OUTPUT_DIRECTORY "${ASTRAI_KERNEL_OUTPUT_DIRECTORY}")
    endif()
endforeach()

set(GEMM_BASE_SOURCES
    gemm/instantiations/bf16_bf16.cu
    gemm/instantiations/bf16_int8.cu
    gemm/instantiations/int8_int8.cu)
set(GEMM_FP8_SOURCES
    gemm/instantiations/fp8_e4m3_fp8_e4m3.cu
    gemm/instantiations/fp8_e5m2_fp8_e5m2.cu)

function(astrai_gemm_objects name schedule arches)
    add_library(${name} OBJECT ${ARGN})
    astrai_compile_target(${name})
    astrai_target_arches(${name} ${arches})
    target_compile_definitions(${name} PRIVATE ASTRAI_GEMM_SCHEDULE=${schedule})
    target_sources(gemm PRIVATE $<TARGET_OBJECTS:${name}>)
endfunction()

astrai_gemm_objects(gemm_base MmaSync "${ASTRAI_CUDA_ARCH_LIST}" ${GEMM_BASE_SOURCES})
if(ASTRAI_FP8_ARCHES)
    astrai_gemm_objects(gemm_fp8 MmaSync "${ASTRAI_FP8_ARCHES}" ${GEMM_FP8_SOURCES})
endif()
if(ASTRAI_TMA_ARCHES)
    astrai_gemm_objects(gemm_tma TmaMma "${ASTRAI_TMA_ARCHES}"
        ${GEMM_BASE_SOURCES} ${GEMM_FP8_SOURCES})
endif()
if(ASTRAI_MX_ARCHES)
    astrai_gemm_objects(gemm_mx Sm120Mma "${ASTRAI_MX_ARCHES}" ${GEMM_FP8_SOURCES})
endif()
