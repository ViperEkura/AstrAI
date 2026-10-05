set(ASTRAI_EXTENSION_SUFFIX ".so")
set(TORCH_LIBS
    "${TORCH_LIB_DIR}/libtorch_python.so"
    "${TORCH_LIB_DIR}/libtorch_cuda.so"
    "${TORCH_LIB_DIR}/libc10_cuda.so"
    "${TORCH_LIB_DIR}/libtorch_cpu.so"
    "${TORCH_LIB_DIR}/libtorch.so"
    "${TORCH_LIB_DIR}/libc10.so"
    CUDA::cudart)

function(astrai_platform_compile_options target)
    target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-O3;-funroll-loops>")
endfunction()

function(astrai_platform_link_options target)
    target_link_options(${target} PRIVATE "-Wl,-rpath,${TORCH_LIB_DIR}")
endfunction()
