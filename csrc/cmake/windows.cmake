set(ASTRAI_EXTENSION_SUFFIX ".pyd")
set(TORCH_LIBS
    "${TORCH_LIB_DIR}/torch_python.lib"
    "${TORCH_LIB_DIR}/torch_cuda.lib"
    "${TORCH_LIB_DIR}/c10_cuda.lib"
    "${TORCH_LIB_DIR}/torch_cpu.lib"
    "${TORCH_LIB_DIR}/torch.lib"
    "${TORCH_LIB_DIR}/c10.lib"
    CUDA::cudart)

function(astrai_platform_compile_options target)
    target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:/O2>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/permissive->")
endfunction()

function(astrai_platform_link_options target)
    # PyTorch's Windows import libraries reference pythonXY.lib via /DEFAULTLIB.
    # setup.py passes Python's Include directory; the matching import library
    # lives in its sibling libs directory.
    get_filename_component(PYTHON_ROOT_DIR "${PYTHON_INCLUDE_DIR}" DIRECTORY)
    target_link_directories(${target} PRIVATE "${PYTHON_ROOT_DIR}/libs")
endfunction()

# Planner translation units include shared tile metadata that exposes inline PTX.
# Compile them with nvcc so MSVC never parses device instructions.
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/gemm/planning.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gemm/plan_table.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gemm/plan_table_builtin.cpp"
    PROPERTIES LANGUAGE CUDA)
