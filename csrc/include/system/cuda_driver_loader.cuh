/*
 * Resolve optional CUDA driver API symbols without making the driver library a
 * hard link dependency. TMA uses this to fall back when the installed driver
 * does not expose cuTensorMapEncodeTiled.
 */
#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace astrai::system {

template <typename Function>
inline Function resolve_cuda_driver_symbol(const char* name) {
#ifdef _WIN32
    HMODULE handle = GetModuleHandleA("nvcuda.dll");
    if (handle == nullptr)
        handle = LoadLibraryA("nvcuda.dll");
    return handle ? reinterpret_cast<Function>(GetProcAddress(handle, name)) : nullptr;
#else
    void* handle = dlopen("libcuda.so.1", RTLD_LAZY);
    if (handle == nullptr)
        handle = dlopen("libcuda.so", RTLD_LAZY);
    return handle ? reinterpret_cast<Function>(dlsym(handle, name)) : nullptr;
#endif
}

}  // namespace astrai::system
