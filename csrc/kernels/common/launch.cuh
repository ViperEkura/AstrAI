// Launch-and-check macros — pure C, no torch/C++ deps, so out-of-tree
// harnesses (tile sweeps, csrc/tests) share the exact production launch
// discipline. A failed launch prints one line to stderr and exits the
// process — a rejected configuration must fail loudly instead of silently
// measuring as a constant ~3us no-op (the tile-sweep lesson).
//
// Define ASTRAI_LAUNCH_FAIL before including this header (directly or via
// another kernel header) to override the failure path.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef ASTRAI_LAUNCH_FAIL
#define ASTRAI_LAUNCH_FAIL(err, what)                                        \
    do {                                                                     \
        std::fprintf(                                                        \
            stderr, "ASTRAI: %s failed: %s (%s:%d)\n", what,                 \
            cudaGetErrorString(err), __FILE__, __LINE__                      \
        );                                                                   \
        std::exit(EXIT_FAILURE);                                             \
    } while (0)
#endif

#define ASTRAI_CUDA_CHECK(expr)                                              \
    do {                                                                     \
        cudaError_t astrai_err_ = (expr);                                    \
        if (astrai_err_ != cudaSuccess) {                                    \
            ASTRAI_LAUNCH_FAIL(astrai_err_, #expr);                          \
        }                                                                    \
    } while (0)

// Check a kernel launch. Wrap the raw <<<>>> with this on the next line;
// a rejected configuration must fail loudly instead of silently measuring
// as a constant ~3us no-op (the tile-sweep lesson).
#define ASTRAI_LAUNCH_CHECK() ASTRAI_CUDA_CHECK(cudaGetLastError())
