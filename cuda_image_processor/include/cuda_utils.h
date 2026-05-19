#pragma once
/**
 * cuda_utils.h
 * CUDA and NPP error-checking macros.
 */

#include <cuda_runtime.h>
#include <npp.h>
#include <iostream>
#include <cstdlib>

// Check CUDA runtime API calls
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Check NPP API calls
#define NPP_CHECK(call)                                                         \
    do {                                                                        \
        NppStatus status = (call);                                              \
        if (status != NPP_SUCCESS) {                                            \
            std::cerr << "[NPP ERROR] status=" << status                        \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)
