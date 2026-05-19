/**
 * utils.cu
 * Utility function implementations: argument parsing, device info.
 */

#include "../include/image_processor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <stdexcept>

// ─────────────────────────────────────────────
// Argument parsing
// Usage:
//   ./image_processor [--input <dir>] [--output <dir>]
//                     [--contrast <float>] [--brightness <float>]
// ─────────────────────────────────────────────
PipelineConfig ParseArgs(int argc, char* argv[]) {
    PipelineConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--input" && i + 1 < argc) {
            config.input_dir = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_dir = argv[++i];
        } else if (arg == "--contrast" && i + 1 < argc) {
            config.contrast = std::stof(argv[++i]);
        } else if (arg == "--brightness" && i + 1 < argc) {
            config.brightness = std::stof(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: image_processor [OPTIONS]\n"
                      << "  --input      <dir>    Input directory with images  (default: data/input)\n"
                      << "  --output     <dir>    Output directory for results (default: data/output)\n"
                      << "  --contrast   <float>  Contrast multiplier alpha    (default: 1.2)\n"
                      << "  --brightness <float>  Brightness offset beta       (default: 10.0)\n"
                      << "  --help                Show this help message\n";
            std::exit(0);
        } else {
            std::cerr << "[WARN] Unknown argument: " << arg << " (ignored)\n";
        }
    }

    return config;
}

// ─────────────────────────────────────────────
// Print CUDA device information
// ─────────────────────────────────────────────
void PrintDeviceInfo() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        std::cerr << "[ERROR] No CUDA-capable devices found!\n";
        std::exit(EXIT_FAILURE);
    }

    std::cout << "\nCUDA Devices found: " << device_count << "\n";

    for (int d = 0; d < device_count; ++d) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);
        std::cout << "  Device " << d << ": " << prop.name << "\n"
                  << "    Compute capability : " << prop.major << "." << prop.minor << "\n"
                  << "    Global memory      : "
                  << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n"
                  << "    SM count           : " << prop.multiProcessorCount << "\n"
                  << "    Max threads/block  : " << prop.maxThreadsPerBlock << "\n";
    }
    std::cout << "\n";
}
