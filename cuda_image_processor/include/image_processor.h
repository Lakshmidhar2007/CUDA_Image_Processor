#pragma once
/**
 * image_processor.h
 * Pipeline configuration and argument parsing declarations.
 */

#include <string>

// Pipeline configuration struct (CLI-driven)
struct PipelineConfig {
    std::string input_dir  = "data/input";
    std::string output_dir = "data/output";
    float       contrast   = 1.2f;   // alpha: contrast multiplier
    float       brightness = 10.0f;  // beta:  brightness offset
};

// Parse command-line arguments into PipelineConfig
PipelineConfig ParseArgs(int argc, char* argv[]);

// Print CUDA device info
void PrintDeviceInfo();
