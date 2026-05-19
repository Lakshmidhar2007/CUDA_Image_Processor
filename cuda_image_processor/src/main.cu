/**
 * CUDA at Scale Independent Project
 * GPU-Accelerated Image Processing Pipeline
 *
 * Pipeline stages:
 *   1. RGB -> Grayscale         (CUDA NPP: nppiRGBToGray)
 *   2. Gaussian Blur            (CUDA NPP: nppiFilterGaussBorder)
 *   3. Sobel Edge Detection     (CUDA NPP: nppiFilterSobelHorizBorder)
 *   4. Brightness/Contrast Adj  (Custom CUDA kernel)
 *   5. Image Inversion          (Custom CUDA kernel)
 *
 * Processes all images in an input directory and writes 5 output
 * variants per image to the output directory.
 *
 * Author: CUDA at Scale Student
 * Style: Google C++ Style Guide
 */

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_filtering_functions.h>
#include <nppi_color_conversion.h>

#include <png.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/image_processor.h"
#include "../include/cuda_utils.h"

namespace fs = std::filesystem;

// ═══════════════════════════════════════════════════════════════
// PNG I/O helpers
// ═══════════════════════════════════════════════════════════════

// Load PNG into a flat RGB buffer (3 bytes/pixel). Returns true on success.
bool LoadPNG(const std::string& path,
             std::vector<unsigned char>& pixels,
             int& width, int& height) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) { std::cerr << "  [ERROR] Cannot open: " << path << "\n"; return false; }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { fclose(fp); return false; }

    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_read_struct(&png, nullptr, nullptr); fclose(fp); return false; }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        return false;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    width  = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth  = png_get_bit_depth(png, info);

    // Normalise to 8-bit RGB
    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE)     png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY ||
        color_type == PNG_COLOR_TYPE_GRAY_ALPHA)  png_set_gray_to_rgb(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS))  png_set_tRNS_to_alpha(png);
    if (color_type & PNG_COLOR_MASK_ALPHA)        png_set_strip_alpha(png);

    png_read_update_info(png, info);

    pixels.resize(width * height * 3);
    std::vector<png_bytep> rows(height);
    for (int y = 0; y < height; ++y)
        rows[y] = pixels.data() + y * width * 3;

    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);
    return true;
}

// Save a single-channel (grayscale) buffer as PNG.
bool SaveGrayPNG(const std::string& path,
                 const std::vector<unsigned char>& pixels,
                 int width, int height) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) { std::cerr << "  [ERROR] Cannot write: " << path << "\n"; return false; }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { fclose(fp); return false; }

    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, nullptr); fclose(fp); return false; }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        fclose(fp);
        return false;
    }

    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8,
                 PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);

    for (int y = 0; y < height; ++y)
        png_write_row(png, const_cast<png_bytep>(pixels.data() + y * width));

    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return true;
}

// ═══════════════════════════════════════════════════════════════
// Custom CUDA Kernel: Brightness + Contrast Adjustment
//   dst[i] = clamp(alpha * src[i] + beta, 0, 255)
// ═══════════════════════════════════════════════════════════════
__global__ void BrightnessContrastKernel(const unsigned char* __restrict__ src,
                                          unsigned char*       __restrict__ dst,
                                          int width, int height,
                                          float alpha, float beta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float v = alpha * static_cast<float>(src[idx]) + beta;
    dst[idx] = static_cast<unsigned char>(fminf(fmaxf(v, 0.f), 255.f));
}

// ═══════════════════════════════════════════════════════════════
// Custom CUDA Kernel: Pixel Inversion
//   dst[i] = 255 - src[i]
// ═══════════════════════════════════════════════════════════════
__global__ void InvertKernel(const unsigned char* __restrict__ src,
                              unsigned char*       __restrict__ dst,
                              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    dst[idx] = 255u - src[idx];
}

// ═══════════════════════════════════════════════════════════════
// Process one image through the full 5-stage pipeline
// ═══════════════════════════════════════════════════════════════
bool ProcessImage(const std::string& input_path,
                  const std::string& output_dir,
                  const PipelineConfig& cfg) {
    // ── Load ──────────────────────────────────────────────────
    int width = 0, height = 0;
    std::vector<unsigned char> host_rgb;

    if (!LoadPNG(input_path, host_rgb, width, height)) return false;

    std::cout << "  " << fs::path(input_path).filename().string()
              << "  [" << width << "x" << height << "]\n";

    const size_t rgb_sz  = static_cast<size_t>(width * height * 3);
    const size_t gray_sz = static_cast<size_t>(width * height);

    // ── Device allocations ────────────────────────────────────
    unsigned char *d_rgb = nullptr, *d_gray = nullptr,
                  *d_blur = nullptr, *d_edge = nullptr,
                  *d_bc   = nullptr, *d_inv  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_rgb,  rgb_sz));
    CUDA_CHECK(cudaMalloc(&d_gray, gray_sz));
    CUDA_CHECK(cudaMalloc(&d_blur, gray_sz));
    CUDA_CHECK(cudaMalloc(&d_edge, gray_sz));
    CUDA_CHECK(cudaMalloc(&d_bc,   gray_sz));
    CUDA_CHECK(cudaMalloc(&d_inv,  gray_sz));

    CUDA_CHECK(cudaMemcpy(d_rgb, host_rgb.data(), rgb_sz, cudaMemcpyHostToDevice));

    NppiSize roi      = {width, height};
    int      rgb_step = width * 3;
    int      g_step   = width;

    // ── Stage 1: RGB → Gray (NPP) ─────────────────────────────
    NPP_CHECK(nppiRGBToGray_8u_C3C1R(d_rgb, rgb_step, d_gray, g_step, roi));

    // ── Stage 2: Gaussian Blur 5×5 (NPP, border replicate) ───
    NPP_CHECK(nppiFilterGaussBorder_8u_C1R(
        d_gray, g_step, roi, {0, 0},
        d_blur, g_step, roi,
        NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE));

    // ── Stage 3: Sobel Edge Detection (NPP, border replicate) ─
    NPP_CHECK(nppiFilterSobelHorizBorder_8u_C1R(
        d_blur, g_step, roi, {0, 0},
        d_edge, g_step, roi,
        NPP_BORDER_REPLICATE));

    // ── Stages 4 & 5: Custom CUDA kernels ────────────────────
    dim3 blk(16, 16);
    dim3 grd((width + 15) / 16, (height + 15) / 16);

    BrightnessContrastKernel<<<grd, blk>>>(
        d_gray, d_bc, width, height, cfg.contrast, cfg.brightness);
    CUDA_CHECK(cudaGetLastError());

    InvertKernel<<<grd, blk>>>(d_gray, d_inv, width, height);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Copy back ─────────────────────────────────────────────
    std::vector<unsigned char> h_gray(gray_sz), h_blur(gray_sz),
                               h_edge(gray_sz), h_bc(gray_sz), h_inv(gray_sz);

    CUDA_CHECK(cudaMemcpy(h_gray.data(), d_gray, gray_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_blur.data(), d_blur, gray_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_edge.data(), d_edge, gray_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bc.data(),   d_bc,   gray_sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_inv.data(),  d_inv,  gray_sz, cudaMemcpyDeviceToHost));

    // ── Save ──────────────────────────────────────────────────
    std::string stem = fs::path(input_path).stem().string();
    auto save = [&](const char* suffix, const std::vector<unsigned char>& buf) {
        SaveGrayPNG(output_dir + "/" + stem + suffix + ".png", buf, width, height);
    };

    save("_1_gray",       h_gray);
    save("_2_blurred",    h_blur);
    save("_3_edges",      h_edge);
    save("_4_brightness", h_bc);
    save("_5_inverted",   h_inv);

    // ── Free ──────────────────────────────────────────────────
    cudaFree(d_rgb); cudaFree(d_gray); cudaFree(d_blur);
    cudaFree(d_edge); cudaFree(d_bc); cudaFree(d_inv);

    return true;
}

// ═══════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════
int main(int argc, char* argv[]) {
    PipelineConfig cfg = ParseArgs(argc, argv);

    std::cout << "=============================================\n"
              << "  CUDA GPU Image Processing Pipeline\n"
              << "=============================================\n"
              << "  Input dir  : " << cfg.input_dir  << "\n"
              << "  Output dir : " << cfg.output_dir << "\n"
              << "  Contrast   : " << cfg.contrast   << "\n"
              << "  Brightness : " << cfg.brightness << "\n"
              << "---------------------------------------------\n";

    PrintDeviceInfo();

    // Collect PNG images
    std::vector<std::string> paths;
    for (const auto& e : fs::directory_iterator(cfg.input_dir)) {
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png") paths.push_back(e.path().string());
    }

    if (paths.empty()) {
        std::cerr << "[ERROR] No PNG images found in: " << cfg.input_dir << "\n";
        return 1;
    }

    std::sort(paths.begin(), paths.end());
    fs::create_directories(cfg.output_dir);

    std::cout << "Found " << paths.size() << " image(s). Processing...\n\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    int  ok  = 0;
    for (const auto& p : paths)
        if (ProcessImage(p, cfg.output_dir, cfg)) ++ok;
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\n=============================================\n"
              << "  Summary\n"
              << "=============================================\n"
              << "  Processed : " << ok << " / " << paths.size() << " images\n"
              << "  Total time: " << std::fixed << std::setprecision(3) << elapsed << " s\n"
              << "  Per image : " << std::fixed << std::setprecision(3)
                                  << elapsed / paths.size() << " s\n"
              << "  Outputs in: " << cfg.output_dir << "\n"
              << "=============================================\n";

    return (ok == static_cast<int>(paths.size())) ? 0 : 1;
}
