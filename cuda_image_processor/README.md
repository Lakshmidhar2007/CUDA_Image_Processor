# CUDA GPU Image Processing Pipeline

> **CUDA at Scale for the Enterprise** — Independent Project  
> GPU-accelerated batch image processing using CUDA NPP and custom CUDA kernels.

---

## Project Overview

This project implements a **5-stage GPU image processing pipeline** that processes batches of PNG images entirely on the GPU. Each input image produces 5 output variants demonstrating different processing stages.

### Pipeline Stages

| Stage | Operation | Implementation |
|-------|-----------|---------------|
| 1 | RGB → Grayscale | CUDA NPP `nppiRGBToGray_8u_C3C1R` |
| 2 | Gaussian Blur (5×5) | CUDA NPP `nppiFilterGaussBorder_8u_C1R` |
| 3 | Sobel Edge Detection | CUDA NPP `nppiFilterSobelHorizBorder_8u_C1R` |
| 4 | Brightness/Contrast Adjust | Custom CUDA kernel `BrightnessContrastKernel` |
| 5 | Image Inversion | Custom CUDA kernel `InvertKernel` |

### Project Structure

```
cuda_image_processor/
├── src/
│   ├── main.cu          # Main pipeline + PNG I/O + CUDA kernels
│   └── utils.cu         # Argument parsing + device info
├── include/
│   ├── image_processor.h  # PipelineConfig struct + function declarations
│   └── cuda_utils.h       # CUDA_CHECK / NPP_CHECK macros
├── scripts/
│   └── generate_test_images.py  # Synthetic PNG generator (pure Python, no deps)
├── data/
│   ├── input/           # Place input PNG images here
│   └── output/          # Generated output images appear here
├── docs/
│   └── description.md   # Project description for peer review
├── Makefile
├── run.sh               # One-shot build + run script
└── README.md
```

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| NVIDIA GPU | Compute Capability 6.0+ (Pascal or newer) |
| CUDA Toolkit | 11.0 or newer (`nvcc` in PATH) |
| libpng | `sudo apt install libpng-dev` (Ubuntu/Debian) |
| CUDA NPP | Included with CUDA Toolkit |
| Python 3 | For test image generation (stdlib only, no Pillow needed) |
| C++17 | Required by `std::filesystem` |

### Install libpng (Ubuntu/Debian)
```bash
sudo apt update && sudo apt install -y libpng-dev
```

### Install libpng (CentOS/RHEL/Rocky)
```bash
sudo yum install -y libpng-devel
```

---

## Quick Start

### Option A — One-shot script (recommended)
```bash
# Clone / enter the project
cd cuda_image_processor

# Edit ARCH in run.sh to match your GPU (see table below), then:
bash run.sh
```

### Option B — Step by step
```bash
# Step 1: Generate 20 synthetic test images
python3 scripts/generate_test_images.py --count 20 --output data/input

# Step 2: Build (edit ARCH= to match your GPU)
make ARCH=-arch=sm_75

# Step 3: Run
make run

# Or run manually with custom args:
./build/image_processor \
    --input     data/input  \
    --output    data/output \
    --contrast  1.3         \
    --brightness 15.0
```

### Option C — Full demo via Make
```bash
make demo
```

---

## GPU Architecture Flag

Edit the `ARCH` variable in `Makefile` or `run.sh` to match your GPU:

| GPU Generation | Examples | Flag |
|---------------|----------|------|
| Pascal | GTX 10xx, Tesla P100 | `sm_60` |
| Volta | Tesla V100 | `sm_70` |
| Turing | RTX 20xx, Tesla T4 | `sm_75` |
| Ampere | RTX 30xx, A100 | `sm_80` / `sm_86` |
| Ada Lovelace | RTX 40xx | `sm_89` |

Check your GPU: `nvidia-smi --query-gpu=name,compute_cap --format=csv`

---

## CLI Arguments

```
./build/image_processor [OPTIONS]

  --input      <dir>    Input directory containing PNG images  (default: data/input)
  --output     <dir>    Output directory for result images     (default: data/output)
  --contrast   <float>  Contrast multiplier (alpha, 0.5–2.0)  (default: 1.3)
  --brightness <float>  Brightness offset (beta, -127–127)     (default: 15.0)
  --help                Show this help message
```

---

## Output

For each input image `img_000_gradient_45.png`, the pipeline produces:

```
data/output/
├── img_000_gradient_45_1_gray.png        # Grayscale
├── img_000_gradient_45_2_blurred.png     # Gaussian blur
├── img_000_gradient_45_3_edges.png       # Sobel edge map
├── img_000_gradient_45_4_brightness.png  # Brightness/contrast adjusted
└── img_000_gradient_45_5_inverted.png    # Inverted
```

Processing 20 images (512×512) generates **100 output PNG files**.

---

## Example Output (console)

```
=============================================
  CUDA GPU Image Processing Pipeline
=============================================
  Input dir  : data/input
  Output dir : data/output
  Contrast   : 1.3
  Brightness : 15

CUDA Devices found: 1
  Device 0: Tesla T4
    Compute capability : 7.5
    Global memory      : 14910 MB
    SM count           : 40
    Max threads/block  : 1024

Found 20 image(s). Processing...

  img_000_gradient_45.png     [512x512]
  img_001_gradient_135.png    [512x512]
  img_002_circles.png         [512x512]
  ...
  img_019_diagonal.png        [512x512]

=============================================
  Summary
=============================================
  Processed : 20 / 20 images
  Total time: 0.847 s
  Per image : 0.042 s
  Outputs in: data/output
=============================================
```

## Actual Output

```
Generating 20 images (512x512) in 'data/input'...
  [  1/20] img_000_gradient_45.png
  [  2/20] img_001_gradient_135.png
  [  3/20] img_002_circles.png
  [  4/20] img_003_checkerboard.png
  [  5/20] img_004_noise.png
  [  6/20] img_005_stripes_h.png
  [  7/20] img_006_stripes_v.png
  [  8/20] img_007_sine_wave.png
  [  9/20] img_008_radial.png
  [ 10/20] img_009_diagonal.png
  [ 11/20] img_010_gradient_45.png
  [ 12/20] img_011_gradient_135.png
  [ 13/20] img_012_circles.png
  [ 14/20] img_013_checkerboard.png
  [ 15/20] img_014_noise.png
  [ 16/20] img_015_stripes_h.png
  [ 17/20] img_016_stripes_v.png
  [ 18/20] img_017_sine_wave.png
  [ 19/20] img_018_radial.png
  [ 20/20] img_019_diagonal.png

Done. 20 images written to 'data/input'.
=============================================
  CUDA GPU Image Processing Pipeline
=============================================
  Input dir  : data/input
  Output dir : data/output
  Contrast   : 1.3
  Brightness : 15
---------------------------------------------

CUDA Devices found: 1
  Device 0: Tesla T4
    Compute capability : 7.5
    Global memory      : 14912 MB
    SM count           : 40
    Max threads/block  : 1024

Found 20 image(s). Processing...

  img_000_gradient_45.png  [512x512]
  img_001_gradient_135.png  [512x512]
  img_002_circles.png  [512x512]
  img_003_checkerboard.png  [512x512]
  img_004_noise.png  [512x512]
  img_005_stripes_h.png  [512x512]
  img_006_stripes_v.png  [512x512]
  img_007_sine_wave.png  [512x512]
  img_008_radial.png  [512x512]
  img_009_diagonal.png  [512x512]
  img_010_gradient_45.png  [512x512]
  img_011_gradient_135.png  [512x512]
  img_012_circles.png  [512x512]
  img_013_checkerboard.png  [512x512]
  img_014_noise.png  [512x512]
  img_015_stripes_h.png  [512x512]
  img_016_stripes_v.png  [512x512]
  img_017_sine_wave.png  [512x512]
  img_018_radial.png  [512x512]
  img_019_diagonal.png  [512x512]

=============================================
  Summary
=============================================
  Processed : 20 / 20 images
  Total time: 1.494 s
  Per image : 0.075 s
  Outputs in: data/output
=============================================
```

---

## Make Targets

```bash
make           # Build only
make generate  # Generate 20 synthetic test images
make run       # Build + run pipeline
make demo      # generate + build + run (full demo)
make clean     # Remove build/ and data/output/
make help      # Show available targets
```

---

## Design Notes

- **No external image library dependency** — PNG I/O is handled directly via `libpng` (standard on any Linux CUDA system), avoiding third-party header downloads.
- **Pure Python test data** — `generate_test_images.py` uses only Python stdlib (no Pillow/OpenCV) to write valid PNG files, so test data generation works anywhere Python 3 is installed.
- **Modular pipeline** — Each stage is independent; stages 1–3 use NPP for performance, stages 4–5 use hand-written CUDA kernels to demonstrate direct GPU programming.
- **Google C++ Style Guide** — naming, formatting, and documentation follow the guide throughout.
