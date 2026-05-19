# ─────────────────────────────────────────────────────────────
# Makefile — CUDA GPU Image Processing Pipeline
# Requires: CUDA Toolkit (nvcc), libpng-dev, CUDA NPP
# ─────────────────────────────────────────────────────────────

NVCC        := nvcc
CXX_STD     := --std=c++17

# Adjust CUDA arch to match your GPU:
#   sm_60  = Pascal  (GTX 10xx, Tesla P100)
#   sm_70  = Volta   (Tesla V100)
#   sm_75  = Turing  (RTX 20xx, Tesla T4)
#   sm_80  = Ampere  (RTX 30xx, A100)
#   sm_86  = Ampere  (RTX 3080/3090)
#   sm_89  = Ada     (RTX 40xx)
ARCH        := -arch=sm_75

INCLUDES    := -Iinclude
LIBS        := -lpng -lnppif -lnppc -lnppig -lnppicc

SRC_DIR     := src
BUILD_DIR   := build
TARGET      := image_processor

SOURCES     := $(SRC_DIR)/main.cu $(SRC_DIR)/utils.cu

.PHONY: all clean run generate help

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/$(TARGET): $(SOURCES) include/image_processor.h include/cuda_utils.h
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(CXX_STD) $(ARCH) $(INCLUDES) $(SOURCES) $(LIBS) -o $@
	@echo "Build successful: $@"

# Generate 20 synthetic PNG test images (requires Python3 + Pillow)
generate:
	@echo "Generating synthetic test images..."
	python3 scripts/generate_test_images.py --count 20 --output data/input
	@echo "Done. Images in data/input/"

# Run the pipeline on data/input → data/output
run: $(BUILD_DIR)/$(TARGET)
	@mkdir -p data/output
	./$(BUILD_DIR)/$(TARGET) \
		--input     data/input  \
		--output    data/output \
		--contrast  1.3         \
		--brightness 15.0

# One-shot: generate data + build + run
demo: generate all run

clean:
	rm -rf $(BUILD_DIR) data/output

help:
	@echo "Targets:"
	@echo "  make           — Build the CUDA executable"
	@echo "  make generate  — Generate 20 synthetic PNG test images"
	@echo "  make run       — Run the image processing pipeline"
	@echo "  make demo      — generate + build + run (full demo)"
	@echo "  make clean     — Remove build artifacts and output images"
