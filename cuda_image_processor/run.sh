#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh — Full demo: generate test data, build, and run the pipeline
# ─────────────────────────────────────────────────────────────────────────────
set -e  # exit on first error

echo "============================================="
echo " CUDA GPU Image Processing Pipeline — run.sh"
echo "============================================="

# ── 0. Check prerequisites ───────────────────────────────────────────────────
command -v nvcc  >/dev/null 2>&1 || { echo "[ERROR] nvcc not found. Install CUDA Toolkit."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "[ERROR] python3 not found."; exit 1; }

echo "[OK] nvcc found:   $(nvcc --version | head -1)"
echo "[OK] python3 found: $(python3 --version)"

# ── 1. Parse arguments ───────────────────────────────────────────────────────
INPUT_DIR="data/input"
OUTPUT_DIR="data/output"
IMAGE_COUNT=20
CONTRAST=1.3
BRIGHTNESS=15.0
ARCH="sm_75"   # ← Change this to match your GPU (sm_60/70/75/80/86/89)

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)       INPUT_DIR="$2";    shift 2 ;;
        --output)      OUTPUT_DIR="$2";   shift 2 ;;
        --count)       IMAGE_COUNT="$2";  shift 2 ;;
        --contrast)    CONTRAST="$2";     shift 2 ;;
        --brightness)  BRIGHTNESS="$2";   shift 2 ;;
        --arch)        ARCH="$2";         shift 2 ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

echo ""
echo "Settings:"
echo "  Input dir  : $INPUT_DIR"
echo "  Output dir : $OUTPUT_DIR"
echo "  Image count: $IMAGE_COUNT"
echo "  CUDA arch  : $ARCH"
echo ""

# ── 2. Generate test images ──────────────────────────────────────────────────
echo "[Step 1/3] Generating $IMAGE_COUNT synthetic PNG test images..."
python3 scripts/generate_test_images.py \
    --count  "$IMAGE_COUNT"  \
    --output "$INPUT_DIR"    \
    --size   512x512

echo ""

# ── 3. Build ─────────────────────────────────────────────────────────────────
echo "[Step 2/3] Building CUDA project (arch=$ARCH)..."
mkdir -p build
nvcc --std=c++17 \
     -arch="$ARCH" \
     -Iinclude \
     src/main.cu src/utils.cu \
     -lpng -lnppif -lnppc -lnppig -lnppicc \
     -o build/image_processor

echo "[OK] Build complete: build/image_processor"
echo ""

# ── 4. Run ───────────────────────────────────────────────────────────────────
echo "[Step 3/3] Running pipeline..."
mkdir -p "$OUTPUT_DIR"

./build/image_processor \
    --input      "$INPUT_DIR"  \
    --output     "$OUTPUT_DIR" \
    --contrast   "$CONTRAST"   \
    --brightness "$BRIGHTNESS"

echo ""
echo "============================================="
echo " Pipeline complete!"
echo " Input images  : $INPUT_DIR"
echo " Output images : $OUTPUT_DIR"
echo "============================================="

# ── 5. Summary ───────────────────────────────────────────────────────────────
IN_COUNT=$(ls "$INPUT_DIR"/*.png 2>/dev/null | wc -l)
OUT_COUNT=$(ls "$OUTPUT_DIR"/*.png 2>/dev/null | wc -l)
echo " Input PNGs  : $IN_COUNT"
echo " Output PNGs : $OUT_COUNT  (5 variants × $IN_COUNT images)"
echo "============================================="
