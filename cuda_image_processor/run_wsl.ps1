# run_wsl.ps1
# ============================================================
# Runs the CUDA image processing pipeline inside WSL2.
# Just double-click or right-click -> "Run with PowerShell"
#
# Prerequisites:
#   - WSL2 installed with Ubuntu
#   - CUDA Toolkit installed inside WSL2
#   - libpng installed inside WSL2: sudo apt install libpng-dev
#   - NVIDIA drivers for WSL2 installed (Windows-side driver)
# ============================================================

param(
    [string]$Arch       = "sm_75",   # Change to match your GPU
    [int]   $Count      = 20,
    [float] $Contrast   = 1.3,
    [float] $Brightness = 15.0
)

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  CUDA GPU Image Processing Pipeline (WSL2)" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# ── Find WSL ─────────────────────────────────────────────────
$wsl = Get-Command wsl -ErrorAction SilentlyContinue
if (-not $wsl) {
    Write-Host "[ERROR] WSL not found. Install it with: wsl --install" -ForegroundColor Red
    Write-Host "        See WINDOWS_SETUP.md for details." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[OK] WSL found" -ForegroundColor Green

# ── Convert Windows path to WSL path ─────────────────────────
$winPath  = (Get-Location).Path
$wslPath  = wsl wslpath -u "$winPath"
$wslPath  = $wslPath.Trim()

Write-Host "[OK] Project path in WSL: $wslPath" -ForegroundColor Green
Write-Host ""

# ── Check GPU in WSL ─────────────────────────────────────────
Write-Host "Checking GPU access in WSL..." -ForegroundColor Yellow
wsl -e bash -c "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo '[WARN] nvidia-smi not available in WSL yet'"
Write-Host ""

# ── Build the run command ─────────────────────────────────────
$cmd = @"
set -e
cd '$wslPath'

echo '--- Checking prerequisites ---'
which nvcc || { echo '[ERROR] nvcc not found in WSL. See WINDOWS_SETUP.md'; exit 1; }
nvcc --version | head -1

echo ''
echo '[Step 1/3] Generating $Count synthetic PNG test images...'
python3 scripts/generate_test_images.py --count $Count --output data/input --size 512x512

echo ''
echo '[Step 2/3] Building (arch=$Arch)...'
mkdir -p build
nvcc --std=c++17 \
     -arch=$Arch \
     -Iinclude \
     src/main.cu src/utils.cu \
     -lpng -lnppif -lnppc -lnppig -lnppicc \
     -o build/image_processor
echo '[OK] Build complete'

echo ''
echo '[Step 3/3] Running pipeline...'
mkdir -p data/output
./build/image_processor \
    --input     data/input \
    --output    data/output \
    --contrast  $Contrast \
    --brightness $Brightness

echo ''
echo '============================================='
echo ' Pipeline complete!'
echo " Output files: \$(ls data/output/*.png 2>/dev/null | wc -l) PNGs in data/output/"
echo '============================================='
"@

Write-Host "Running pipeline inside WSL2..." -ForegroundColor Yellow
wsl -e bash -c $cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "SUCCESS! Output images are in: $winPath\data\output\" -ForegroundColor Green
    Write-Host "Open File Explorer and navigate there to view results." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[ERROR] Pipeline failed. See output above for details." -ForegroundColor Red
    Write-Host "Consult WINDOWS_SETUP.md for troubleshooting." -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
