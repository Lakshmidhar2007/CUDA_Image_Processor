@echo off
REM ============================================================
REM  run.bat — Build and run on Windows (native CUDA + vcpkg)
REM  Run from: x64 Native Tools Command Prompt for VS 2022
REM ============================================================

setlocal

REM ── Configuration ────────────────────────────────────────────
REM Edit ARCH to match your GPU (see WINDOWS_SETUP.md)
set ARCH=sm_75

REM vcpkg root — change if installed elsewhere
set VCPKG_ROOT=C:\vcpkg

set INPUT_DIR=data\input
set OUTPUT_DIR=data\output
set CONTRAST=1.3
set BRIGHTNESS=15.0

REM ── Override defaults with command-line args ─────────────────
:parse
if "%~1"=="--arch"        ( set ARCH=%~2       & shift & shift & goto parse )
if "%~1"=="--input"       ( set INPUT_DIR=%~2  & shift & shift & goto parse )
if "%~1"=="--output"      ( set OUTPUT_DIR=%~2 & shift & shift & goto parse )
if "%~1"=="--contrast"    ( set CONTRAST=%~2   & shift & shift & goto parse )
if "%~1"=="--brightness"  ( set BRIGHTNESS=%~2 & shift & shift & goto parse )

echo =====================================================
echo  CUDA GPU Image Processing Pipeline  [Windows]
echo =====================================================
echo  ARCH       : %ARCH%
echo  Input dir  : %INPUT_DIR%
echo  Output dir : %OUTPUT_DIR%
echo  Contrast   : %CONTRAST%
echo  Brightness : %BRIGHTNESS%
echo =====================================================

REM ── Check nvcc ───────────────────────────────────────────────
where nvcc >nul 2>&1
if errorlevel 1 (
    echo [ERROR] nvcc not found. Install CUDA Toolkit and add to PATH.
    echo         Or use WSL2 instead ^(see WINDOWS_SETUP.md^).
    exit /b 1
)
echo [OK] nvcc found

REM ── Check Python ─────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not found. Install Python 3 from https://python.org
    exit /b 1
)
echo [OK] python found

REM ── Step 1: Generate test images ────────────────────────────
echo.
echo [Step 1/3] Generating synthetic PNG test images...
if not exist "%INPUT_DIR%" mkdir "%INPUT_DIR%"
python scripts\generate_test_images.py --count 20 --output %INPUT_DIR% --size 512x512
if errorlevel 1 ( echo [ERROR] Image generation failed. & exit /b 1 )

REM ── Step 2: Build ────────────────────────────────────────────
echo.
echo [Step 2/3] Building CUDA project...
if not exist build mkdir build

nvcc --std=c++17 ^
     -arch=%ARCH% ^
     -Iinclude ^
     -I"%VCPKG_ROOT%\installed\x64-windows\include" ^
     src\main.cu src\utils.cu ^
     -L"%VCPKG_ROOT%\installed\x64-windows\lib" ^
     -lpng -lnppif -lnppc -lnppig -lnppicc ^
     -o build\image_processor.exe

if errorlevel 1 (
    echo [ERROR] Build failed. Check CUDA Toolkit and vcpkg libpng installation.
    echo         See WINDOWS_SETUP.md for details.
    exit /b 1
)
echo [OK] Build successful: build\image_processor.exe

REM ── Step 3: Run ──────────────────────────────────────────────
echo.
echo [Step 3/3] Running pipeline...
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

build\image_processor.exe ^
    --input     %INPUT_DIR%  ^
    --output    %OUTPUT_DIR% ^
    --contrast  %CONTRAST%   ^
    --brightness %BRIGHTNESS%

if errorlevel 1 ( echo [ERROR] Pipeline execution failed. & exit /b 1 )

echo.
echo =====================================================
echo  Done! Output images are in: %OUTPUT_DIR%
echo =====================================================

REM Count output files
set /a count=0
for %%f in ("%OUTPUT_DIR%\*.png") do set /a count+=1
echo  Output PNGs: %count%
echo =====================================================

endlocal
