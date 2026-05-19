# Running on Windows — Complete Setup Guide

## Option 1: WSL2 + CUDA (Recommended)

WSL2 lets you run Linux on Windows with full NVIDIA GPU access.
This is the officially supported way to run CUDA on Windows for development.

---

### Step 1: Install WSL2

Open **PowerShell as Administrator** and run:

```powershell
wsl --install
```

Restart your PC when prompted. This installs Ubuntu by default.

If you already have WSL1, upgrade to WSL2:
```powershell
wsl --set-default-version 2
wsl --set-version Ubuntu 2
```

---

### Step 2: Install NVIDIA drivers for WSL2

Download and install the **CUDA on WSL** driver from NVIDIA:
https://developer.nvidia.com/cuda/wsl

> IMPORTANT: Install the Windows driver — do NOT install a separate Linux
> driver inside WSL. The Windows driver handles both.

Verify inside WSL:
```bash
nvidia-smi
```
You should see your GPU listed.

---

### Step 3: Install CUDA Toolkit inside WSL2

Open your Ubuntu WSL terminal and run:

```bash
# Add NVIDIA CUDA repo for Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-3

# Add to PATH (add these lines to ~/.bashrc too)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
```

---

### Step 4: Install libpng

```bash
sudo apt update && sudo apt install -y libpng-dev
```

---

### Step 5: Copy project into WSL and run

In WSL terminal:

```bash
# Navigate to your Windows files (e.g. Downloads folder)
cd /mnt/c/Users/<YourWindowsUsername>/Downloads

# Unzip the project
unzip cuda_image_processor.zip
cd cuda_image_processor

# Find your GPU arch (look at "Compute Capability" row)
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# Edit run.sh to set the correct arch (e.g. sm_86 for RTX 3080)
nano run.sh
# Change: ARCH="sm_75"  →  ARCH="sm_XX"   (your GPU's value)

# Run everything
bash run.sh
```

Output images appear in `data/output/` — accessible from Windows at:
`\\wsl$\Ubuntu\home\<user>\...` or via File Explorer → Linux section.

---

## Option 2: Native Windows with Visual Studio (Advanced)

If you cannot use WSL2, you can build natively on Windows.
This requires more manual setup.

### Requirements
- Visual Studio 2022 (Community is free) with "Desktop development with C++"
- CUDA Toolkit for Windows: https://developer.nvidia.com/cuda-downloads
- libpng for Windows (via vcpkg)

### Step 1: Install vcpkg and libpng
```powershell
# In PowerShell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install libpng:x64-windows
.\vcpkg integrate install
```

### Step 2: Generate test images (Python required)
```powershell
python scripts\generate_test_images.py --count 20 --output data\input
```

### Step 3: Build with nvcc on Windows
Open **x64 Native Tools Command Prompt for VS 2022**, then:

```cmd
cd cuda_image_processor

nvcc --std=c++17 ^
     -arch=sm_75 ^
     -Iinclude ^
     -IC:\vcpkg\installed\x64-windows\include ^
     src\main.cu src\utils.cu ^
     -LC:\vcpkg\installed\x64-windows\lib ^
     -lpng -lnppif -lnppc -lnppig -lnppicc ^
     -o build\image_processor.exe
```

### Step 4: Run
```cmd
mkdir data\output
build\image_processor.exe ^
    --input data\input ^
    --output data\output ^
    --contrast 1.3 ^
    --brightness 15.0
```

---

## Checking Your GPU Arch Flag

| GPU | Compute Cap | Flag |
|-----|------------|------|
| GTX 1060/1080/1080 Ti | 6.1 | sm_61 |
| GTX 1650/1660 | 7.5 | sm_75 |
| RTX 2060/2070/2080 | 7.5 | sm_75 |
| RTX 3060/3070/3080/3090 | 8.6 | sm_86 |
| RTX 4060/4070/4080/4090 | 8.9 | sm_89 |
| Tesla T4 | 7.5 | sm_75 |
| Tesla V100 | 7.0 | sm_70 |
| A100 | 8.0 | sm_80 |

Find yours:
```powershell
# Windows PowerShell
nvidia-smi --query-gpu=name,compute_cap --format=csv
```
