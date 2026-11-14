# DESCЯA — ENVIRONMENT SETUP & HARDWARE ACCELERATION GUIDE
**Phase 1: Deep Learning Foundation & Model Development**  
**Date:** 2026-11-14  
**Project:** Descra (AI Product Description Generator)  
**Authors:** Rudraksh Sachdeva (2481025), Shivansee (2481187), K Jahnvi (2481221), Kriti Abrol (2481214)

---

## 1. System Requirements & Runtime Architecture

- **Operating System:** Windows 11 (64-bit AMD64)
- **Python Version:** Python 3.10.11 (or 3.11.x)
- **Host GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
- **NVIDIA Driver:** >= 527.41 (Tested: Driver Version 555.97)
- **CUDA Runtime:** CUDA 12.4 (`cu124` PyTorch release)

> [!IMPORTANT]
> Never install ML dependencies into the global Python runtime. All dependencies are strictly encapsulated within the project-level `.venv` virtual environment to preserve environment reproducibility and prevent DLL/C-extension collisions.

---

## 2. Step-by-Step Environment Setup

### Step 2.1: Virtual Environment Creation
From the repository root `C:\DescriptAI`:

```powershell
# Verify Python 3.10 installation
& "C:\Program Files\Python310\python.exe" --version

# Create isolated .venv environment
& "C:\Program Files\Python310\python.exe" -m venv .venv
```

*(Note: If Python 3.11 is preferred, install via official Python.org installer `python-3.11.9-amd64.exe` or `winget install Python.Python.3.11`, then replace the executable path).*

### Step 2.2: Virtual Environment Activation
In Windows PowerShell:

```powershell
# If script execution is restricted on Windows, allow CurrentUser:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### Step 2.3: PyTorch with CUDA 12.4 Installation
Install the official CUDA-accelerated PyTorch build matching Driver 555.97:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Step 2.4: Core NLP & Model Training Dependencies
```powershell
.\.venv\Scripts\python.exe -m pip install transformers datasets accelerate tokenizers sentencepiece safetensors pandas scikit-learn matplotlib seaborn pyyaml tqdm beautifulsoup4 lxml pytest pytest-cov sacrebleu rouge-score bert-score evaluate
```

---

## 3. Hardware Verification & Benchmarking

Verify full GPU tensor initialization, matrix multiplication, and GPU synchronization:

```powershell
.\.venv\Scripts\python.exe scripts/verify_environment.py
```

### Expected Output:
```text
=================================================================
DESCRA — ENVIRONMENT & HARDWARE ACCELERATION VERIFICATION
=================================================================
Python Executable: C:\DescriptAI\.venv\Scripts\python.exe
Python Version:    3.10.11
Isolated .venv:    True (Prefix: C:\DescriptAI\.venv)
-----------------------------------------------------------------
PyTorch Version:      2.6.0+cu124
Transformers Version: 5.20.2
Datasets Version:     5.2.0
Accelerate Version:   1.15.0
Tokenizers Version:   0.23.3
SentencePiece Ver:    0.2.3
-----------------------------------------------------------------
CUDA Available (PyTorch): True
PyTorch CUDA Runtime:     CUDA 12.4
Detected GPU Devices:     1
Active GPU Index:         0 (NVIDIA GeForce RTX 4050 Laptop GPU)
Dedicated VRAM (Total):   6140.5 MB (6.00 GB)
Available VRAM (Free):    5075.0 MB (4.96 GB)
-----------------------------------------------------------------
Executing GPU Tensor Allocation & Matrix Multiplication Benchmark...
Matrix Dimension:         1024 x 1024 (FP32)
GPU Synchronization:      VERIFIED (torch.cuda.synchronize() succeeded)
Tensor Placement:         VERIFIED (Allocated on cuda:0)
Output Numerical Finite:  VERIFIED (True)
VRAM Post-Execution:      20.12 MB Allocated
=================================================================
STATUS: ENVIRONMENT & CUDA ACCELERATION FULLY VERIFIED
=================================================================
```

---

## 4. Troubleshooting & Windows-Specific Notes

1. **CUDA driver call isolation in Sandboxes:**
   On Windows, low-level NVIDIA driver communication (`nvml.dll` and `nvcuda.dll`) requires direct host driver handles. Standard container sandboxes without GPU passthrough fallback cleanly to CPU. Run verification outside isolated sandboxes to utilize the RTX 4050.
2. **PyTorch DataLoader Multi-processing on Windows:**
   Always set `dataloader_num_workers: 0` in training configurations to prevent Windows `spawn` child process deadlocks.
3. **CPU Fallback Mode:**
   If CUDA is unavailable on another collaborator's system, `scripts/verify_environment.py` and all Descra modules automatically fallback to CPU mode (`torch.device("cpu")`), scaling batch size to 2-4.

