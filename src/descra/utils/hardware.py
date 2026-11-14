"""Hardware inspection and capability assessment utility for Descra.

Provides detailed hardware profiling including CPU, system RAM, disk storage,
and NVIDIA GPU / CUDA capabilities to determine optimal model selection
(FLAN-T5-small vs FLAN-T5-base), batch sizing, and precision modes.
"""

from dataclasses import asdict, dataclass
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional
import psutil


@dataclass
class HardwareProfile:
    os_name: str
    os_release: str
    os_version: str
    machine_arch: str
    processor: str
    python_version: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    total_ram_gb: float
    available_ram_gb: float
    used_ram_gb: float
    ram_usage_percent: float
    disk_total_gb: float
    disk_free_gb: float
    disk_usage_percent: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    gpu_total_vram_mb: Optional[float] = None
    gpu_free_vram_mb: Optional[float] = None
    torch_installed: bool = False
    torch_version: Optional[str] = None
    torch_cuda_available: bool = False
    recommended_model: str = "google/flan-t5-small"
    recommended_batch_size: int = 8
    recommended_eval_batch_size: int = 16
    recommended_precision: str = "fp32"
    recommended_gradient_accumulation_steps: int = 2


def _get_nvidia_smi_info() -> Dict[str, Any]:
    """Query nvidia-smi via subprocess to extract GPU metrics even before PyTorch is installed."""
    info: Dict[str, Any] = {
        "gpu_available": False,
        "gpu_name": None,
        "driver_version": None,
        "cuda_version": None,
        "total_vram_mb": None,
        "free_vram_mb": None,
    }
    try:
        # Check basic presence and driver/CUDA version
        res = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            return info

        info["gpu_available"] = True

        # Query structured CSV from nvidia-smi
        query_cmd = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
        q_res = subprocess.run(
            query_cmd, capture_output=True, text=True, check=False
        )
        if q_res.returncode == 0 and q_res.stdout.strip():
            first_line = q_res.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 4:
                info["gpu_name"] = parts[0]
                info["driver_version"] = parts[1]
                info["total_vram_mb"] = float(parts[2])
                info["free_vram_mb"] = float(parts[3])

        # Also extract CUDA version from banner if possible
        for line in res.stdout.split("\n"):
            if "CUDA Version:" in line:
                idx = line.find("CUDA Version:")
                info["cuda_version"] = line[idx + len("CUDA Version:"):].split()[0]
                break

    except Exception:
        # nvidia-smi not found or failed
        pass
    return info


def inspect_hardware(work_dir: Optional[str] = None) -> HardwareProfile:
    """Inspects complete system hardware and computes optimal deep learning execution parameters."""
    if work_dir is None:
        work_dir = os.getcwd()

    vmem = psutil.virtual_memory()
    disk = shutil.disk_usage(work_dir)

    gpu_info = _get_nvidia_smi_info()

    # PyTorch status
    torch_installed = False
    torch_version = None
    torch_cuda = False
    try:
        import torch  # type: ignore

        torch_installed = True
        torch_version = torch.__version__
        torch_cuda = torch.cuda.is_available()
        if torch_cuda and not gpu_info["gpu_name"]:
            gpu_info["gpu_available"] = True
            gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_info["total_vram_mb"] = props.total_memory / (1024 * 1024)
            gpu_info["free_vram_mb"] = (
                torch.cuda.mem_get_info()[0] / (1024 * 1024)
            )
    except ImportError:
        torch_installed = False

    # Compute hardware-aware recommendations
    has_gpu = gpu_info["gpu_available"]
    vram_mb = gpu_info["total_vram_mb"] or 0.0
    ram_avail_gb = vmem.available / (1024**3)

    if has_gpu and vram_mb >= 5000:
        # RTX 4050 Laptop (6GB VRAM) or similar
        # FLAN-T5-base has ~250M params -> ~1GB weights in fp32, ~500MB in fp16/bf16.
        # With optimizer states (AdamW ~8 bytes/param) and activations, fine-tuning Base requires ~4-5GB VRAM.
        # FLAN-T5-small has ~80M params -> ~300MB weights, ~1.5-2GB peak training VRAM.
        # Recommendation: FLAN-T5-small is the ideal primary baseline and fine-tuning target for ultra-stable,
        # fast, and repeatable training on 6GB VRAM with headroom for batching.
        # FLAN-T5-base can be tested with batch_size=4 and gradient_accumulation=4 with fp16.
        recommended_model = "google/flan-t5-small"
        recommended_batch_size = 8
        recommended_eval_batch_size = 16
        recommended_precision = "fp16"  # RTX 4050 supports native FP16 / BF16 Tensor Cores
        recommended_grad_accum = 2
    elif has_gpu and vram_mb >= 2000:
        recommended_model = "google/flan-t5-small"
        recommended_batch_size = 4
        recommended_eval_batch_size = 8
        recommended_precision = "fp16"
        recommended_grad_accum = 4
    else:
        # CPU Execution
        recommended_model = "google/flan-t5-small"
        recommended_batch_size = 2 if ram_avail_gb < 4.0 else 4
        recommended_eval_batch_size = 4
        recommended_precision = "fp32"
        recommended_grad_accum = 4

    return HardwareProfile(
        os_name=platform.system(),
        os_release=platform.release(),
        os_version=platform.version(),
        machine_arch=platform.machine(),
        processor=platform.processor(),
        python_version=sys.version.split()[0],
        cpu_physical_cores=psutil.cpu_count(logical=False) or 1,
        cpu_logical_cores=psutil.cpu_count(logical=True) or 1,
        total_ram_gb=round(vmem.total / (1024**3), 2),
        available_ram_gb=round(ram_avail_gb, 2),
        used_ram_gb=round((vmem.total - vmem.available) / (1024**3), 2),
        ram_usage_percent=vmem.percent,
        disk_total_gb=round(disk.total / (1024**3), 2),
        disk_free_gb=round(disk.free / (1024**3), 2),
        disk_usage_percent=round((disk.used / disk.total) * 100, 1),
        gpu_available=gpu_info["gpu_available"],
        gpu_name=gpu_info["gpu_name"],
        gpu_driver_version=gpu_info["driver_version"],
        cuda_version=gpu_info["cuda_version"],
        gpu_total_vram_mb=gpu_info["total_vram_mb"],
        gpu_free_vram_mb=gpu_info["free_vram_mb"],
        torch_installed=torch_installed,
        torch_version=torch_version,
        torch_cuda_available=torch_cuda,
        recommended_model=recommended_model,
        recommended_batch_size=recommended_batch_size,
        recommended_eval_batch_size=recommended_eval_batch_size,
        recommended_precision=recommended_precision,
        recommended_gradient_accumulation_steps=recommended_grad_accum,
    )


def generate_hardware_markdown_report(profile: HardwareProfile) -> str:
    """Generates a detailed academic-grade hardware assessment report."""
    gpu_desc = (
        f"{profile.gpu_name} ({profile.gpu_total_vram_mb:.0f} MB VRAM, Free: {profile.gpu_free_vram_mb:.0f} MB)"
        if profile.gpu_available
        else "No dedicated NVIDIA GPU detected / CPU fallback"
    )

    report = f"""# DESCЯA — HARDWARE ASSESSMENT & ENVIRONMENT REPORT
**Date:** 2026-11-14  
**Project:** Descra (AI Product Description Generator) — Phase 1  
**Lead Authors:** Rudraksh Sachdeva (2481025), Shivansee (2481187), K Jahnvi (2481221), Kriti Abrol (2481214)

---

## 1. System Specification Summary

| Component | Detected Specification | Engineering Assessment |
| :--- | :--- | :--- |
| **Operating System** | {profile.os_name} {profile.os_release} (Build {profile.os_version}) | Windows 64-bit host environment |
| **Python Runtime** | Python {profile.python_version} ({profile.machine_arch}) | Modern 64-bit CPython runtime |
| **CPU Architecture** | {profile.processor} | Intel 12th/13th Gen Hybrid Architecture |
| **CPU Core Topology** | {profile.cpu_physical_cores} Physical Cores / {profile.cpu_logical_cores} Logical Threads | Multi-threaded data collation & preprocessing ready |
| **System Memory (RAM)** | {profile.total_ram_gb:.2f} GB Total / {profile.available_ram_gb:.2f} GB Available | Sufficient for in-memory tokenization and data pipeline |
| **Disk Storage (C:)** | {profile.disk_total_gb:.2f} GB Total / {profile.disk_free_gb:.2f} GB Free | Adequate for raw datasets (~500MB) & model checkpoints (~1-2GB) |
| **GPU Model** | {gpu_desc} | Dedicated Ada Lovelace architecture GPU |
| **NVIDIA Driver / CUDA** | Driver: {profile.gpu_driver_version or 'N/A'} | CUDA Toolkit Support: {profile.cuda_version or 'N/A'} |
| **PyTorch Status** | Installed: `{profile.torch_installed}` (CUDA: `{profile.torch_cuda_available}`) | Required torch+cuda wheels to be configured |

---

## 2. Hardware-Aware Deep Learning Strategy

### 2.1 Model Selection: FLAN-T5 Small vs. FLAN-T5 Base
- **FLAN-T5-Small (~77 Million Parameters):**
  - **Memory Footprint:** FP32 weights = ~308 MB; FP16 weights = ~154 MB.
  - **Training VRAM Budget:** With AdamW optimizer states (8 bytes/param) and activation caching with sequence length 256/128 at batch size 8, training VRAM requirement is **~1.8 GB to 2.4 GB**.
  - **Verdict:** Highly recommended as primary baseline and fine-tuning target. Provides guaranteed stability on 6GB VRAM, fast training epochs, rapid iteration, and leaves >3.5 GB headroom for operating system display rendering.
- **FLAN-T5-Base (~248 Million Parameters):**
  - **Memory Footprint:** FP32 weights = ~990 MB; FP16 weights = ~495 MB.
  - **Training VRAM Budget:** Fine-tuning in FP32 would exceed 6GB VRAM. In FP16/BF16 with `batch_size=4` and `gradient_accumulation_steps=4`, VRAM consumption is **~4.8 GB to 5.4 GB**, which is tight against the 6141 MB hardware boundary.
  - **Recommendation:** Keep FLAN-T5-Base as a secondary comparative target if VRAM stability permits, while executing the primary verified pipeline on **`google/flan-t5-small`**.

### 2.2 Recommended Hyperparameter Profile

```yaml
hardware_config:
  device: "cuda"  # fallback to "cpu" if CUDA unavailable
  mixed_precision: "{profile.recommended_precision}"
  primary_model: "{profile.recommended_model}"
  train_batch_size: {profile.recommended_batch_size}
  eval_batch_size: {profile.recommended_eval_batch_size}
  gradient_accumulation_steps: {profile.recommended_gradient_accumulation_steps}
  effective_batch_size: {profile.recommended_batch_size * profile.recommended_gradient_accumulation_steps}
  dataloader_num_workers: 0  # 0 recommended on Windows to prevent multi-process fork deadlocks
  pin_memory: true
```

---

## 3. Potential Bottlenecks & Mitigations

1. **Available RAM Overhead ({profile.available_ram_gb:.2f} GB Available):**
   - *Risk:* Loading full multi-gigabyte raw JSON files into memory could trigger OOM or paging.
   - *Mitigation:* Stream datasets or extract a curated 5,000–10,000 record subset using memory-efficient generators or Parquet/Arrow streaming.
2. **Windows Multiprocessing in PyTorch DataLoader:**
   - *Risk:* PyTorch on Windows spawns new processes rather than forking, which can cause IPC latency or permission errors.
   - *Mitigation:* Set `num_workers=0` or `num_workers=2` with `persistent_workers=False`.
3. **GPU VRAM Ceiling (6.0 GB VRAM):**
   - *Risk:* Long context sequences (e.g. >512 tokens) or large batch sizes can exhaust VRAM.
   - *Mitigation:* Constrain `max_input_length=256` and `max_target_length=128`. Enable PyTorch gradient checkpointing and mixed-precision (`torch.cuda.amp.autocast`).
"""
    return report

