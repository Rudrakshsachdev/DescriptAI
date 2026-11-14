# DESCЯA — HARDWARE ASSESSMENT & ENVIRONMENT REPORT
**Date:** 2026-11-14  
**Project:** Descra (AI Product Description Generator) — Phase 1  
**Lead Authors:** Rudraksh Sachdeva (2481025), Shivansee (2481187), K Jahnvi (2481221), Kriti Abrol (2481214)

---

## 1. System Specification Summary

| Component | Detected Specification | Engineering Assessment |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 (Build 10.0.26300) | Windows 64-bit host environment |
| **Python Runtime** | Python 3.13.2 (AMD64) | Modern 64-bit CPython runtime |
| **CPU Architecture** | Intel64 Family 6 Model 154 Stepping 3, GenuineIntel | Intel 12th/13th Gen Hybrid Architecture |
| **CPU Core Topology** | 10 Physical Cores / 16 Logical Threads | Multi-threaded data collation & preprocessing ready |
| **System Memory (RAM)** | 15.64 GB Total / 2.70 GB Available | Sufficient for in-memory tokenization and data pipeline |
| **Disk Storage (C:)** | 475.30 GB Total / 40.35 GB Free | Adequate for raw datasets (~500MB) & model checkpoints (~1-2GB) |
| **GPU Model** | NVIDIA GeForce RTX 4050 Laptop GPU (6141 MB VRAM, Free: 5921 MB) | Dedicated Ada Lovelace architecture GPU |
| **NVIDIA Driver / CUDA** | Driver: 555.97 | CUDA Toolkit Support: 12.5 |
| **PyTorch Status** | Installed: `False` (CUDA: `False`) | Required torch+cuda wheels to be configured |

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
  mixed_precision: "fp16"
  primary_model: "google/flan-t5-small"
  train_batch_size: 8
  eval_batch_size: 16
  gradient_accumulation_steps: 2
  effective_batch_size: 16
  dataloader_num_workers: 0  # 0 recommended on Windows to prevent multi-process fork deadlocks
  pin_memory: true
```

---

## 3. Potential Bottlenecks & Mitigations

1. **Available RAM Overhead (2.70 GB Available):**
   - *Risk:* Loading full multi-gigabyte raw JSON files into memory could trigger OOM or paging.
   - *Mitigation:* Stream datasets or extract a curated 5,000–10,000 record subset using memory-efficient generators or Parquet/Arrow streaming.
2. **Windows Multiprocessing in PyTorch DataLoader:**
   - *Risk:* PyTorch on Windows spawns new processes rather than forking, which can cause IPC latency or permission errors.
   - *Mitigation:* Set `num_workers=0` or `num_workers=2` with `persistent_workers=False`.
3. **GPU VRAM Ceiling (6.0 GB VRAM):**
   - *Risk:* Long context sequences (e.g. >512 tokens) or large batch sizes can exhaust VRAM.
   - *Mitigation:* Constrain `max_input_length=256` and `max_target_length=128`. Enable PyTorch gradient checkpointing and mixed-precision (`torch.cuda.amp.autocast`).
