# DESCЯA — AI-Powered Product Description Generator

## Phase 1: Deep Learning Foundation & Model Development

### Academic Project Overview
**DESCЯA** is an end-to-end Deep Learning system engineered to generate concise, meaningful, engaging, and factually grounded e-commerce product descriptions from structured technical attributes.

### Team Members
- **Rudraksh Sachdeva** — 2481025
- **Shivansee** — 2481187
- **K Jahnvi** — 2481221
- **Kriti Abrol** — 2481214

---

## Hardware Profile & Execution Configuration

Automated profiling via `python scripts/inspect_hardware.py`:
- **Operating System:** Windows 11 (64-bit AMD64)
- **CPU:** Intel 12th/13th Gen Core (10 Physical Cores / 16 Logical Threads)
- **RAM:** 15.64 GB Total (~2.70 GB Available)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MB VRAM)
- **NVIDIA Driver:** 555.97 | **CUDA Support:** 12.5
- **Primary Architecture Selected:** `google/flan-t5-small` (~77M parameters)
- **Target Precision:** FP16 mixed precision
- **Recommended Batch Size:** 8 (with 2 gradient accumulation steps = effective batch size 16)

---

## Directory Architecture

```
descra/
│
├── data/
│   ├── raw/                  # Original raw immutable source subsets
│   ├── interim/              # Sanitized intermediate formats
│   ├── processed/            # Canonical train/val/test splits
│   └── manifests/            # Dataset provenance and hash manifests
│
├── configs/
│   ├── data.yaml             # Dataset ingestion, filtering & split configs
│   ├── model.yaml            # FLAN-T5 tokenizer and decoding parameters
│   ├── training.yaml         # Optimizer, learning rate, and checkpoint configs
│   └── evaluation.yaml       # SacreBLEU, ROUGE, BERTScore, latency configs
│
├── src/
│   └── descra/
│       ├── utils/            # Hardware inspection, logging, reproducibility
│       ├── data/             # Acquisition, inspection, preprocessing, splitting
│       ├── models/           # Loader, generator, configuration
│       ├── training/         # Trainer, callbacks, checkpointing
│       ├── evaluation/       # BLEU, ROUGE, BERTScore, latency, comparison
│       └── inference/        # Generation pipeline and postprocessing
│
├── scripts/
│   ├── inspect_hardware.py   # System hardware profiling utility
│   ├── download_dataset.py   # Controlled subset acquisition
│   ├── prepare_dataset.py    # Preprocessing and deterministic splitting
│   ├── train.py              # Supervised Seq2Seq fine-tuning
│   ├── evaluate.py           # Evaluation on held-out test split
│   └── generate.py           # Interactive / batch description generation
│
├── reports/
│   ├── hardware/             # System specs & hardware recommendation
│   ├── dataset/              # Dataset research, EDA, quality analysis
│   ├── training/             # Loss curves and checkpoint metrics
│   └── evaluation/           # SacreBLEU, ROUGE, BERTScore, latency benchmarks
│
├── docs/
│   ├── ROADMAP_PHASE1.md     # Milestone tracking and risk mitigation
│   ├── DATASET_DOCUMENTATION.md
│   ├── TRANSFORMER_ARCHITECTURE.md
│   └── VIVA_PREPARATION.md
│
├── tests/                    # Pytest test suite
├── pyproject.toml            # Project packaging and metadata
└── requirements.txt          # Python dependencies
```

---

## Developer Execution Commands

### Hardware Profiling
```powershell
.\.venv\Scripts\python.exe scripts/inspect_hardware.py
```

### Dataset EDA Report & Publication Figures
```powershell
.\.venv\Scripts\python.exe scripts/generate_eda.py
```

### Baseline Model Inference
```powershell
# Interactive / Sample product description generation with FLAN-T5-small
.\.venv\Scripts\python.exe scripts/run_baseline.py --sample
```

### Baseline Model Validation Evaluation (Greedy vs. Beam Search)
```powershell
# Evaluates BLEU, ROUGE-1/2/L, factual adherence, and latency on validation set
.\.venv\Scripts\python.exe scripts/evaluate_baseline.py --sample-size 100
```

### GPU Latency & VRAM Benchmarking Matrix
```powershell
# Benchmarks batch sizes {1, 4, 8} and sequence lengths {128, 256, 512} on RTX 4050
.\.venv\Scripts\python.exe scripts/benchmark_baseline.py
```

### Automated Pytest Suite
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```


