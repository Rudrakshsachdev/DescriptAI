# AGENTS.md — System & Autonomous Developer Guidelines for DESCЯA

This file outlines operational protocols, architectural rules, and technical standards for AI coding agents and human developers operating within the **Descra** repository.

---

## 1. Project Identification & Stack

- **Project Name:** Descra (AI-Powered Product Description Generator)
- **Phase:** Phase 1 — Deep Learning Foundation & Model Development
- **Operating System:** Windows 11 (AMD64)
- **Shell:** PowerShell
- **Python Version:** Python 3.10.11 (strictly in `.venv`, `C:\DescriptAI\.venv\Scripts\python.exe`)
- **PyTorch Version:** 2.6.0+cu124 (CUDA 12.4 runtime)
- **GPU Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6,140 MB VRAM)
- **Primary Model:** `google/flan-t5-small` (60M parameters, Seq2Seq)
- **Repository Remote:** `https://github.com/Rudrakshsachdev/DescriptAI` (`main` branch)

---

## 2. Strict Architectural Rules

1. **Local Model Execution Only:**
   - Under no circumstances use external generative AI APIs (OpenAI, Anthropic, Gemini, Cohere) for text generation.
   - All generation, evaluation, and fine-tuning must be executed locally on PyTorch using local Transformer checkpoints.

2. **No Monolithic Scripts:**
   - Preprocessing, model definition, training loops, evaluation metrics, and utilities must reside in separate, modular files under `src/descra/`.

3. **Configuration Driven:**
   - Hardcoded hyperparameters, file paths, and model dimensions are forbidden.
   - All variables belong in `configs/`:
     - `configs/data.yaml` (dataset paths, splits, gating limits)
     - `configs/model.yaml` (model name, dimensions, generation parameters)
     - `configs/training.yaml` (optimizer, batch size, learning rates, epochs)
     - `configs/evaluation.yaml` (metrics, test splits, thresholds)

4. **Hardware-Aware Execution:**
   - Target GPU has **6 GB VRAM**.
   - Maximum physical training batch size: `8` (with gradient accumulation for larger effective sizes).
   - Maximum input sequence length: `512` tokens.
   - Maximum target sequence length: `256` tokens.
   - Always verify CUDA execution with `torch.cuda.is_available()`.
   - Always run with FP16 mixed precision enabled.

5. **Data Isolation & Cleanliness:**
   - Never commit raw or processed datasets (`data/raw/*`, `data/processed/*`) to Git.
   - Never overwrite raw source data (`data/raw/`). Preprocessing must be pure, deterministic, and non-destructive.
   - Never train on validation or test splits. Zero cross-split data leakage is strictly enforced.

---

## 3. Directory Layout & Module Responsibilities

```
C:\DescriptAI
├── configs/                 # YAML configuration files
│   ├── data.yaml            # Dataset paths and schema constraints
│   ├── model.yaml           # Model architecture and decoding parameters
│   ├── training.yaml        # Training loop and optimizer hyperparameters
│   └── evaluation.yaml      # Metric thresholds and test configurations
├── data/                    # Dataset directory (data files git-ignored)
│   ├── raw/                 # Original source files (meta_All_Beauty.parquet)
│   ├── interim/             # Intermediate transformations
│   ├── processed/           # Canonical train, validation, test, needs_review JSONL
│   └── manifests/           # Dataset provenance and hash manifests
├── docs/                    # Architectural and learning documentation
│   ├── ROADMAP_PHASE1.md    # 10-step master phase 1 implementation plan
│   ├── ENVIRONMENT_SETUP.md # Python and PyTorch installation record
│   ├── DATASET_DOCUMENTATION.md # Schema and dataset documentation
│   ├── TRANSFORMER_ARCHITECTURE.md # Mathematical and architectural foundations
│   ├── LEARNING_ROADMAP.md  # 9-module team learning syllabus
│   ├── GLOSSARY.md          # 40+ Deep Learning / NLP terms with Descra examples
│   └── PRACTICAL_EXERCISES.md # 9 hands-on coding exercises with solutions
├── notebooks/               # Jupyter exploration and experimentation
│   └── 01_dataset_exploration.ipynb # Step 4 exploratory data analysis
├── reports/                 # Markdown reports, JSON metrics, figures
│   ├── hardware/            # Hardware audit reports
│   ├── dataset/             # Ingestion and preprocessing reports
│   └── eda/                 # EDA summary report, JSON metrics, and 6 figures
├── scripts/                 # Standalone execution entrypoints
│   ├── inspect_hardware.py  # GPU & VRAM diagnostic tool
│   ├── inspect_dataset.py   # Dataset schema inspection tool
│   ├── run_pipeline.py      # End-to-end data acquisition & preprocessing pipeline
│   ├── generate_eda.py      # Automated EDA report & figure generator
│   └── build_notebook.py    # Programmatic Jupyter notebook builder
├── src/descra/              # Production library package
│   ├── data/                # Data loading, inspection, preprocessing, splitting
│   │   ├── acquisition.py
│   │   ├── dataset.py
│   │   ├── inspection.py
│   │   ├── preprocessing.py
│   │   └── splitting.py
│   ├── models/              # Transformer models, baseline inference, fine-tuning
│   ├── evaluation/          # BLEU, ROUGE, BERTScore, latency profiling
│   └── utils/               # Hardware detection, configuration loaders, logging
└── tests/                   # Pytest automated test suite
    ├── test_hardware.py
    ├── test_preprocessing.py
    └── test_dataset.py
```

---

## 4. Standard Developer Commands

Run all commands using the project virtual environment:

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Run automated test suite
.\.venv\Scripts\python.exe -m pytest tests/ -v

# Run hardware diagnostic
.\.venv\Scripts\python.exe scripts/inspect_hardware.py

# Run EDA generation
.\.venv\Scripts\python.exe scripts/generate_eda.py

# Run unit tests with standard python launcher
python -m pytest tests/
```

---

## 5. Coding Standards for Agents

- **Type Hints:** Use standard Python 3.10+ type hints (`list[str]`, `dict[str, Any]`, `Path | None`).
- **Imports:** Group imports: standard library, third-party (`torch`, `transformers`, `pandas`), local (`descra.*`).
- **Paths:** Always use `pathlib.Path` instead of raw string concatenations for cross-platform compatibility.
- **Error Handling:** Catch explicit exceptions (`FileNotFoundError`, `ValueError`, `torch.cuda.CudaError`), log them via `descra.utils.logging`, and never swallow errors silently.
- **Testing:** Every new module in `src/descra/` must have a corresponding test suite in `tests/test_<module>.py`.
