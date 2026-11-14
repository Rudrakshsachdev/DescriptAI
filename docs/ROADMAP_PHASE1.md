# DESCЯA — PHASE 1 IMPLEMENTATION ROADMAP & RISK MATRIX
**Project:** Descra (AI Product Description Generator)  
**Phase:** 1 — Deep Learning Foundation & Model Development  
**Academic Year:** 2026-2027  
**Team Members:**
- Rudraksh Sachdeva (2481025)
- Shivansee (2481187)
- K Jahnvi (2481221)
- Kriti Abrol (2481214)

---

## 1. Project Phase 1 Architecture Overview

Descra Phase 1 focuses on building a **reproducible, factually grounded, locally executed Seq2Seq Transformer model** for e-commerce product description generation.

```
[ Structured Product Metadata ]
  ├── Product Name: "Sony WH-1000XM5"
  ├── Category: "Electronics"
  ├── Features: ["Industry Leading Noise Cancellation", "30h Battery Life", "Speak-to-Chat"]
  └── Specifications: {"Weight": "250g", "Bluetooth": "5.2"}
              │
              ▼
[ Deterministic Text-to-Text Prompt Formulation ]
  "Generate a concise, factual product description based on: ..."
              │
              ▼
[ Tokenizer: SentencePiece (FLAN-T5 Vocabulary) ]
  ├── Input IDs (padded/truncated to max 256 tokens)
  └── Attention Mask (1 for tokens, 0 for padding)
              │
              ▼
[ Transformer Architecture: FLAN-T5 Small (Encoder-Decoder) ]
  ├── Encoder: Bidirectional Self-Attention + Relative Position Bias
  ├── Cross-Attention: Decoder attends to Encoder hidden representations
  └── Decoder: Causal Masked Autoregressive Generation
              │
              ▼
[ Decoded Output Postprocessing ]
  ├── Whitespace normalization & special token stripping
  └── Output validation (length, repetition, hallucinations)
              │
              ▼
[ Rigorous Academic Evaluation ]
  ├── SacreBLEU (N-gram precision & brevity penalty)
  ├── ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
  ├── BERTScore (Semantic vector similarity)
  └── Hardware Latency (P50, P95, tokens/sec)
```

---

## 2. Milestone Execution Roadmap (Steps 1 to 10)

| Step | Milestone Name | Key Technical Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **Step 1** | **Repository Inspection & Hardware Assessment** | Hardware profiler (`src/descra/utils/hardware.py`), `scripts/inspect_hardware.py`, Hardware Assessment report, Dataset comparison report, Roadmap & Risk Matrix | **COMPLETED** |
| **Step 2** | **Environment & Dependency Setup** | Isolated Python 3.10 `.venv`, PyTorch 2.6.0+cu124 (CUDA 12.4 runtime verified with 1024x1024 GPU GEMM), Hugging Face `transformers 4.49.0`, `scikit-learn`, `matplotlib`, `seaborn` | **COMPLETED** |
| **Step 3** | **Dataset Acquisition & Preprocessing** | UCSD McAuley Amazon 2023 `All_Beauty` ingestion (`meta_All_Beauty.parquet`), canonical normalization, deduplication, factual alignment gating, zero-leakage 80/10/10 split (`train.jsonl`, `validation.jsonl`, `test.jsonl`, `needs_review.jsonl`), 21 unit tests | **COMPLETED** |
| **Step 4** | **Exploratory Data Analysis (EDA)** | `scripts/generate_eda.py`, `reports/eda/EDA_REPORT.md`, `reports/eda/eda_summary.json`, 6 publication figures in `reports/eda/figures/`, Jupyter notebook `notebooks/01_dataset_exploration.ipynb` | **COMPLETED** |
| **Step 5** | **Deep Learning Foundations & Team Documentation** | Theoretical foundations (`docs/TRANSFORMER_ARCHITECTURE.md`), 9-module syllabus (`docs/LEARNING_ROADMAP.md`), 40+ term glossary (`docs/GLOSSARY.md`), 9 hands-on exercises (`docs/PRACTICAL_EXERCISES.md`), contributor guide (`CONTRIBUTING.md`), autonomous guidelines (`AGENTS.md`) | **COMPLETED** |
| **Step 6** | **Baseline Model Development** | Model loader (`src/descra/models/loader.py`), generation pipeline (`src/descra/inference/pipeline.py`), multiple decoding strategies (Greedy, Beam Search, Sampling) | Pending Step 5 |
| **Step 7** | **Baseline Experimentation** | Benchmark pretrained FLAN-T5 zero-shot generation across 10 diverse categories, record outputs, latency, failure modes in JSON/CSV | Pending Step 6 |
| **Step 8** | **Fine-Tuning FLAN-T5** | PyTorch Seq2Seq training loop / HF Seq2SeqTrainer, AdamW optimizer, linear warmup scheduler, label masking (`-100`), gradient clipping, loss curve plots, best checkpoint saving | Pending Step 7 |
| **Step 9** | **Model Evaluation & Comparison** | SacreBLEU, ROUGE-1/2/L, BERTScore, latency benchmark (P50/P95), comparative analysis: Baseline vs. Fine-Tuned on identical held-out test split | Pending Step 8 |
| **Step 10**| **Final Review & Phase 1 Handover** | Complete test suite (`pytest tests/`), README, Viva preparation documentation, handover notes for Phase 2 FastAPI integration | Pending Step 9 |

---

## 3. Technical Risk Matrix & Mitigation Strategies

| Risk ID | Risk Description | Severity | Probability | Engineering Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **R-01** | **Python 3.13 / PyTorch CUDA Compatibility** | High | Medium | Python 3.13 is very recent. PyTorch 2.5/2.6 provides experimental wheels. If wheel issues arise, create a dedicated Python 3.11/3.12 virtual environment (`python -m venv .venv`) where PyTorch + CUDA 12.4/12.5 is rock solid. |
| **R-02** | **GPU VRAM OOM on 6GB RTX 4050** | High | Low | Primary model is `flan-t5-small` (~77M params) requiring <2.5GB training VRAM. Cap sequence lengths to `max_input_length=256` and `max_target_length=128`. Enable `fp16=True` mixed precision and gradient accumulation steps = 2. |
| **R-03** | **Data Leakage Across Splits** | Critical | Medium | Products with identical identifiers or near-identical descriptions must not leak across train and test sets. Group splits strictly by `product_id` and run an automated n-gram / exact-match cross-split leak detector before training. |
| **R-04** | **Hallucination of Product Attributes** | Medium | High | Construct input prompts with explicit factual grounding instructions: *"Generate a factual product description using ONLY the supplied specifications."* In evaluation, test token overlap between generated specs and input features. |
| **R-05** | **Windows DataLoader Multiprocessing Deadlocks** | Medium | Medium | PyTorch DataLoader worker processes on Windows use `spawn` which can cause slow initialization and lock contention. Set `num_workers=0` (or `num_workers=2` with `persistent_workers=False`). |
| **R-06** | **Network Sandbox Constraints** | Low | High | Standard terminal sandbox enforces zero-network mode. Download commands and package installations requiring network must use `BypassSandbox: true` while all preprocessing, training, testing, and evaluation run safely in standard sandboxed mode. |

---

## 4. Hardware-Aware Model Configuration

- **Primary Architecture:** `google/flan-t5-small`
  - Parameters: 77M
  - Encoder Layers: 6, Decoder Layers: 6
  - Hidden Dimension: 512, Attention Heads: 8
  - Target Memory in FP16: ~150 MB (Inference), ~2.2 GB (Training with AdamW)
- **Batch Sizing:**
  - `per_device_train_batch_size: 8`
  - `gradient_accumulation_steps: 2` (Effective batch size = 16)
  - `per_device_eval_batch_size: 16`
- **Optimizer:** `AdamW(lr=3e-4, weight_decay=0.01, eps=1e-8)` with linear warmup (10% of total steps)
- **Precision:** `FP16` (NVIDIA Tensor Core acceleration on RTX 4050 Laptop GPU)

