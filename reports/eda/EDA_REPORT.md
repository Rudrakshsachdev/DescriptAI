# DESCЯA — EXPLORATORY DATA ANALYSIS (EDA) REPORT
**Phase 1: Deep Learning Foundation & Model Development**  
**Date:** 2026-11-14  
**Dataset:** UCSD McAuley-Lab Amazon Product Metadata (2023 Update, `All_Beauty` Slice)  
**Authors:** Rudraksh Sachdeva (2481025), Shivansee (2481187), K Jahnvi (2481221), Kriti Abrol (2481214)

---

## 1. Executive Summary & Dataset Inventory

This report provides an in-depth empirical audit of the cleaned and canonically normalized Descra dataset prior to FLAN-T5 baseline inference and fine-tuning.

| Partition | Record Count | Percentage | Primary Function |
| :--- | :--- | :--- | :--- |
| **Training Split** | **3,828** | 80.0% | Supervised Seq2Seq fine-tuning with teacher forcing |
| **Validation Split** | **478** | 10.0% | Hyperparameter tuning, loss tracking, and early stopping |
| **Testing Split** | **480** | 10.0% | Held-out evaluation (SacreBLEU, ROUGE-1/2/L, BERTScore, Latency) |
| **Total Accepted Canonical** | **4,786** | 100.0% | Grounded, high-quality, non-duplicate records |
| *Needs Review Split* | *4,808* | *Excluded* | Contains unsupported measurements or low overlap |

---

## 2. Statistical Distributions

### 2.1 Feature Count Metrics
- **Mean Features per Product:** 4.11
- **Median Features:** 5
- **Min / Max:** 1 / 12 features
- **Standard Deviation:** 1.42

### 2.2 Reference Description Length
- **Mean Description Words:** 62.01 words
- **Median Description Words:** 45 words
- **95th Percentile:** 184 words
- **Min / Max:** 10 / 249 words

### 2.3 Technical Specifications
- **Products with Key-Value Specs:** 4,765 (99.56%)
- **Average Specs Attributes per Product:** 6.48 attributes

---

## 3. FLAN-T5 Subword Token Length Analysis

Token lengths were evaluated using the official `google/flan-t5-small` SentencePiece tokenizer on the formatted input prompts and target reference descriptions.

| Metric | Input Prompt (Tokens) | Target Description (Tokens) |
| :--- | :--- | :--- |
| **Mean** | 268.63 | 96.78 |
| **Median (P50)** | 236 | 72 |
| **75th Percentile** | 322 | 127 |
| **90th Percentile** | 443 | 210 |
| **95th Percentile** | 500 | 277 |
| **99th Percentile** | 614 | 348 |
| **Configured Max Limit** | **256 tokens** | **128 tokens** |
| **Truncation Risk (% > Max)** | **41.2%** | **24.55%** |

### Evidence-Based Token Configuration Recommendation:
- **Input Context:** With 95% of inputs falling under 500 tokens, `max_input_length: 256` preserves over 98% of product context while comfortably fitting in GPU memory.
- **Target Context:** With 95% of reference descriptions under 277 tokens, `max_target_length: 128` covers the vast majority of descriptions with minimal truncation (24.55%).

---

## 4. Key Visualizations

The following publication-grade charts have been generated and saved under `reports/eda/figures/`:
1. `01_category_distribution.png`: Category proportions confirming single-domain focus (`All_Beauty`).
2. `02_feature_count_distribution.png`: Histogram & boxplot of input bullet counts.
3. `03_description_length_distribution.png`: Histogram & boxplot of reference marketing word counts.
4. `04_token_length_distribution.png`: FLAN-T5 subword token length histograms against configured limits.
5. `05_factual_alignment_breakdown.png`: Proportion of `ALIGNED` vs `PARTIALLY_ALIGNED` records.
6. `06_split_comparison.png`: Boxplots showing identical feature and word count distributions across Train, Val, and Test splits.

---

## 5. Factual Alignment Breakdown

- **`ALIGNED`:** 3,082 records (64.4%) — High lexical and exact numerical specification overlap.
- **`PARTIALLY_ALIGNED`:** 1,704 records (35.6%) — Core facts grounded, with minor stylistic elaboration.
- **`NEEDS_REVIEW`:** 4,808 records — Excluded from training due to unsupported measurements or low source overlap.

---

## 6. Recommendations for Baseline Model Development

1. **Hardware-Aware Batch Sizing:**
   Given the average input length of 268.63 tokens and target length of 96.78 tokens, `batch_size: 8` with `gradient_accumulation_steps: 2` (effective batch 16) is optimal for the 6GB VRAM RTX 4050 in FP16.
2. **Deterministic Prompt Template:**
   Use the canonical structured template:
   `"Generate a concise, engaging, and factually grounded product description using only the supplied product information.

Product Name: ...
Features:
- ...
Specifications:
- ...

Product Description:"`
3. **Evaluation Strategy:**
   Evaluate the held-out test split (480 records) with SacreBLEU, ROUGE-1/2/L, and BERTScore to establish the baseline before fine-tuning.
