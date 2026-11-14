# DESCЯA — DATASET DOCUMENTATION & PROVENANCE SPECIFICATION
**Phase 1: Deep Learning Foundation & Model Development**  
**Dataset Version:** `v1.0`  
**Date:** 2026-11-14  
**Authors:** Rudraksh Sachdeva (2481025), Shivansee (2481187), K Jahnvi (2481221), Kriti Abrol (2481214)

---

## 1. Dataset Origin & Academic Licensing

- **Dataset Source:** UCSD E-Commerce Dataset (Julian McAuley Lab, University of California San Diego)
- **Version:** `Amazon-Reviews-2023 (McAuley-Lab)`
- **Category:** `All_Beauty` (Primary e-commerce catalog slice rich in product feature bullets, formulation specs, and manufacturer descriptions)
- **Official Source URL:** `https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023`
- **Academic Attribution:**
  > Jianmo Ni, Jiacheng Li, Julian McAuley. *Justifying recommendations using distantly-labeled reviews and fine-grained aspects.* In EMNLP-IJCNLP, 2019.  
  > Hou et al. *Bridging Language and Items for Recommendation and Retrieval.* arXiv:2403.03952, 2024.
- **License Terms:** Open for Academic, Educational, and Non-Commercial Research.
- **Redistribution Policy:** Raw files remain outside version control in `data/raw/` with integrity tracked by SHA-256 in `data/manifests/source_manifest.json`.

---

## 2. Canonical Dataset Schema

Each processed record is structured into the canonical Descra schema:

```json
{
  "product_id": "B01CUPMQZE",
  "product_name": "Howard LC0008 Leather Conditioner, 8-Ounce",
  "category": "All Beauty",
  "features": [
    "Material: 304 Stainless Steel; Brass tip",
    "Lengths Available: 88mm, 93mm, 98mm",
    "Accepts cartridge needles with vice style tattoo machines"
  ],
  "specifications": {
    "Package Dimensions": "7.1 x 5.5 x 3 inches; 2.38 Pounds",
    "UPC": "617390882781"
  },
  "reference_description": "Howard LC0008 Leather Conditioner deeply penetrates leather fibers to restore natural oils...",
  "alignment_status": "ALIGNED",
  "alignment_score": 0.42
}
```

---

## 3. Data Cleaning & Preprocessing Pipeline

Implemented in [src/descra/data/preprocessing.py](file:///C:/DescriptAI/src/descra/data/preprocessing.py):
1. **HTML Stripping & Entity Decoding:** Unescapes entities (`&amp;` $\rightarrow$ `&`, `&quot;` $\rightarrow$ `"`) and strips markup tags (`<p>`, `<b>`, `<div>`, `<br>`).
2. **Technical Specification Preservation:** Regex preserves units and numbers (`40-hour`, `500 ml`, `2.4 GHz`, `100 W`, `256 GB`, `88mm`, `304 Stainless Steel`).
3. **Deduplication:** Filters out both duplicate product identifiers (`parent_asin`) and duplicate boilerplate marketing descriptions across disparate sellers.
4. **Length Filters:** Enforces minimum 10 words, maximum 250 words for descriptions; minimum 1 feature bullet; minimum 5 characters for titles.

---

## 4. Input-Target Factual Alignment Verification

To guarantee that the Seq2Seq Transformer trains on **factually grounded** reference descriptions, every candidate record is audited:
- Combines product name, bullet features, and specifications into structured source facts.
- Audits reference description for unsupported numerical claims, dimensional contradictions, or ungrounded measurements.
- Categorization:
  - `ALIGNED`: High token overlap with complete numerical consistency.
  - `PARTIALLY_ALIGNED`: Substantive factual overlap with minor descriptive elaboration.
  - `NEEDS_REVIEW`: Contains unsupported measurements or specifications; isolated in `data/processed/needs_review.jsonl`.
  - `INSUFFICIENT_INPUT`: Minimal feature overlap; excluded from training.

---

## 5. Dataset Splitting & Zero-Leakage Audit

Implemented in [src/descra/data/splitting.py](file:///C:/DescriptAI/src/descra/data/splitting.py):
- **Split Ratios:** 80% Train, 10% Validation, 10% Test (Random Seed: 42).
- **Group-Based Partitioning:** Stratified strictly by `product_id` and normalized description to ensure zero product identity or boilerplate leakage.
- **Audit Verification:**
  - Product ID Overlap (Train $\cap$ Test): **0**
  - Normalized Description Overlap (Train $\cap$ Test): **0**
  - **Audit Status:** `PASSED (Zero Cross-Split Leakage)`

### Partition Sizes:
| Split | File Path | Record Count | Percentage |
| :--- | :--- | :--- | :--- |
| **Training** | `data/processed/train.jsonl` | **3,828** | 80.0% |
| **Validation** | `data/processed/validation.jsonl` | **478** | 10.0% |
| **Testing** | `data/processed/test.jsonl` | **480** | 10.0% |
| **Total Accepted** | | **4,786** | 100.0% |
| *Needs Review* | `data/processed/needs_review.jsonl` | *4,808* | *Excluded* |

