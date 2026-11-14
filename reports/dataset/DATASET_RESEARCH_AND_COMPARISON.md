# DESCЯA — DATASET RESEARCH & COMPARATIVE EVALUATION
**Phase 1: Deep Learning Foundation & Model Development**  
**Date:** 2026-11-14  
**Authors:** Rudraksh Sachdeva (2481025), Shivansee (2481187), K Jahnvi (2481221), Kriti Abrol (2481214)

---

## 1. Executive Summary

A critical requirement of Phase 1 is ensuring that the training dataset is genuinely suitable for **factually grounded product description generation**.
Many naive implementations mistakenly use customer reviews as targets or train on free-form conversational corpora. For DESCЯA:
- **Input:** Structured product attributes (Product Name, Category, Bullet-Point Features, Technical Specifications).
- **Target:** High-quality, coherent, informative, and grammatically sound manufacturer-style reference product description.

We conducted a structured evaluation of four candidate public e-commerce datasets against our engineering criteria.

---

## 2. Candidate Dataset Evaluation Matrix

| Criterion | 1. Amazon Product Metadata (UCSD / McAuley 2018/2023) | 2. PromptCloud E-Commerce Product Catalog (Flipkart / Kaggle) | 3. WDC Web Data Commons (Schema.org Product Microdata) | 4. Hugging Face E-Commerce Catalog (e.g. `McAuley-Lab/Amazon-Reviews-2023` metadata subset) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Source** | [UCSD Julian McAuley / Ni Jianmo](https://nijianmo.github.io/amazon/index.html) | [PromptCloud / Kaggle](https://www.kaggle.com/datasets/promptcloud/flipkart-products) | [University of Mannheim WDC](http://webdatacommons.org/) | [Hugging Face Datasets Hub](https://huggingface.co/datasets) |
| **License** | Open for non-commercial research & education | Open Database License (ODbL) / CC-BY | Open Access / CC0 | Creative Commons / Research Use |
| **Total Scale** | ~15 Million records across 29 categories | ~20,000 records | >100 Million microdata entities | Category slices available (10k - 500k) |
| **Product Names** | High availability (>99%) | High availability (>99%) | Variable (frequent missing names) | High availability (>99%) |
| **Bullet Features (`feature`)** | Rich bullet lists in 65–85% of records | Available in specification strings | Heterogeneous attribute-value pairs | Rich bullet lists in ~75% of records |
| **Technical Specs (`details`/`tech1`)** | Key-value dictionary available | Structured string dictionary | Graph properties (often sparse) | Key-value dictionary available |
| **Reference Descriptions** | Real manufacturer descriptions (not user reviews) | Long-form manufacturer descriptions | Noisy web snippets | Real manufacturer descriptions |
| **Missing Description %** | ~15-25% depending on category | ~3-5% | >40% | ~12-18% |
| **HTML / Noise Level** | Moderate (requires HTML entity and tag stripping) | Low-to-moderate | High (raw HTML microdata parsing) | Moderate |
| **Suitability for Seq2Seq** | **Exceptional (Industry Standard NLP Benchmark)** | **Very Good** | Poor (too unstructured) | **Exceptional** |
| **Acquisition Footprint** | Category-specific JSON.gz (~50MB - 300MB per category) | ~15MB compressed CSV/JSON | Multiple Gigabytes | Category Parquet streaming |

---

## 3. Deep Dive into the Selected Candidate: Amazon Product Metadata (Curated Electronics & Home Subsets)

### Why this is the optimal academic choice:
1. **Separation of Metadata and Reviews:**
   The UCSD / McAuley dataset strictly separates *Product Reviews* (`ratings, reviewerID, reviewText`) from *Product Metadata* (`title, feature, description, price, brand, details, categories`). This ensures zero confusion between customer subjective opinions and factual product marketing descriptions.
2. **Canonical Schema Alignment:**
   The metadata matches our target canonical schema:
   ```json
   {
     "product_id": "B010OYASRG",
     "product_name": "Soundcore Anker Life Q20 Hybrid Active Noise Cancelling Headphones",
     "category": "Electronics",
     "features": [
       "Custom 40 mm dynamic drivers produce Hi-Res Audio",
       "Hybrid active noise cancellation reduces ambient noise by up to 90%",
       "40-hour playtime in wireless ANC mode (60 hours in standard mode)"
     ],
     "specifications": {
       "Connectivity": "Wireless Bluetooth 5.0",
       "Playtime": "40 Hours",
       "Item Weight": "9.3 ounces"
     },
     "reference_description": "Life Q20 hybrid active noise cancelling headphones are engineered with internal and external microphones to cancel out an extended range of low and mid-frequency noises..."
   }
   ```
3. **Data Quality & Controlled Subset Strategy:**
   - Instead of processing 15M records, we acquire a focused category slice (e.g., **Electronics** and/or **Office Products / Home & Kitchen**) targeting **5,000 to 10,000 clean, validated records**.
   - This ensures fast download, zero storage bloat (<200MB uncompressed), swift exploratory data analysis, and deterministic training times on our RTX 4050 GPU (10-20 minutes for fine-tuning).

---

## 4. Licensing, Ethics & Attribution

- **Dataset Source:** UCSD E-Commerce Dataset (Julian McAuley et al., University of California, San Diego).
- **Academic Attribution:**
  > Jianmo Ni, Jiacheng Li, Julian McAuley. *Justifying recommendations using distantly-labeled reviews and fine-grained aspects.* In EMNLP-IJCNLP, 2019.
- **License Terms:** The dataset is released strictly for academic, educational, and research purposes.
- **Usage Compliance:**
  - Non-commercial educational use by university students.
  - No user personally identifiable information (PII) is included in product metadata.
  - Raw source files will be preserved unchanged; all preprocessing is deterministic and logged in manifests.

