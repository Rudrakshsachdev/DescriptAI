# CATEGORY AVAILABILITY & DATASET COVERAGE AUDIT
## Descra — Phase 1: Step 7 Baseline Experimentation

**Date:** November 18, 2026
**Dataset Active:** All_Beauty (Amazon Reviews 2023)
**Evaluation Partition:** `data/processed/validation.jsonl`

---

## 1. Executive Summary & Scope Declaration

The Descra Phase 1 Roadmap originally proposed zero-shot evaluation across ten distinct product categories. In this audit, we systematically evaluated local disk availability, remote source requirements, and dataset ingestion eligibility across all ten candidates. As documented below, **only 1 category (`All_Beauty`) is locally acquired and canonicalized**, while the remaining 9 categories are currently absent from local disk.

> [!IMPORTANT]
> **Scope Status: PARTIALLY COMPLETE (Scoped to Available Category)**. We adhere strictly to engineering integrity guidelines: zero data fabrication. The experimental framework is fully category-agnostic and ready to process additional categories as soon as they are acquired.

## 2. Formal Category Availability Matrix

| # | Category Name | Source Dataset | Local Availability | Raw Count | Accepted | Rejected | Preprocessing Status | Eligible? |
|---|---|---|---|---:|---:|---:|---|:---:|
| 1 | **Beauty and Personal Care** | UCSD / McAuley Amazon Reviews 2023 (All_Beauty) | `AVAILABLE_PROCESSED` | 112,590 | 4,786 | 107,804 | CANONICAL_PROCESSED_V1.0 | **YES** |
| 2 | **Electronics** | UCSD / McAuley Amazon Reviews 2023 (Electronics) | `ABSENT_NOT_DOWNLOADED` | 0 | 0 | 0 | NOT_ACQUIRED | NO |
| 3 | **Clothing** | UCSD / McAuley Amazon Reviews 2023 (Clothing_Shoes_and_Jewelry) | `ABSENT_NOT_DOWNLOADED` | 0 | 0 | 0 | NOT_ACQUIRED | NO |
| 4 | **Home and Kitchen** | UCSD / McAuley Amazon Reviews 2023 (Home_and_Kitchen) | `ABSENT_NOT_DOWNLOADED` | 0 | 0 | 0 | NOT_ACQUIRED | NO |
| 5 | **Sports and Outdoors** | UCSD / McAuley Amazon Reviews 2023 (Sports_and_Outdoors) | `ABSENT_NOT_DOWNLOADED` | 0 | 0 | 0 | NOT_ACQUIRED | NO |
| 6 | **Toys and Games** | UCSD / McAuley Amazon Reviews 2023 (Toys_and_Games) | `ABSENT_NOT_DOWNLOADED` | 0 | 0 | 0 | NOT_ACQUIRED | NO |
| 7 | **Health and Household** | UCSD / McAuley Amazon Reviews 2023 (Health_and_Personal_Care) | `ABSENT_NOT_DOWNLOADED` | 0 | 0 | 0 | NOT_ACQUIRED | NO |
| 8 | **Automotive** | UCSD / McAuley Amazon Reviews 2023 (Automotive) | `ABSENT_NOT_DOWNLOADED` | 0 | 0 | 0 | NOT_ACQUIRED | NO |
| 9 | **Office Products** | UCSD / McAuley Amazon Reviews 2023 (Office_Products) | `ABSENT_NOT_DOWNLOADED` | 0 | 0 | 0 | NOT_ACQUIRED | NO |
| 10 | **Pet Supplies** | UCSD / McAuley Amazon Reviews 2023 (Pet_Supplies) | `ABSENT_NOT_DOWNLOADED` | 0 | 0 | 0 | NOT_ACQUIRED | NO |

## 3. Analysis of Missing Categories & Technical Justification

The 9 unavailable categories (Electronics, Clothing, Home & Kitchen, Sports & Outdoors, Toys & Games, Health & Household, Automotive, Office Products, Pet Supplies) were deferred from acquisition due to:
1. **Storage Footprint:** Full parquet archives for major Amazon categories (e.g., Electronics, Clothing) span multiple gigabytes (2 GB to 5 GB each).
2. **Network Sandbox Isolation:** Automated runs enforce zero-network sandbox rules. Downloading massive files requires manual bypass.
3. **Compute Efficiency:** Phase 1 focuses on Transformer architecture, baseline behavior, and fine-tuning pipelines. A canonical subset of 4,786 records provides statistically rigorous evaluation without redundant disk bloat.

## 4. Product Archetype Diversity in Active Dataset

While cataloged under the overarching metadata domain `All_Beauty`, the active dataset exhibits high intra-domain diversity across distinct functional product archetypes. In the evaluated validation split:

| Product Archetype Cluster | Validation Count | Percentage |
|---|---:|---:|
| **Skincare & Serums** | 59 | 12.3% |
| **Massage & Facial Tools** | 13 | 2.7% |
| **Hair Care & Brushes** | 77 | 16.1% |
| **Fragrance & Deodorants** | 41 | 8.6% |
| **Bath, Body & Accessories** | 34 | 7.1% |
| **Cosmetics & Lip Care** | 60 | 12.6% |
| **Other / General Grooming** | 194 | 40.6% |

## 5. Formal Revised-Scope Recommendation

1. **Baseline Phase 1 Acceptance:** Accept `All_Beauty` as the canonical benchmark domain for Phase 1 baseline and fine-tuning experiments.
2. **Future Ingestion Pipeline:** Utilize the developed `src/descra/data/acquisition.py` and `prepare_dataset.py` to ingest single category shards for `Musical_Instruments` and `Cell_Phones` during Phase 2 scalability testing.
