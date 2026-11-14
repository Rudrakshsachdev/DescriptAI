# DESCЯA — DATA PREPROCESSING & CANONICAL SCHEMA REPORT
**Phase 1: Deep Learning Foundation & Model Development**  
**Date:** 2026-11-14  
**Canonical Schema Version:** `v1.0`  

---

## 1. Executive Summary

A core objective of Descra is generating **factually grounded product descriptions** from structured product facts (Product Name, Feature Bullets, and Technical Specifications).
Uncleaned e-commerce metadata frequently contains malformed HTML, missing descriptions, empty bullet arrays, and factual hallucinations.

| Metric | Measured Value |
| :--- | :--- |
| **Total Raw Source Records** | 112,590 |
| **Accepted Canonical Records** | **4,786** |
| **Filtered / Excluded Records** | 107,804 |
| **Average Feature Bullets per Product** | 4.11 |
| **Average Reference Description Words** | 62.01 words |

---

## 2. Rejection Reasons Breakdown

| Exclusion Reason | Records Filtered | Percentage of Filtered |
| :--- | :--- | :--- |
| `MISSING_TITLE` | 12 | 0.0% |
| `SHORT_TITLE` | 29 | 0.0% |
| `NO_FEATURES` | 95,215 | 88.3% |
| `NO_DESCRIPTION` | 4,667 | 4.3% |
| `DESCRIPTION_TOO_SHORT` | 1,033 | 1.0% |
| `DESCRIPTION_TOO_LONG` | 1,066 | 1.0% |
| `DUPLICATE_ASIN` | 0 | 0.0% |
| `DUPLICATE_DESCRIPTION` | 974 | 0.9% |
| `UNALIGNED_OR_CONTRADICTED` | 4,808 | 4.5% |

---

## 3. Factual Alignment Analysis

Every record underwent token overlap and numerical specification verification:
- **`ALIGNED`:** 3,082 records (high semantic and numerical consistency).
- **`PARTIALLY_ALIGNED`:** 1,704 records (core facts match with minor descriptive elaboration).
- **`NEEDS_REVIEW`:** 1,634 records (contains unsupported measurements or claims; routed to review dataset).
- **`INSUFFICIENT_INPUT`:** 3,174 records (insufficient source overlap; excluded from training).

---

## 4. Canonical JSON Schema Example

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
