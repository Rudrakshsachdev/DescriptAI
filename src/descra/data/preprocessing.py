"""Preprocessing and canonical schema normalization module for Descra.

Performs deterministic text sanitization (HTML tag removal, entity decoding,
whitespace normalization), technical specification preservation, factual alignment checking,
and record quality filtering according to configs/data.yaml.
"""

from dataclasses import asdict, dataclass
import html
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import pyarrow.parquet as pq

from descra.utils.logging import get_logger

logger = get_logger("descra.data.preprocessing")

# Regex for HTML stripping and entity decoding
HTML_TAG_REGEX = re.compile(r"<[^>]+>", re.IGNORECASE)
WHITESPACE_REGEX = re.compile(r"\s+")

# Numerical specifications and units pattern
# Preserves numbers with units like: 40-hour, 500ml, 2.4GHz, 100W, 256GB, 88mm, 5.3, etc.
NUMERICAL_FACT_REGEX = re.compile(
    r"\b\d+(?:\.\d+)?(?:[- ]?(?:hour|hr|hrs|mah|v|w|hz|khz|ghz|gb|mb|tb|mm|cm|m|inch|inches|oz|ounce|ounces|lb|lbs|pound|pounds|ml|l|pack|count|pcs|piece|pieces|v|a|bt|bluetooth))\b",
    re.IGNORECASE,
)


@dataclass
class CanonicalRecord:
    product_id: str
    product_name: str
    category: str
    features: List[str]
    specifications: Dict[str, str]
    reference_description: str
    alignment_status: str
    alignment_score: float
    raw_details: Optional[Dict[str, Any]] = None


@dataclass
class PreprocessingStatistics:
    total_raw_records: int
    accepted_records: int
    rejected_records: int
    rejection_reasons: Dict[str, int]
    alignment_breakdown: Dict[str, int]
    avg_input_features: float
    avg_description_words: float
    canonical_schema_version: str = "v1.0"


def clean_text(text: Optional[str]) -> str:
    """Sanitizes text by unescaping HTML entities, removing tags, and normalizing whitespace

    while strictly preserving numerical values, dimensions, punctuation, and casing.
    """
    if not text or not isinstance(text, str):
        return ""

    # Unescape HTML entities (&amp; -> &, &quot; -> ", etc.)
    text = html.unescape(text)

    # Strip HTML tags
    text = HTML_TAG_REGEX.sub(" ", text)

    # Normalize whitespace
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    return text


def parse_details_specifications(raw_details: Any) -> Dict[str, str]:
    """Parses raw details column into a structured key-value specification dictionary."""
    specs: Dict[str, str] = {}
    if not raw_details:
        return specs

    data = None
    if isinstance(raw_details, dict):
        data = raw_details
    elif isinstance(raw_details, str):
        raw_str = raw_details.strip()
        if raw_str.startswith("{") and raw_str.endswith("}"):
            try:
                data = json.loads(raw_str)
            except Exception:
                # Handle unquoted or malformed JSON
                pass

    if isinstance(data, dict):
        for k, v in data.items():
            clean_k = clean_text(str(k))
            clean_v = clean_text(str(v))
            if clean_k and clean_v and clean_k.lower() not in ["customer reviews", "best sellers rank"]:
                specs[clean_k] = clean_v

    return specs


def extract_numerical_facts(text: str) -> Set[str]:
    """Extracts numerical facts and measured units from text for factual alignment verification."""
    matches = NUMERICAL_FACT_REGEX.findall(text.lower())
    # Also extract bare numbers
    bare_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    facts = set(matches)
    for num in bare_numbers:
        if len(num) > 1 or num in ["1", "2", "3", "4", "5", "6", "8", "10", "12"]:
            facts.add(num)
    return facts


def verify_factual_alignment(
    product_name: str,
    features: List[str],
    specifications: Dict[str, str],
    reference_description: str,
) -> Tuple[str, float]:
    """Assesses factual grounding of reference description against supplied source facts.

    Returns:
        (alignment_status, alignment_score)
        Statuses: ALIGNED, PARTIALLY_ALIGNED, NEEDS_REVIEW, CONTRADICTED, INSUFFICIENT_INPUT
    """
    source_text = f"{product_name} {' '.join(features)} {' '.join(f'{k} {v}' for k, v in specifications.items())}"

    source_tokens = set(clean_text(source_text).lower().split())
    desc_tokens = set(clean_text(reference_description).lower().split())

    if len(desc_tokens) == 0:
        return "INSUFFICIENT_INPUT", 0.0

    # Content token overlap (excluding trivial stop words)
    stopwords = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "is", "it", "of", "by", "from", "your", "this", "that"}
    meaningful_desc_tokens = desc_tokens - stopwords
    if not meaningful_desc_tokens:
        return "INSUFFICIENT_INPUT", 0.0

    overlap = meaningful_desc_tokens.intersection(source_tokens)
    overlap_ratio = len(overlap) / len(meaningful_desc_tokens)

    # Numerical specifications check
    source_numbers = extract_numerical_facts(source_text)
    desc_numbers = extract_numerical_facts(reference_description)

    unsupported_numbers = desc_numbers - source_numbers

    # Score calculation
    # Heavy penalty for unsupported specific measurements or numbers
    num_penalty = len(unsupported_numbers) * 0.15
    alignment_score = max(0.0, min(1.0, overlap_ratio - num_penalty))

    if len(unsupported_numbers) > 4:
        status = "NEEDS_REVIEW"
    elif alignment_score >= 0.35 and len(unsupported_numbers) <= 1:
        status = "ALIGNED"
    elif alignment_score >= 0.20 and len(unsupported_numbers) <= 3:
        status = "PARTIALLY_ALIGNED"
    elif alignment_score < 0.15:
        status = "INSUFFICIENT_INPUT"
    else:
        status = "NEEDS_REVIEW"

    return status, round(alignment_score, 3)


def preprocess_dataset(
    raw_parquet_path: str,
    config: Dict[str, Any],
    max_records: Optional[int] = None,
) -> Tuple[List[CanonicalRecord], List[CanonicalRecord], PreprocessingStatistics]:
    """Processes raw parquet dataset into canonical records, separating accepted records from review records."""
    if not os.path.exists(raw_parquet_path):
        raise FileNotFoundError(f"Raw dataset file not found: {raw_parquet_path}")

    filters = config.get("filters", {})
    min_desc_words = filters.get("min_description_words", 10)
    max_desc_words = filters.get("max_description_words", 250)
    min_features = filters.get("min_features_count", 1)
    min_title_chars = filters.get("min_title_chars", 5)

    target_size = max_records or config.get("sample_target_size", 10000)

    logger.info(f"Reading raw parquet dataset from: {raw_parquet_path}")
    df = pd.read_parquet(raw_parquet_path)
    total_raw = len(df)
    logger.info(f"Loaded {total_raw:,} raw records.")

    accepted_records: List[CanonicalRecord] = []
    review_records: List[CanonicalRecord] = []
    rejection_reasons: Dict[str, int] = {
        "MISSING_TITLE": 0,
        "SHORT_TITLE": 0,
        "NO_FEATURES": 0,
        "NO_DESCRIPTION": 0,
        "DESCRIPTION_TOO_SHORT": 0,
        "DESCRIPTION_TOO_LONG": 0,
        "DUPLICATE_ASIN": 0,
        "DUPLICATE_DESCRIPTION": 0,
        "UNALIGNED_OR_CONTRADICTED": 0,
    }
    alignment_breakdown: Dict[str, int] = {
        "ALIGNED": 0,
        "PARTIALLY_ALIGNED": 0,
        "NEEDS_REVIEW": 0,
        "INSUFFICIENT_INPUT": 0,
        "CONTRADICTED": 0,
    }

    seen_asins: Set[str] = set()
    seen_descriptions: Set[str] = set()

    for idx, row in df.iterrows():
        if len(accepted_records) >= target_size:
            break

        asin = str(row.get("parent_asin") or f"PROD_{idx:06d}").strip()
        if asin in seen_asins:
            rejection_reasons["DUPLICATE_ASIN"] += 1
            continue

        raw_title = row.get("title")
        if not raw_title or pd.isna(raw_title):
            rejection_reasons["MISSING_TITLE"] += 1
            continue

        clean_title = clean_text(str(raw_title))
        if len(clean_title) < min_title_chars:
            rejection_reasons["SHORT_TITLE"] += 1
            continue

        # Features
        raw_feats = row.get("features")
        cleaned_features: List[str] = []
        if raw_feats is not None and hasattr(raw_feats, "__len__"):
            for item in raw_feats:
                cf = clean_text(str(item))
                if len(cf) > 3:
                    cleaned_features.append(cf)

        if len(cleaned_features) < min_features:
            rejection_reasons["NO_FEATURES"] += 1
            continue

        # Description
        raw_desc = row.get("description")
        desc_pieces = []
        if raw_desc is not None and hasattr(raw_desc, "__len__"):
            if isinstance(raw_desc, (list, tuple)) or hasattr(raw_desc, "tolist"):
                desc_pieces = [clean_text(str(p)) for p in raw_desc if clean_text(str(p))]
            else:
                desc_pieces = [clean_text(str(raw_desc))]

        clean_desc = " ".join(desc_pieces).strip()
        if not clean_desc:
            rejection_reasons["NO_DESCRIPTION"] += 1
            continue

        norm_desc_key = clean_desc.lower()
        if norm_desc_key in seen_descriptions:
            rejection_reasons["DUPLICATE_DESCRIPTION"] += 1
            continue

        desc_word_count = len(clean_desc.split())
        if desc_word_count < min_desc_words:
            rejection_reasons["DESCRIPTION_TOO_SHORT"] += 1
            continue
        if desc_word_count > max_desc_words:
            rejection_reasons["DESCRIPTION_TOO_LONG"] += 1
            continue

        # Specifications
        specs = parse_details_specifications(row.get("details"))
        category = clean_text(str(row.get("main_category") or "General"))

        # Factual Alignment Check
        align_status, align_score = verify_factual_alignment(
            product_name=clean_title,
            features=cleaned_features,
            specifications=specs,
            reference_description=clean_desc,
        )
        alignment_breakdown[align_status] += 1

        record = CanonicalRecord(
            product_id=asin,
            product_name=clean_title,
            category=category,
            features=cleaned_features,
            specifications=specs,
            reference_description=clean_desc,
            alignment_status=align_status,
            alignment_score=align_score,
        )

        seen_asins.add(asin)
        seen_descriptions.add(norm_desc_key)

        # Quality filtering decision
        if align_status in ["ALIGNED", "PARTIALLY_ALIGNED"]:
            accepted_records.append(record)
        else:
            rejection_reasons["UNALIGNED_OR_CONTRADICTED"] += 1
            review_records.append(record)

    total_accepted = len(accepted_records)
    total_rejected = sum(rejection_reasons.values())

    avg_feats = (
        round(sum(len(r.features) for r in accepted_records) / max(1, total_accepted), 2)
    )
    avg_desc_w = (
        round(sum(len(r.reference_description.split()) for r in accepted_records) / max(1, total_accepted), 2)
    )

    stats = PreprocessingStatistics(
        total_raw_records=total_raw,
        accepted_records=total_accepted,
        rejected_records=total_rejected,
        rejection_reasons=rejection_reasons,
        alignment_breakdown=alignment_breakdown,
        avg_input_features=avg_feats,
        avg_description_words=avg_desc_w,
    )

    logger.info(f"Preprocessing completed: {total_accepted:,} accepted records, {total_rejected:,} excluded.")
    return accepted_records, review_records, stats


def generate_preprocessing_markdown_report(stats: PreprocessingStatistics) -> str:
    """Generates an academic-grade markdown report for dataset preprocessing."""
    report = f"""# DESCЯA — DATA PREPROCESSING & CANONICAL SCHEMA REPORT
**Phase 1: Deep Learning Foundation & Model Development**  
**Date:** 2026-11-14  
**Canonical Schema Version:** `{stats.canonical_schema_version}`  

---

## 1. Executive Summary

A core objective of Descra is generating **factually grounded product descriptions** from structured product facts (Product Name, Feature Bullets, and Technical Specifications).
Uncleaned e-commerce metadata frequently contains malformed HTML, missing descriptions, empty bullet arrays, and factual hallucinations.

| Metric | Measured Value |
| :--- | :--- |
| **Total Raw Source Records** | {stats.total_raw_records:,} |
| **Accepted Canonical Records** | **{stats.accepted_records:,}** |
| **Filtered / Excluded Records** | {stats.rejected_records:,} |
| **Average Feature Bullets per Product** | {stats.avg_input_features} |
| **Average Reference Description Words** | {stats.avg_description_words} words |

---

## 2. Rejection Reasons Breakdown

| Exclusion Reason | Records Filtered | Percentage of Filtered |
| :--- | :--- | :--- |
"""
    total_rej = max(1, stats.rejected_records)
    for reason, count in stats.rejection_reasons.items():
        pct = (count / total_rej) * 100
        report += f"| `{reason}` | {count:,} | {pct:.1f}% |\n"

    report += f"""
---

## 3. Factual Alignment Analysis

Every record underwent token overlap and numerical specification verification:
- **`ALIGNED`:** {stats.alignment_breakdown.get('ALIGNED', 0):,} records (high semantic and numerical consistency).
- **`PARTIALLY_ALIGNED`:** {stats.alignment_breakdown.get('PARTIALLY_ALIGNED', 0):,} records (core facts match with minor descriptive elaboration).
- **`NEEDS_REVIEW`:** {stats.alignment_breakdown.get('NEEDS_REVIEW', 0):,} records (contains unsupported measurements or claims; routed to review dataset).
- **`INSUFFICIENT_INPUT`:** {stats.alignment_breakdown.get('INSUFFICIENT_INPUT', 0):,} records (insufficient source overlap; excluded from training).

---

## 4. Canonical JSON Schema Example

```json
{{
  "product_id": "B01CUPMQZE",
  "product_name": "Howard LC0008 Leather Conditioner, 8-Ounce",
  "category": "All Beauty",
  "features": [
    "Material: 304 Stainless Steel; Brass tip",
    "Lengths Available: 88mm, 93mm, 98mm",
    "Accepts cartridge needles with vice style tattoo machines"
  ],
  "specifications": {{
    "Package Dimensions": "7.1 x 5.5 x 3 inches; 2.38 Pounds",
    "UPC": "617390882781"
  }},
  "reference_description": "Howard LC0008 Leather Conditioner deeply penetrates leather fibers to restore natural oils...",
  "alignment_status": "ALIGNED",
  "alignment_score": 0.42
}}
```
"""
    return report
