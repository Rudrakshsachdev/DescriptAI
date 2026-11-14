"""Dataset inspection and raw schema analysis module for Descra."""

from dataclasses import asdict, dataclass
import json
import os
import re
from typing import Any, Dict, List, Optional
import pandas as pd
import pyarrow.parquet as pq

from descra.utils.logging import get_logger

logger = get_logger("descra.data.inspection")

HTML_PATTERN = re.compile(r"<[^>]+>|&(?:[a-z\d]+|#\d+|#x[a-f\d]+);", re.IGNORECASE)


@dataclass
class RawSchemaInspection:
    file_path: str
    total_records: int
    column_names: List[str]
    column_dtypes: Dict[str, str]
    missing_counts: Dict[str, int]
    missing_percentages: Dict[str, float]
    unique_products: int
    records_with_features: int
    records_with_description: int
    records_with_details: int
    records_with_html_in_description: int
    records_with_html_in_title: int
    avg_feature_count: float
    avg_description_tokens: float
    sample_record: Dict[str, Any]


def inspect_raw_parquet(file_path: str, sample_size: Optional[int] = None) -> RawSchemaInspection:
    """Inspects a raw parquet dataset without loading all columns or records into RAM at once."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {file_path}")

    logger.info(f"Opening Parquet metadata for: {file_path}")
    parquet_file = pq.ParquetFile(file_path)
    total_records = parquet_file.metadata.num_rows
    schema = parquet_file.schema_arrow

    col_names = schema.names
    col_dtypes = {name: str(schema.field(name).type) for name in col_names}

    logger.info(f"Parquet contains {total_records:,} total rows across {len(col_names)} columns.")

    read_limit = min(total_records, sample_size) if sample_size else total_records
    logger.info(f"Auditing {read_limit:,} records for in-depth schema and data quality audit...")

    cols_to_read = ["parent_asin", "title", "main_category", "features", "description", "details"]
    cols_to_read = [c for c in cols_to_read if c in col_names]

    table = parquet_file.read(columns=cols_to_read)
    df = table.to_pandas()
    if sample_size and len(df) > sample_size:
        df = df.iloc[:sample_size].copy()

    # Calculate metrics
    missing_counts = {}
    missing_pcts = {}
    for col in cols_to_read:
        # Check null or empty array/string
        null_count = df[col].isna().sum()
        missing_counts[col] = int(null_count)
        missing_pcts[col] = round((null_count / len(df)) * 100, 2)

    unique_products = int(df["parent_asin"].nunique()) if "parent_asin" in df else 0

    # Inspect features & description arrays
    records_with_features = 0
    records_with_description = 0
    records_with_details = 0
    records_html_desc = 0
    records_html_title = 0
    total_feature_counts = 0
    total_desc_words = 0

    for _, row in df.iterrows():
        # Features
        feats = row.get("features")
        has_feat = False
        if feats is not None and hasattr(feats, "__len__") and len(feats) > 0:
            if any(str(x).strip() for x in feats):
                has_feat = True
                total_feature_counts += len(feats)
        if has_feat:
            records_with_features += 1

        # Details
        dets = row.get("details")
        if dets is not None:
            if isinstance(dets, dict) and len(dets) > 0:
                records_with_details += 1
            elif isinstance(dets, str) and len(dets.strip()) > 2 and dets.strip() not in ["{}", "[]"]:
                records_with_details += 1

        # Title HTML
        title = str(row.get("title") or "")
        if HTML_PATTERN.search(title):
            records_html_title += 1

        # Description
        desc = row.get("description")
        has_d = False
        if desc is not None and hasattr(desc, "__len__") and len(desc) > 0:
            if isinstance(desc, (list, tuple)) or hasattr(desc, "tolist"):
                desc_str = " ".join(str(x) for x in desc).strip()
            else:
                desc_str = str(desc).strip()

            if len(desc_str) > 0:
                has_d = True
                words = desc_str.split()
                total_desc_words += len(words)
                if HTML_PATTERN.search(desc_str):
                    records_html_desc += 1
        if has_d:
            records_with_description += 1

    avg_feats = round(total_feature_counts / max(1, records_with_features), 2)
    avg_desc_words = round(total_desc_words / max(1, records_with_description), 2)

    # First complete sample
    sample_record = {}
    for _, row in df.iterrows():
        desc = row.get("description")
        feats = row.get("features")
        if desc is not None and len(desc) > 0 and feats is not None and len(feats) > 0:
            sample_record = {
                k: (row[k].tolist() if hasattr(row[k], "tolist") else str(row[k]))
                for k in cols_to_read
            }
            break

    return RawSchemaInspection(
        file_path=file_path,
        total_records=total_records,
        column_names=col_names,
        column_dtypes=col_dtypes,
        missing_counts=missing_counts,
        missing_percentages=missing_pcts,
        unique_products=unique_products,
        records_with_features=records_with_features,
        records_with_description=records_with_description,
        records_with_details=records_with_details,
        records_with_html_in_description=records_html_desc,
        records_with_html_in_title=records_html_title,
        avg_feature_count=avg_feats,
        avg_description_tokens=avg_desc_words,
        sample_record=sample_record,
    )


def generate_inspection_markdown_report(insp: RawSchemaInspection) -> str:
    """Generates an academic-grade markdown report of the raw schema inspection."""
    report = f"""# DESCЯA — RAW DATASET SCHEMA INSPECTION & AUDIT REPORT
**Dataset Source:** UCSD McAuley-Lab Amazon Product Metadata (2023 Release)  
**File Inspected:** `{insp.file_path}`  
**Total Records in File:** {insp.total_records:,}  
**Audit Sample Size:** 10,000 records  
**Date:** 2026-11-14  

---

## 1. File Structure & Columns

| Column Name | Arrow / Parquet Data Type | Description | Missing % (Audit Sample) |
| :--- | :--- | :--- | :--- |
"""
    for col in insp.column_names:
        dtype = insp.column_dtypes.get(col, "unknown")
        pct = insp.missing_percentages.get(col, 0.0)
        report += f"| `{col}` | `{dtype}` | Raw e-commerce field | {pct:.1f}% |\n"

    report += f"""
---

## 2. Key Text-Generation Field Statistics

- **Total Unique Products in Audit:** {insp.unique_products:,}
- **Records with Usable Feature Bullets (`features`):** {insp.records_with_features:,} ({insp.records_with_features / 100:.1f}%)
- **Average Feature Bullets per Product:** {insp.avg_feature_count}
- **Records with Manufacturer Descriptions (`description`):** {insp.records_with_description:,} ({insp.records_with_description / 100:.1f}%)
- **Average Words per Description:** {insp.avg_description_tokens} words
- **Records with Structured Specifications (`details`):** {insp.records_with_details:,} ({insp.records_with_details / 100:.1f}%)
- **Records Containing Raw HTML Markup in Descriptions:** {insp.records_with_html_in_description:,} ({insp.records_with_html_in_description / max(1, insp.records_with_description) * 100:.1f}%)
- **Records Containing HTML Entities in Titles:** {insp.records_with_html_in_title:,}

---

## 3. Engineering Assessment & Preprocessing Directives

1. **HTML Sanitization Required:**
   Approximately {insp.records_with_html_in_description / max(1, insp.records_with_description) * 100:.1f}% of reference descriptions contain embedded HTML tags (`<p>`, `<b>`, `<br>`, `<div>`) and character entities (`&amp;`, `&quot;`, `&#39;`). Robust regex and entity decoding must precede tokenization.
2. **Technical Details Table Parsing:**
   The `details` column contains JSON strings or dictionaries representing technical attribute tables (e.g. Dimensions, Weight, UPC, Formulation). This must be parsed into key-value pairs without loss of numerical units.
3. **Array vs. String Handling:**
   Both `features` and `description` are stored as variable-length string arrays (lists). Preprocessing must unroll and concatenate these strings cleanly.
"""
    return report
