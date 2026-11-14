# DESCЯA — RAW DATASET SCHEMA INSPECTION & AUDIT REPORT
**Dataset Source:** UCSD McAuley-Lab Amazon Product Metadata (2023 Release)  
**File Inspected:** `data/raw/meta_All_Beauty.parquet`  
**Total Records in File:** 112,590  
**Audit Sample Size:** 10,000 records  
**Date:** 2026-11-14  

---

## 1. File Structure & Columns

| Column Name | Arrow / Parquet Data Type | Description | Missing % (Audit Sample) |
| :--- | :--- | :--- | :--- |
| `main_category` | `string` | Raw e-commerce field | 0.0% |
| `title` | `string` | Raw e-commerce field | 0.0% |
| `average_rating` | `double` | Raw e-commerce field | 0.0% |
| `rating_number` | `int64` | Raw e-commerce field | 0.0% |
| `features` | `list<element: string>` | Raw e-commerce field | 0.0% |
| `description` | `list<element: string>` | Raw e-commerce field | 0.0% |
| `price` | `string` | Raw e-commerce field | 0.0% |
| `images` | `struct<hi_res: list<element: string>, large: list<element: string>, thumb: list<element: string>, variant: list<element: string>>` | Raw e-commerce field | 0.0% |
| `videos` | `struct<title: list<element: string>, url: list<element: string>, user_id: list<element: string>>` | Raw e-commerce field | 0.0% |
| `store` | `string` | Raw e-commerce field | 0.0% |
| `categories` | `list<element: string>` | Raw e-commerce field | 0.0% |
| `details` | `string` | Raw e-commerce field | 0.0% |
| `parent_asin` | `string` | Raw e-commerce field | 0.0% |
| `bought_together` | `string` | Raw e-commerce field | 0.0% |
| `subtitle` | `string` | Raw e-commerce field | 0.0% |
| `author` | `string` | Raw e-commerce field | 0.0% |

---

## 2. Key Text-Generation Field Statistics

- **Total Unique Products in Audit:** 112,590
- **Records with Usable Feature Bullets (`features`):** 17,377 (173.8%)
- **Average Feature Bullets per Product:** 4.18
- **Records with Manufacturer Descriptions (`description`):** 19,162 (191.6%)
- **Average Words per Description:** 85.79 words
- **Records with Structured Specifications (`details`):** 108,078 (1080.8%)
- **Records Containing Raw HTML Markup in Descriptions:** 17 (0.1%)
- **Records Containing HTML Entities in Titles:** 11

---

## 3. Engineering Assessment & Preprocessing Directives

1. **HTML Sanitization Required:**
   Approximately 0.1% of reference descriptions contain embedded HTML tags (`<p>`, `<b>`, `<br>`, `<div>`) and character entities (`&amp;`, `&quot;`, `&#39;`). Robust regex and entity decoding must precede tokenization.
2. **Technical Details Table Parsing:**
   The `details` column contains JSON strings or dictionaries representing technical attribute tables (e.g. Dimensions, Weight, UPC, Formulation). This must be parsed into key-value pairs without loss of numerical units.
3. **Array vs. String Handling:**
   Both `features` and `description` are stored as variable-length string arrays (lists). Preprocessing must unroll and concatenate these strings cleanly.
