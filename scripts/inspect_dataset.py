"""CLI script to run raw schema inspection on downloaded dataset."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from descra.data.inspection import (
    generate_inspection_markdown_report,
    inspect_raw_parquet,
)


def main():
    parser = argparse.ArgumentParser(description="Inspect raw dataset schema.")
    parser.add_argument(
        "--file-path",
        type=str,
        default="data/raw/meta_All_Beauty.parquet",
        help="Path to raw parquet dataset.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of records to sample for in-depth inspection (None = all records).",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/dataset/RAW_SCHEMA_INSPECTION.md",
        help="Path to markdown output report.",
    )
    args = parser.parse_args()

    insp = inspect_raw_parquet(args.file_path, sample_size=args.sample_size)
    report_md = generate_inspection_markdown_report(insp)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_report)), exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("=" * 60)
    print("RAW SCHEMA INSPECTION COMPLETE")
    print("=" * 60)
    print(f"Total Rows in File:     {insp.total_records:,}")
    sample_str = f"{args.sample_size:,}" if args.sample_size else f"{insp.total_records:,} (Complete File)"
    print(f"Sampled for Audit:      {sample_str}")
    print(f"Columns:                {', '.join(insp.column_names[:6])}...")
    print(f"Unique Products:        {insp.unique_products:,}")
    print(f"With Usable Features:   {insp.records_with_features:,}")
    print(f"With Descriptions:      {insp.records_with_description:,}")
    print(f"With Raw HTML Markup:   {insp.records_with_html_in_description:,}")
    print(f"Report Generated At:    {args.output_report}")
    print("=" * 60)


if __name__ == "__main__":
    main()
