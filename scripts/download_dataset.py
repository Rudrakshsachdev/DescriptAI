"""CLI script to acquire official raw dataset subset for Descra."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from descra.data.acquisition import download_raw_subset


def main():
    parser = argparse.ArgumentParser(description="Download raw e-commerce metadata subset.")
    parser.add_argument(
        "--category",
        type=str,
        default="All_Beauty",  # ~59MB, ideal for ultra-fast, high quality 110k record base
        help="Category to download (e.g., All_Beauty, Musical_Instruments, Electronics).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to store raw files.",
    )
    args = parser.parse_args()

    meta = download_raw_subset(category=args.category, output_dir=args.output_dir)
    print("\n" + "=" * 60)
    print("DATASET ACQUISITION SUMMARY")
    print("=" * 60)
    print(f"Dataset Version:  {meta.dataset_version}")
    print(f"Category:         {meta.category}")
    print(f"Raw File:         {meta.raw_file_path}")
    print(f"File Size:        {meta.file_size_mb:.2f} MB ({meta.file_size_bytes:,} bytes)")
    print(f"SHA-256 Hash:     {meta.sha256_checksum}")
    print(f"License:          {meta.license_type}")
    print("=" * 60)


if __name__ == "__main__":
    main()

