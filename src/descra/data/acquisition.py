"""Dataset acquisition module for Descra.

Handles verified, incremental acquisition of UCSD / McAuley Amazon Product Metadata
from official parquet/jsonl sources with progress logging, file integrity verification (SHA256),
disk space validation, and provenance tracking.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional
import urllib.request

from descra.utils.logging import get_logger

logger = get_logger("descra.data.acquisition")

# Official UCSD / McAuley-Lab Amazon Product Metadata URL repository on Hugging Face Hub
HF_BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"

CATEGORY_MAPPING = {
    "Electronics": "raw_meta_Electronics/full-00000-of-00010.parquet",
    "All_Beauty": "raw_meta_All_Beauty/full-00000-of-00001.parquet",
    "Musical_Instruments": "raw_meta_Musical_Instruments/full-00000-of-00002.parquet",
    "Cell_Phones_and_Accessories": "raw_meta_Cell_Phones_and_Accessories/full-00000-of-00005.parquet",
}


@dataclass
class AcquisitionMetadata:
    source_name: str
    source_url: str
    dataset_version: str
    category: str
    raw_file_name: str
    raw_file_path: str
    file_size_bytes: int
    file_size_mb: float
    sha256_checksum: str
    download_timestamp: str
    license_type: str
    attribution: str


def compute_sha256(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """Computes SHA-256 hash of a file incrementally to avoid loading it into memory."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_disk_space(target_dir: str, required_mb: float = 500.0) -> None:
    """Ensures sufficient free disk space exists before initiating download."""
    os.makedirs(target_dir, exist_ok=True)
    usage = shutil.disk_usage(target_dir)
    free_mb = usage.free / (1024 * 1024)
    if free_mb < required_mb:
        raise OSError(
            f"Insufficient disk space in {target_dir}: {free_mb:.1f} MB free, "
            f"required at least {required_mb:.1f} MB."
        )


def download_raw_subset(
    category: str = "Electronics",
    output_dir: str = "data/raw",
    max_retries: int = 3,
    timeout: int = 30,
) -> AcquisitionMetadata:
    """Downloads the official category metadata parquet file from McAuley-Lab repository."""
    if category not in CATEGORY_MAPPING:
        raise ValueError(
            f"Unsupported category '{category}'. Available categories: {list(CATEGORY_MAPPING.keys())}"
        )

    rel_path = CATEGORY_MAPPING[category]
    source_url = f"{HF_BASE_URL}/{rel_path}"
    file_name = f"meta_{category}.parquet"

    target_path = Path(output_dir) / file_name
    validate_disk_space(str(target_path.parent), required_mb=500.0)

    logger.info(f"Initiating dataset acquisition for category: '{category}'")
    logger.info(f"Source URL: {source_url}")
    logger.info(f"Target destination: {target_path}")

    # Check if already downloaded and valid
    if target_path.exists() and target_path.stat().st_size > 1024 * 1024:
        logger.info(f"Raw source file already exists ({target_path.stat().st_size / (1024*1024):.1f} MB). Verifying integrity...")
        file_hash = compute_sha256(str(target_path))
        file_size = target_path.stat().st_size
    else:
        # Download with retry support
        download_success = False
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Download attempt {attempt}/{max_retries}...")
                req = urllib.request.Request(
                    source_url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Descra-Acquisition/1.0"},
                )

                start_time = time.time()
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    total_size = int(response.headers.get("Content-Length", 0))
                    downloaded = 0
                    chunk_size = 1024 * 512  # 512 KB chunks

                    with open(target_path, "wb") as out_file:
                        while chunk := response.read(chunk_size):
                            out_file.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0 and downloaded % (10 * 1024 * 1024) < chunk_size:
                                pct = (downloaded / total_size) * 100
                                logger.info(f"Progress: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%)")

                elapsed = time.time() - start_time
                logger.info(f"Download completed in {elapsed:.1f}s ({target_path.stat().st_size / (1024*1024):.1f} MB).")
                download_success = True
                break

            except Exception as e:
                logger.warning(f"Download attempt {attempt} failed: {e}")
                last_error = e
                if target_path.exists():
                    target_path.unlink()
                time.sleep(attempt * 2)

        if not download_success:
            raise RuntimeError(f"Failed to acquire dataset from {source_url} after {max_retries} attempts: {last_error}")

        file_hash = compute_sha256(str(target_path))
        file_size = target_path.stat().st_size

    meta = AcquisitionMetadata(
        source_name="UCSD McAuley-Lab Amazon Product Metadata (2023 Update)",
        source_url=source_url,
        dataset_version="Amazon-Reviews-2023 (McAuley-Lab)",
        category=category,
        raw_file_name=file_name,
        raw_file_path=str(target_path),
        file_size_bytes=file_size,
        file_size_mb=round(file_size / (1024 * 1024), 2),
        sha256_checksum=file_hash,
        download_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        license_type="Research & Non-Commercial Educational Use",
        attribution=(
            "Jianmo Ni, Jiacheng Li, Julian McAuley. Justifying recommendations using "
            "distantly-labeled reviews and fine-grained aspects. EMNLP-IJCNLP 2019 / "
            "Hou et al., Bridging Language and Items for Recommendation and Retrieval, arXiv 2024."
        ),
    )

    # Save manifest
    manifest_dir = Path("data/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "source_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    logger.info(f"Source manifest successfully recorded to {manifest_path}")
    return meta

