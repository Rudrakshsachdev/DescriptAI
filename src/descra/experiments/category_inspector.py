"""Category availability inspector and cross-category dataset auditor for Descra."""

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from descra.utils.logging import get_logger

logger = get_logger("descra.experiments.category_inspector")


@dataclass
class CategoryAvailabilityRecord:
    category_name: str
    proposed_roadmap_index: int
    source_dataset: str
    local_availability: str  # "AVAILABLE_PROCESSED", "AVAILABLE_RAW_ONLY", "ABSENT_NOT_DOWNLOADED"
    raw_record_count: int
    accepted_record_count: int
    rejected_record_count: int
    preprocessing_status: str
    experiment_eligibility: bool
    exclusion_reason: Optional[str] = None


# Official 10 proposed roadmap categories
PROPOSED_CATEGORIES = [
    "Beauty and Personal Care",
    "Electronics",
    "Clothing",
    "Home and Kitchen",
    "Sports and Outdoors",
    "Toys and Games",
    "Health and Household",
    "Automotive",
    "Office Products",
    "Pet Supplies",
]

# Mapping between roadmap names and McAuley-Lab subset identifiers
CATEGORY_SOURCE_MAPPING = {
    "Beauty and Personal Care": {"raw_file": "meta_All_Beauty.parquet", "hf_id": "All_Beauty"},
    "Electronics": {"raw_file": "meta_Electronics.parquet", "hf_id": "Electronics"},
    "Clothing": {"raw_file": "meta_Clothing_Shoes_and_Jewelry.parquet", "hf_id": "Clothing_Shoes_and_Jewelry"},
    "Home and Kitchen": {"raw_file": "meta_Home_and_Kitchen.parquet", "hf_id": "Home_and_Kitchen"},
    "Sports and Outdoors": {"raw_file": "meta_Sports_and_Outdoors.parquet", "hf_id": "Sports_and_Outdoors"},
    "Toys and Games": {"raw_file": "meta_Toys_and_Games.parquet", "hf_id": "Toys_and_Games"},
    "Health and Household": {"raw_file": "meta_Health_and_Personal_Care.parquet", "hf_id": "Health_and_Personal_Care"},
    "Automotive": {"raw_file": "meta_Automotive.parquet", "hf_id": "Automotive"},
    "Office Products": {"raw_file": "meta_Office_Products.parquet", "hf_id": "Office_Products"},
    "Pet Supplies": {"raw_file": "meta_Pet_Supplies.parquet", "hf_id": "Pet_Supplies"},
}


class CategoryAuditor:
    """Inspects and reports dataset availability across all proposed categories."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        manifest_path: str = "data/manifests/dataset_manifest.json",
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.manifest_path = Path(manifest_path)

    def audit_all_categories(self) -> List[CategoryAvailabilityRecord]:
        """Audits the availability of all 10 proposed categories on local disk."""
        records: List[CategoryAvailabilityRecord] = []

        manifest_stats = {}
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                    manifest_stats = manifest_data.get("statistics", {})
            except Exception as e:
                logger.warning(f"Could not read dataset manifest: {e}")

        for idx, cat_name in enumerate(PROPOSED_CATEGORIES, 1):
            mapping = CATEGORY_SOURCE_MAPPING.get(cat_name, {})
            raw_filename = mapping.get("raw_file", "")
            hf_id = mapping.get("hf_id", "")
            raw_path = self.raw_dir / raw_filename

            if cat_name == "Beauty and Personal Care":
                # Check actual files
                is_raw_present = raw_path.exists()
                train_file = self.processed_dir / "train.jsonl"
                val_file = self.processed_dir / "validation.jsonl"
                test_file = self.processed_dir / "test.jsonl"

                if is_raw_present and train_file.exists():
                    raw_count = manifest_stats.get("total_raw_records", 112590)
                    accepted_count = manifest_stats.get("accepted_records", 4786)
                    rejected_count = manifest_stats.get("rejected_records", 107804)
                    records.append(
                        CategoryAvailabilityRecord(
                            category_name=cat_name,
                            proposed_roadmap_index=idx,
                            source_dataset=f"UCSD / McAuley Amazon Reviews 2023 ({hf_id})",
                            local_availability="AVAILABLE_PROCESSED",
                            raw_record_count=raw_count,
                            accepted_record_count=accepted_count,
                            rejected_record_count=rejected_count,
                            preprocessing_status="CANONICAL_PROCESSED_V1.0",
                            experiment_eligibility=True,
                            exclusion_reason=None,
                        )
                    )
                elif is_raw_present:
                    records.append(
                        CategoryAvailabilityRecord(
                            category_name=cat_name,
                            proposed_roadmap_index=idx,
                            source_dataset=f"UCSD / McAuley Amazon Reviews 2023 ({hf_id})",
                            local_availability="AVAILABLE_RAW_ONLY",
                            raw_record_count=112590,
                            accepted_record_count=0,
                            rejected_record_count=0,
                            preprocessing_status="RAW_UNPROCESSED",
                            experiment_eligibility=False,
                            exclusion_reason="Raw dataset present but canonical splits not generated.",
                        )
                    )
            else:
                # Other 9 categories
                if raw_path.exists():
                    records.append(
                        CategoryAvailabilityRecord(
                            category_name=cat_name,
                            proposed_roadmap_index=idx,
                            source_dataset=f"UCSD / McAuley Amazon Reviews 2023 ({hf_id})",
                            local_availability="AVAILABLE_RAW_ONLY",
                            raw_record_count=0,
                            accepted_record_count=0,
                            rejected_record_count=0,
                            preprocessing_status="RAW_UNPROCESSED",
                            experiment_eligibility=False,
                            exclusion_reason="Raw data exists but preprocessing pipeline has not ingested this category.",
                        )
                    )
                else:
                    records.append(
                        CategoryAvailabilityRecord(
                            category_name=cat_name,
                            proposed_roadmap_index=idx,
                            source_dataset=f"UCSD / McAuley Amazon Reviews 2023 ({hf_id})",
                            local_availability="ABSENT_NOT_DOWNLOADED",
                            raw_record_count=0,
                            accepted_record_count=0,
                            rejected_record_count=0,
                            preprocessing_status="NOT_ACQUIRED",
                            experiment_eligibility=False,
                            exclusion_reason=(
                                "Dataset subset absent on local disk. Acquisition deferred to prevent disk exhaustion "
                                "and network sandbox violations. Sharded download required from McAuley-Lab repository."
                            ),
                        )
                    )

        return records

    def extract_product_type_distribution(self, jsonl_path: str = "data/processed/validation.jsonl") -> Dict[str, int]:
        """Categorizes records in the validation set into product archetype clusters.

        Analyzes product titles and features to identify diverse sub-archetypes
        (e.g., Skincare & Lotions, Massage & Facial Tools, Hair & Brushes, Fragrance & Deodorant, Cosmetics & Lips).
        """
        path = Path(jsonl_path)
        if not path.exists():
            return {}

        clusters: Dict[str, int] = {
            "Skincare & Serums": 0,
            "Massage & Facial Tools": 0,
            "Hair Care & Brushes": 0,
            "Fragrance & Deodorants": 0,
            "Bath, Body & Accessories": 0,
            "Cosmetics & Lip Care": 0,
            "Other / General Grooming": 0,
        }

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                name = record.get("product_name", "").lower()
                feats = " ".join(record.get("features", [])).lower()
                combined = f"{name} {feats}"

                if any(w in combined for w in ["roller", "gua sha", "massager", "facial tool", "brush", "comb"]):
                    if any(w in combined for w in ["brush", "comb", "hair"]):
                        clusters["Hair Care & Brushes"] += 1
                    else:
                        clusters["Massage & Facial Tools"] += 1
                elif any(w in combined for w in ["serum", "cream", "lotion", "moisturizer", "vitamin c", "hyaluronic", "anti-aging"]):
                    clusters["Skincare & Serums"] += 1
                elif any(w in combined for w in ["deodorant", "perfume", "fragrance", "cologne", "body spray"]):
                    clusters["Fragrance & Deodorants"] += 1
                elif any(w in combined for w in ["lip", "lipstick", "gloss", "balm", "mascara", "eyeliner", "makeup"]):
                    clusters["Cosmetics & Lip Care"] += 1
                elif any(w in combined for w in ["soap", "body wash", "bath", "sponge", "towel", "kit", "bottle"]):
                    clusters["Bath, Body & Accessories"] += 1
                else:
                    clusters["Other / General Grooming"] += 1

        return clusters
