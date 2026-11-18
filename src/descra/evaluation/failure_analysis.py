"""Failure mode analysis and categorization for Descra product description generations."""

from collections import Counter
import re
from typing import Any, Dict, List, Optional

from descra.evaluation.factual_checks import detect_repeated_ngrams, extract_numbers_and_units
from descra.utils.logging import get_logger

logger = get_logger("descra.evaluation.failure_analysis")


def classify_generation_failure(
    product_id: str,
    product_name: str,
    prompt: str,
    generated_text: str,
    reference_text: str = "",
    max_new_tokens: int = 128,
) -> List[Dict[str, Any]]:
    """Analyzes a single generated description and returns all detected failure flags with evidence."""
    failures: List[Dict[str, Any]] = []
    clean_gen = generated_text.strip()

    # 1. EMPTY OUTPUT
    if not clean_gen:
        failures.append({
            "category": "EMPTY_OUTPUT",
            "severity": "CRITICAL",
            "description": "Model generated an empty string or whitespace-only output.",
            "evidence": repr(generated_text),
        })
        return failures

    words = clean_gen.split()
    word_count = len(words)

    # 2. BRAND ONLY / TRUNCATED OUTPUT
    # If output is 1-3 words and matches product title or brand
    if word_count <= 3:
        failures.append({
            "category": "BRAND_OR_EXTREMELY_SHORT",
            "severity": "HIGH",
            "description": f"Output is extremely short ({word_count} words), likely single brand/token collapse.",
            "evidence": clean_gen,
        })

    # 3. PROMPT ECHOING
    # If the generated text mirrors the instruction prompt header
    if "Generate a clear and engaging product description" in clean_gen or "Features:" in clean_gen or "Specifications:" in clean_gen:
        failures.append({
            "category": "PROMPT_ECHO",
            "severity": "HIGH",
            "description": "Model mirrored instruction prompt syntax or structural labels.",
            "evidence": clean_gen[:100],
        })

    # 4. REPETITIVE LOOPS
    repeated_3grams = detect_repeated_ngrams(clean_gen, n=3)
    if len(repeated_3grams) >= 2:
        failures.append({
            "category": "REPETITIVE_OUTPUT",
            "severity": "HIGH",
            "description": f"Degenerative repetitive loops detected ({len(repeated_3grams)} repeated tri-grams).",
            "evidence": repeated_3grams[:3],
        })

    # 5. UNSUPPORTED NUMERICAL CLAIMS
    prompt_numbers = extract_numbers_and_units(prompt)
    output_numbers = extract_numbers_and_units(clean_gen)
    unsupported_numbers = output_numbers - prompt_numbers
    if unsupported_numbers:
        failures.append({
            "category": "UNSUPPORTED_NUMBERS",
            "severity": "MEDIUM",
            "description": f"Numerical entities generated that are absent from input prompt ({len(unsupported_numbers)} ungrounded entities).",
            "evidence": list(unsupported_numbers),
        })

    # 6. GENERIC CONVERSATIONAL FILLER
    filler_patterns = [
        r"\b(simple,? easy to use)\b",
        r"\b(great way to have a place)\b",
        r"\b(is a unique product)\b",
        r"\b(is designed to help you)\b",
        r"\b(suitable for all)\b",
    ]
    matched_fillers = [p for p in filler_patterns if re.search(p, clean_gen, re.IGNORECASE)]
    if matched_fillers and word_count < 25:
        failures.append({
            "category": "GENERIC_FILLER",
            "severity": "LOW",
            "description": "Output consists largely of generic conversational filler with little concrete specification.",
            "evidence": matched_fillers,
        })

    return failures


def aggregate_failure_statistics(
    all_failure_records: List[Dict[str, Any]],
    total_samples: int,
) -> Dict[str, Any]:
    """Computes summary frequencies and rates of detected failure categories."""
    category_counts = Counter()
    severity_counts = Counter()

    for item in all_failure_records:
        for f in item.get("failures", []):
            category_counts[f["category"]] += 1
            severity_counts[f["severity"]] += 1

    summary = {
        "total_evaluations": total_samples,
        "total_failures_detected": len(all_failure_records),
        "failure_rate_pct": round(100.0 * len(all_failure_records) / total_samples, 2) if total_samples > 0 else 0.0,
        "category_counts": dict(category_counts),
        "category_rates_pct": {k: round(100.0 * v / total_samples, 2) for k, v in category_counts.items()} if total_samples > 0 else {},
        "severity_counts": dict(severity_counts),
    }

    return summary
