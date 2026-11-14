"""Preliminary heuristic factual verification and hallucination detection for Descra.

Evaluates numerical consistency, input attribute preservation, repetitive loops,
and ungrounded claims in generated product descriptions.
"""

from collections import Counter
import re
from typing import Any, Dict, List, Set

from descra.data.preprocessing import clean_text


def extract_numbers_and_units(text: str) -> Set[str]:
    """Extracts numerical measurements, dimensions, percentages, and counts from text."""
    # Matches patterns like '100ml', '2.5 inches', '40-hour', '10%', '24k'
    unit_pattern = re.compile(
        r"\b\d+(?:\.\d+)?(?:\s*(?:mm|cm|m|in|inch|inches|ft|oz|ounce|ounces|ml|l|liter|liters|g|kg|lbs|pound|pounds|v|volt|volts|w|watt|watts|mah|fps|hz|khz|ghz|gb|mb|tb|k|pack|piece|pieces|count)\b|%)",
        re.IGNORECASE,
    )
    matches = unit_pattern.findall(text)
    # Also capture bare numbers
    bare_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)

    facts = {m.strip().lower() for m in matches}
    for num in bare_numbers:
        # Ignore trivial single digits like 0, 1 if part of grammar, but keep specific quantities
        if len(num) > 1 or num in ["2", "3", "4", "5", "6", "8", "10", "12", "24", "48"]:
            facts.add(num.lower())
    return facts


def detect_repeated_ngrams(text: str, n: int = 3) -> List[str]:
    """Finds repeated n-grams indicating degenerative repetitive loops."""
    tokens = text.lower().split()
    if len(tokens) < n:
        return []
    ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    return [ng for ng, cnt in counts.items() if cnt > 1]


def verify_factual_adherence(
    input_text: str,
    generated_text: str,
    reference_text: str = "",
) -> Dict[str, Any]:
    """Evaluates factual adherence and checks for hallucinated attributes or degeneracies.

    Args:
        input_text: Full input prompt containing title, features, specs.
        generated_text: Model-generated description.
        reference_text: Optional ground-truth human description.

    Returns:
        Structured factual check report.
    """
    clean_gen = clean_text(generated_text)
    clean_inp = clean_text(input_text)

    # 1. Degeneracy Checks
    gen_words = clean_gen.split()
    is_empty = len(gen_words) == 0
    is_too_short = len(gen_words) < 5
    repeated_3grams = detect_repeated_ngrams(clean_gen, n=3)
    is_repetitive = len(repeated_3grams) > 0

    if is_empty:
        return {
            "factual_consistency_score": 0.0,
            "is_empty": True,
            "is_too_short": True,
            "is_repetitive": False,
            "repeated_ngrams": [],
            "source_numbers_count": 0,
            "generated_numbers_count": 0,
            "unsupported_numbers": [],
            "unsupported_number_count": 0,
            "input_token_preservation_ratio": 0.0,
            "notes": ["Generated text is empty."],
        }

    # 2. Numerical Grounding
    input_nums = extract_numbers_and_units(clean_inp)
    gen_nums = extract_numbers_and_units(clean_gen)

    # Identify numbers in generated text that do not appear in the input
    unsupported_nums = sorted(list(gen_nums - input_nums))

    # 3. Content Token Preservation
    stopwords = {
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "is",
        "it", "of", "by", "from", "your", "this", "that", "these", "those", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "but", "if", "as", "into", "through", "during", "before", "after",
        "above", "below", "up", "down", "in", "out", "off", "over", "under", "again",
    }

    # Tokenize words cleanly without trailing punctuation
    input_tokens = set(re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)?\b", clean_inp.lower())) - stopwords
    gen_tokens = set(re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)?\b", clean_gen.lower())) - stopwords

    if gen_tokens:
        grounded_gen_tokens = gen_tokens.intersection(input_tokens)
        preservation_ratio = len(grounded_gen_tokens) / len(gen_tokens)
    else:
        preservation_ratio = 0.0

    # 4. Composite Factual Consistency Heuristic Score
    # Baseline score = preservation ratio of content words
    # Penalty: 0.20 per unsupported number/unit
    # Penalty: 0.30 if repetitive loop detected
    penalty = (len(unsupported_nums) * 0.20) + (0.30 if is_repetitive else 0.0)
    consistency_score = max(0.0, min(1.0, preservation_ratio - penalty))

    notes = []
    if is_repetitive:
        notes.append(f"Repetitive phrasing detected: {repeated_3grams[:2]}")
    if unsupported_nums:
        notes.append(f"Unsupported numbers/units found: {unsupported_nums}")
    if is_too_short:
        notes.append("Output is unusually short (< 5 words).")
    if not notes:
        notes.append("Output adheres cleanly to supplied facts without unsupported numbers.")

    return {
        "factual_consistency_score": round(consistency_score, 3),
        "is_empty": is_empty,
        "is_too_short": is_too_short,
        "is_repetitive": is_repetitive,
        "repeated_ngrams": repeated_3grams,
        "source_numbers_count": len(input_nums),
        "generated_numbers_count": len(gen_nums),
        "unsupported_numbers": unsupported_nums,
        "unsupported_number_count": len(unsupported_nums),
        "input_token_preservation_ratio": round(preservation_ratio, 3),
        "notes": notes,
    }

