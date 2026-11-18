"""Systematic generation failure analysis and hallucination detection for Descra.

Categorizes failure modes across:
- Empty output
- Degenerative repetition loops
- Generic conversational boilerplate
- Prompt echoing / instruction leakage
- Missing critical input features
- Unsupported specifications
- Hallucinated or inconsistent numerical claims
"""

from collections import Counter
from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from descra.data.preprocessing import clean_text
from descra.evaluation.factual_checks import (
    detect_repeated_ngrams,
    extract_numbers_and_units,
)


@dataclass
class FailureRecord:
    example_id: str
    product_id: str
    category: str
    strategy: str
    generated_output: str
    failure_type: str  # e.g., EMPTY_OUTPUT, REPETITIVE_NGRAMS, PROMPT_ECHO, etc.
    severity: str      # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    evidence: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentFailureAnalyzer:
    """Detects, categorizes, and serializes generation failures across decoded outputs."""

    # Keywords indicative of prompt echoing
    PROMPT_ECHO_INDICATORS = [
        "generate a clear and engaging",
        "product name:",
        "category:",
        "features:",
        "specifications:",
        "write a concise product description",
        "do not invent specifications",
    ]

    # Common generic filler phrases
    GENERIC_FILLER_PHRASES = [
        "the kit is a simple, easy to use",
        "it is a great product for daily use",
        "high quality product that you will love",
        "perfect for all your needs",
        "made with the best quality materials",
    ]

    def analyze_single_output(
        self,
        example_id: str,
        product_id: str,
        category: str,
        strategy: str,
        input_prompt: str,
        generated_output: str,
        product_name: str = "",
        input_features: Optional[List[str]] = None,
        input_specifications: Optional[Dict[str, str]] = None,
    ) -> List[FailureRecord]:
        """Performs exhaustive failure mode classification for a single generated text."""
        failures: List[FailureRecord] = []
        raw_output = generated_output.strip()
        clean_out = clean_text(raw_output)
        clean_prompt = clean_text(input_prompt).lower()

        # 1. Empty Output Check
        words = clean_out.split()
        if not raw_output or len(words) == 0:
            failures.append(
                FailureRecord(
                    example_id=example_id,
                    product_id=product_id,
                    category=category,
                    strategy=strategy,
                    generated_output=raw_output,
                    failure_type="EMPTY_OUTPUT",
                    severity="CRITICAL",
                    evidence=f"Raw text length: {len(raw_output)}, word count: {len(words)}",
                    explanation="Model emitted immediate EOS token or whitespace without producing any descriptive text.",
                )
            )
            return failures

        # 2. Output Collapse (Ultra-short / single token)
        if len(words) <= 2:
            failures.append(
                FailureRecord(
                    example_id=example_id,
                    product_id=product_id,
                    category=category,
                    strategy=strategy,
                    generated_output=raw_output,
                    failure_type="OUTPUT_COLLAPSE",
                    severity="HIGH",
                    evidence=f"Word count: {len(words)}, Output: '{raw_output}'",
                    explanation="Output collapsed into 1-2 words (often isolated brand name or entity name) instead of a descriptive sentence.",
                )
            )

        # 3. Prompt Echoing
        out_lower = clean_out.lower()
        echo_matches = [phrase for phrase in self.PROMPT_ECHO_INDICATORS if phrase in out_lower]
        if echo_matches:
            failures.append(
                FailureRecord(
                    example_id=example_id,
                    product_id=product_id,
                    category=category,
                    strategy=strategy,
                    generated_output=raw_output,
                    failure_type="PROMPT_ECHO",
                    severity="HIGH",
                    evidence=f"Matched prompt tokens: {echo_matches}",
                    explanation="Model verbatim reproduced instruction labels or prompt framing headers rather than synthesizing product details.",
                )
            )

        # 4. Degenerative Repetition Loops
        repeated_3grams = detect_repeated_ngrams(clean_out, n=3)
        if len(repeated_3grams) >= 1:
            failures.append(
                FailureRecord(
                    example_id=example_id,
                    product_id=product_id,
                    category=category,
                    strategy=strategy,
                    generated_output=raw_output,
                    failure_type="REPETITIVE_NGRAMS",
                    severity="MEDIUM" if len(repeated_3grams) == 1 else "HIGH",
                    evidence=f"Repeated 3-grams: {repeated_3grams[:3]}",
                    explanation=f"Model fell into degenerate looping cycles, repeating n-gram sequences {len(repeated_3grams)} times.",
                )
            )

        # 5. Unsupported Numerical Claims (Hallucinations)
        input_nums = extract_numbers_and_units(clean_prompt)
        gen_nums = extract_numbers_and_units(clean_out)
        unsupported_nums = sorted(list(gen_nums - input_nums))
        if unsupported_nums:
            failures.append(
                FailureRecord(
                    example_id=example_id,
                    product_id=product_id,
                    category=category,
                    strategy=strategy,
                    generated_output=raw_output,
                    failure_type="UNSUPPORTED_NUMBERS",
                    severity="HIGH",
                    evidence=f"Unsupported numbers/units: {unsupported_nums} (Input numbers: {sorted(list(input_nums))})",
                    explanation="Model introduced numerical quantities, metrics, or pack sizes that are absent from the provided metadata.",
                )
            )

        # 6. Missing Critical Features
        if input_features and len(words) >= 5:
            # Extract salient content tokens from features (excluding short stopwords)
            stopwords = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "with", "is", "of", "by", "this", "that"}
            feat_tokens: Set[str] = set()
            for f in input_features:
                tokens = re.findall(r"\b[a-z]{4,}\b", f.lower())
                feat_tokens.update(t for t in tokens if t not in stopwords)

            out_tokens = set(re.findall(r"\b[a-z]{4,}\b", out_lower))
            overlap = feat_tokens.intersection(out_tokens)
            # If input had rich features but output preserved virtually none of them
            if len(feat_tokens) >= 5 and len(overlap) == 0:
                failures.append(
                    FailureRecord(
                        example_id=example_id,
                        product_id=product_id,
                        category=category,
                        strategy=strategy,
                        generated_output=raw_output,
                        failure_type="MISSING_FEATURES",
                        severity="MEDIUM",
                        evidence=f"Feature terms count: {len(feat_tokens)}, Matched terms: 0",
                        explanation="Generated text failed to preserve or mention any key thematic keywords from the supplied feature bullets.",
                    )
                )

        # 7. Generic Filler
        filler_matches = [f for f in self.GENERIC_FILLER_PHRASES if f in out_lower]
        if filler_matches:
            failures.append(
                FailureRecord(
                    example_id=example_id,
                    product_id=product_id,
                    category=category,
                    strategy=strategy,
                    generated_output=raw_output,
                    failure_type="GENERIC_FILLER",
                    severity="LOW",
                    evidence=f"Generic phrases matched: {filler_matches}",
                    explanation="Model relied on non-specific generic conversational padding with low information density.",
                )
            )

        return failures

    def aggregate_failures(self, all_failures: List[FailureRecord], total_samples_per_strategy: Dict[str, int]) -> Dict[str, Any]:
        """Summarizes failure counts, frequencies, and percentages across decoding strategies."""
        summary_by_type: Dict[str, int] = Counter()
        summary_by_severity: Dict[str, int] = Counter()
        strategy_breakdown: Dict[str, Dict[str, int]] = {}

        for strat in total_samples_per_strategy.keys():
            strategy_breakdown[strat] = Counter()

        for f in all_failures:
            summary_by_type[f.failure_type] += 1
            summary_by_severity[f.severity] += 1
            if f.strategy in strategy_breakdown:
                strategy_breakdown[f.strategy][f.failure_type] += 1

        strategy_rates = {}
        for strat, total in total_samples_per_strategy.items():
            type_counts = dict(strategy_breakdown.get(strat, {}))
            total_fail_instances = sum(type_counts.values())
            strategy_rates[strat] = {
                "total_evaluated_samples": total,
                "total_failure_instances": total_fail_instances,
                "failure_counts": type_counts,
                "failure_percentages": {
                    k: round((v / total * 100), 2) for k, v in type_counts.items()
                } if total > 0 else {},
            }

        return {
            "total_failure_instances": len(all_failures),
            "breakdown_by_type": dict(summary_by_type),
            "breakdown_by_severity": dict(summary_by_severity),
            "strategy_breakdown": strategy_rates,
            "detailed_failures": [f.to_dict() for f in all_failures],
        }
