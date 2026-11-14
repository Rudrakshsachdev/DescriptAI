"""Unit tests for evaluation metrics and heuristic factual adherence checks."""

import pytest

from descra.evaluation.factual_checks import (
    detect_repeated_ngrams,
    extract_numbers_and_units,
    verify_factual_adherence,
)
from descra.evaluation.metrics import calculate_bleu, calculate_rouge


def test_extract_numbers_and_units():
    text = "Bottle of 100ml lotion with 20% vitamin C and 2.5 inches length, 2-pack."
    facts = extract_numbers_and_units(text)

    assert "100ml" in facts
    assert "20%" in facts
    assert "2.5 inches" in facts or "2.5" in facts
    assert "2-pack" in facts or "2" in facts


def test_detect_repeated_ngrams():
    repetitive_text = "great hydrating cream great hydrating cream for daily use"
    repeated = detect_repeated_ngrams(repetitive_text, n=3)
    assert len(repeated) > 0
    assert "great hydrating cream" in repeated

    clean_text = "A soothing moisturizer designed specifically for sensitive skin."
    assert len(detect_repeated_ngrams(clean_text, n=3)) == 0


def test_verify_factual_adherence_grounded():
    inp = "Product Name: Lip Balm. Features: Organic beeswax. Specifications: Size: 10g."
    gen = "Organic beeswax lip balm with 10g size."

    result = verify_factual_adherence(inp, gen)
    assert result["factual_consistency_score"] > 0.5
    assert not result["is_empty"]
    assert not result["is_repetitive"]
    assert len(result["unsupported_numbers"]) == 0


def test_verify_factual_adherence_unsupported_numbers():
    inp = "Product Name: Facial Cream. Features: Daily use."
    # Output invents 500ml and 48-hour moisture
    gen = "Daily facial cream with 500ml volume providing 48-hour moisture."

    result = verify_factual_adherence(inp, gen)
    assert len(result["unsupported_numbers"]) > 0
    # Penalty should reduce score
    assert result["factual_consistency_score"] < 0.5


def test_verify_factual_adherence_empty_output():
    result = verify_factual_adherence("Input prompt", "")
    assert result["is_empty"] is True
    assert result["factual_consistency_score"] == 0.0


def test_calculate_bleu():
    preds = ["A gentle face wash for oily skin."]
    refs = ["A gentle face wash designed for oily skin."]

    res = calculate_bleu(preds, refs)
    assert "corpus_bleu" in res
    assert res["corpus_bleu"] > 0.0
    assert res["bleu_1"] > 50.0


def test_calculate_rouge():
    preds = ["Gentle moisturizing cream."]
    refs = ["Gentle moisturizing cream for dry skin."]

    res = calculate_rouge(preds, refs)
    assert "rouge1_f1" in res
    assert res["rouge1_f1"] > 0.6
    assert res["rougeL_f1"] > 0.6

