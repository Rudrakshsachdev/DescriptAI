"""Unit tests for text sanitization, specification parsing, and factual alignment."""

import pytest
from descra.data.preprocessing import (
    clean_text,
    extract_numerical_facts,
    parse_details_specifications,
    verify_factual_alignment,
)


def test_clean_text_html_and_entities():
    raw_html = "<p>Premium <b>Wireless</b> Headphones &amp; Earbuds with 40-hour battery life!&nbsp;</p>"
    cleaned = clean_text(raw_html)
    assert "<p>" not in cleaned
    assert "<b>" not in cleaned
    assert "&amp;" not in cleaned
    assert "40-hour" in cleaned
    assert "Premium Wireless Headphones & Earbuds with 40-hour battery life!" == cleaned


def test_clean_text_preserves_technical_units():
    tech_str = "Supports 2.4 GHz, 5.3 Bluetooth, 500 ml reservoir, 100 W power, and 256 GB storage."
    cleaned = clean_text(tech_str)
    assert "2.4 GHz" in cleaned
    assert "5.3 Bluetooth" in cleaned
    assert "500 ml" in cleaned
    assert "100 W" in cleaned
    assert "256 GB" in cleaned


def test_parse_details_specifications():
    raw_json_str = '{"Package Dimensions": "7.1 x 5.5 x 3 inches; 2.38 Pounds", "UPC": "617390882781"}'
    specs = parse_details_specifications(raw_json_str)
    assert specs["Package Dimensions"] == "7.1 x 5.5 x 3 inches; 2.38 Pounds"
    assert specs["UPC"] == "617390882781"


def test_factual_alignment_aligned():
    p_name = "Anker Life Q20 Active Noise Cancelling Headphones"
    features = [
        "Custom 40 mm dynamic drivers",
        "Hybrid active noise cancellation reduces ambient noise by 90%",
        "40-hour playtime in wireless ANC mode",
    ]
    specs = {"Connectivity": "Bluetooth 5.0", "Battery": "40 hours"}
    desc = "Life Q20 hybrid active noise cancelling headphones feature 40 mm drivers and 40-hour playtime with Bluetooth 5.0."

    status, score = verify_factual_alignment(p_name, features, specs, desc)
    assert status in ["ALIGNED", "PARTIALLY_ALIGNED"]
    assert score > 0.3


def test_factual_alignment_unsupported_numbers():
    p_name = "Basic Wired Earphones"
    features = ["Standard 3.5mm jack", "Lightweight earbuds"]
    specs = {}
    desc = "Enjoy 60-hour playtime with 500W power, 100m range, and 4000mAh battery."

    status, score = verify_factual_alignment(p_name, features, specs, desc)
    # The reference description invents numbers not present in features/specs
    assert status in ["NEEDS_REVIEW", "INSUFFICIENT_INPUT", "CONTRADICTED"]

