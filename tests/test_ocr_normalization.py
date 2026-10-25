
"""Tests for conservative OCR field normalization."""

import pytest

from app.ocr.normalization import (
    normalize_field_value,
    normalize_invoice_number,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("INV-00004.", "INV-00004"),
        ("INV-00014.", "INV-00014"),
        ("INV-00024.", "INV-00024"),
        (" INV-00123. ", "INV-00123"),
        ("INV-00123...", "INV-00123"),
        ("INV-00123", "INV-00123"),
        ("INV.00123", "INV.00123"),
        ("INV-00123-A", "INV-00123-A"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_invoice_number(raw, expected):
    assert normalize_invoice_number(raw) == expected


def test_only_terminal_punctuation_is_removed():
    assert normalize_invoice_number("INV.00123.") == "INV.00123"


def test_normalization_does_not_correct_misread_digits():
    assert normalize_invoice_number("1NV-00004.") == "1NV-00004"


def test_field_normalization_is_label_specific():
    assert normalize_field_value(
        "INVOICE_NUMBER", "INV-00004."
    ) == "INV-00004"

    assert normalize_field_value(
        "DUE_DATE", "3026.04.26"
    ) == "3026.04.26"


def test_field_label_matching_is_case_insensitive():
    assert normalize_field_value(
        "invoice_number", "INV-00004."
    ) == "INV-00004"


@pytest.mark.parametrize("value", [None, 123, [], {}])
def test_invoice_number_rejects_non_string(value):
    with pytest.raises(TypeError):
        normalize_invoice_number(value)


def test_field_normalizer_rejects_invalid_arguments():
    with pytest.raises(TypeError):
        normalize_field_value(None, "INV-00004.")

    with pytest.raises(TypeError):
        normalize_field_value("INVOICE_NUMBER", None)
