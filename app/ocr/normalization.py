
"""Conservative field-specific normalization for DocGuard AI OCR."""

from __future__ import annotations

import re


_TERMINAL_INVOICE_PUNCTUATION = re.compile(r"[.,;:]+$")


def normalize_invoice_number(value: str) -> str:
    """Remove OCR punctuation artifacts from the end of an invoice number.

    This deliberately does not change letters, digits, internal punctuation,
    or whitespace within the identifier.
    """
    if not isinstance(value, str):
        raise TypeError("Invoice number must be a string")

    normalized = value.strip()
    normalized = _TERMINAL_INVOICE_PUNCTUATION.sub("", normalized)
    return normalized.strip()


def normalize_field_value(label: str, value: str) -> str:
    """Apply normalization only when the entity label is INVOICE_NUMBER."""
    if not isinstance(label, str):
        raise TypeError("Field label must be a string")
    if not isinstance(value, str):
        raise TypeError("Field value must be a string")

    if label.strip().upper() == "INVOICE_NUMBER":
        return normalize_invoice_number(value)

    return value
