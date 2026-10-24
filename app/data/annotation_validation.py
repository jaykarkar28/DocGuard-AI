
from __future__ import annotations

from collections import Counter
from typing import Any


FIELD_LABELS = {
    "invoice_number": "INVOICE_NUMBER",
    "invoice_date": "INVOICE_DATE",
    "due_date": "DUE_DATE",
    "supplier_name": "SUPPLIER_NAME",
    "customer_name": "CUSTOMER_NAME",
    "currency": "CURRENCY",
    "subtotal": "SUBTOTAL",
    "tax_amount": "TAX_AMOUNT",
    "total_amount": "TOTAL_AMOUNT",
}


def _expected_entity_pairs(invoice: dict[str, Any]) -> Counter:
    """Build expected (label, text) pairs directly from source data."""
    expected = Counter()

    for field, label in FIELD_LABELS.items():
        value = invoice.get(field)
        if value is not None and value != "":
            expected[(label, str(value))] += 1

    line_items = invoice.get("line_items")
    if isinstance(line_items, list):
        line_item_fields = {
            "description": "LINE_ITEM_DESCRIPTION",
            "quantity": "LINE_ITEM_QUANTITY",
            "unit_price": "LINE_ITEM_UNIT_PRICE",
            "line_total": "LINE_ITEM_TOTAL",
        }

        for item in line_items:
            if not isinstance(item, dict):
                continue

            for field, label in line_item_fields.items():
                value = item.get(field)
                if value is not None and value != "":
                    expected[(label, str(value))] += 1

    return expected


def validate_source_alignment(
    record: dict[str, Any],
    annotation: dict[str, Any],
) -> list[str]:
    """
    Compare source invoice values with generated entity annotations.

    Returns an empty list when the source values and annotation pairs match.
    """
    errors: list[str] = []
    document_id = record.get("document_id", "<unknown>")

    for field in ("document_id", "document_type", "split_group_id"):
        if record.get(field) != annotation.get(field):
            errors.append(
                f"{document_id}: {field} mismatch "
                f"(source={record.get(field)!r}, "
                f"annotation={annotation.get(field)!r})"
            )

    invoice = record.get("invoice")
    if not isinstance(invoice, dict):
        return errors + [f"{document_id}: source invoice is not an object"]

    entities = annotation.get("entities")
    if not isinstance(entities, list):
        return errors + [f"{document_id}: annotation entities must be a list"]

    actual = Counter()
    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            errors.append(
                f"{document_id}: entity at index {index} is not an object"
            )
            continue

        label = entity.get("label")
        value = entity.get("text")

        if not isinstance(label, str) or not isinstance(value, str):
            errors.append(
                f"{document_id}: entity at index {index} "
                "must have string label and text"
            )
            continue

        actual[(label, value)] += 1

    expected = _expected_entity_pairs(invoice)

    for pair, count in (expected - actual).items():
        label, value = pair
        errors.append(
            f"{document_id}: missing annotation "
            f"{label}={value!r} (missing {count})"
        )

    for pair, count in (actual - expected).items():
        label, value = pair
        errors.append(
            f"{document_id}: unexpected annotation "
            f"{label}={value!r} (extra {count})"
        )

    return errors
