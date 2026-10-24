"""Additional quality checks for rendered invoice annotations.

These checks complement annotation_validation.validate_source_alignment:
- geometry validation for annotation boxes;
- pairwise overlap diagnostics;
- deterministic render-reproducibility comparison against source records.

The reproducibility check detects stale or modified annotations, but it is not
an independent proof that the renderer's own coordinates are semantically
correct. A renderer bug can reproduce consistently and still require visual
or independently implemented validation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


BBox = tuple[float, float, float, float]


def _parse_bbox(value: Any) -> BBox | None:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or not all(isinstance(item, (int, float)) for item in value)
    ):
        return None
    x0, y0, x1, y1 = (float(item) for item in value)
    return x0, y0, x1, y1


def validate_annotation_geometry(
    annotation: dict[str, Any],
) -> list[str]:
    """Return geometry errors for malformed or out-of-page entity boxes."""
    document_id = str(annotation.get("document_id", "<unknown>"))
    errors: list[str] = []
    width = annotation.get("width")
    height = annotation.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        return [f"{document_id}: annotation width/height must be positive integers"]

    entities = annotation.get("entities")
    if not isinstance(entities, list):
        return [f"{document_id}: annotation entities must be a list"]

    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            errors.append(f"{document_id}: entity at index {index} is not an object")
            continue
        bbox = _parse_bbox(entity.get("bbox"))
        label = entity.get("label", "<unknown>")
        if bbox is None:
            errors.append(f"{document_id}: entity {index} ({label}) has malformed bbox")
            continue
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            errors.append(f"{document_id}: entity {index} ({label}) has zero/negative-area bbox")
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            errors.append(
                f"{document_id}: entity {index} ({label}) bbox {list(bbox)} "
                f"falls outside page {width}x{height}"
            )
    return errors


def bbox_iou(first: Any, second: Any) -> float:
    """Compute intersection-over-union for [x0, y0, x1, y1] boxes."""
    a = _parse_bbox(first)
    b = _parse_bbox(second)
    if a is None or b is None:
        raise ValueError("Both boxes must contain four numeric coordinates")
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    inter_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = inter_w * inter_h
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def find_suspicious_overlaps(
    annotation: dict[str, Any],
    *,
    iou_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """Return overlapping entity pairs as review warnings, not hard errors."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    entities = annotation.get("entities")
    if not isinstance(entities, list):
        return []
    document_id = str(annotation.get("document_id", "<unknown>"))
    findings: list[dict[str, Any]] = []
    for left_index, left in enumerate(entities):
        if not isinstance(left, dict):
            continue
        left_bbox = _parse_bbox(left.get("bbox"))
        if left_bbox is None:
            continue
        for right_index in range(left_index + 1, len(entities)):
            right = entities[right_index]
            if not isinstance(right, dict):
                continue
            right_bbox = _parse_bbox(right.get("bbox"))
            if right_bbox is None:
                continue
            iou = bbox_iou(left_bbox, right_bbox)
            if iou >= iou_threshold and iou > 0:
                findings.append({
                    "document_id": document_id,
                    "left_index": left_index,
                    "left_label": left.get("label", "<unknown>"),
                    "left_text": left.get("text", ""),
                    "right_index": right_index,
                    "right_label": right.get("label", "<unknown>"),
                    "right_text": right.get("text", ""),
                    "iou": round(iou, 6),
                    "severity": "warning",
                    "message": "Entity boxes overlap; review whether the overlap is intentional.",
                })
    return findings


def validate_annotation_entity_pairs(
    record: dict[str, Any],
    annotation: dict[str, Any],
) -> list[str]:
    """Compare source-derived entity label/text multiplicities to annotations."""
    document_id = str(record.get("document_id", "<unknown>"))
    invoice = record.get("invoice")
    if not isinstance(invoice, dict):
        return [f"{document_id}: source invoice is not an object"]
    actual_entities = annotation.get("entities")
    if not isinstance(actual_entities, list):
        return [f"{document_id}: annotation entities must be a list"]

    labels = {
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
    expected: Counter[tuple[str, str]] = Counter()
    for field, label in labels.items():
        value = invoice.get(field)
        if value is not None and value != "":
            expected[(label, str(value))] += 1
    line_item_labels = {
        "description": "LINE_ITEM_DESCRIPTION",
        "quantity": "LINE_ITEM_QUANTITY",
        "unit_price": "LINE_ITEM_UNIT_PRICE",
        "line_total": "LINE_ITEM_TOTAL",
    }
    line_items = invoice.get("line_items")
    if isinstance(line_items, list):
        for item in line_items:
            if isinstance(item, dict):
                for field, label in line_item_labels.items():
                    value = item.get(field)
                    if value is not None and value != "":
                        expected[(label, str(value))] += 1

    actual: Counter[tuple[str, str]] = Counter()
    errors: list[str] = []
    for index, entity in enumerate(actual_entities):
        if not isinstance(entity, dict):
            errors.append(f"{document_id}: entity at index {index} is not an object")
            continue
        label, value = entity.get("label"), entity.get("text")
        if not isinstance(label, str) or not isinstance(value, str):
            errors.append(f"{document_id}: entity at index {index} must have string label/text")
            continue
        actual[(label, value)] += 1

    for (label, value), count in (expected - actual).items():
        errors.append(f"{document_id}: missing annotation {label}={value!r} (missing {count})")
    for (label, value), count in (actual - expected).items():
        errors.append(f"{document_id}: unexpected annotation {label}={value!r} (extra {count})")
    return errors


def compare_rendered_annotations(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> list[str]:
    """Compare key renderer outputs; useful for detecting stale/tampered records."""
    document_id = str(expected.get("document_id", "<unknown>"))
    errors: list[str] = []
    for key in ("document_id", "template_id", "width", "height", "entities"):
        if expected.get(key) != actual.get(key):
            errors.append(f"{document_id}: rendered annotation mismatch for {key}")
    return errors
