
"""Field-level OCR evaluation using independent rendered annotations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any


def normalize_whitespace(value: str) -> str:
    """Normalize whitespace without hiding punctuation or case errors."""
    return " ".join(value.split())


def edit_distance(
    reference: Sequence[Any],
    hypothesis: Sequence[Any],
) -> int:
    """Calculate Levenshtein distance using O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))

    for row_index, ref_item in enumerate(reference, start=1):
        current = [row_index]

        for col_index, hyp_item in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_item == hyp_item else 1
            current.append(
                min(
                    previous[col_index] + 1,
                    current[col_index - 1] + 1,
                    previous[col_index - 1] + substitution_cost,
                )
            )

        previous = current

    return previous[-1]


def error_rate(reference: str, hypothesis: str, *, unit: str) -> float:
    """Return CER or WER; return infinity for insertions against empty text."""
    reference = normalize_whitespace(reference)
    hypothesis = normalize_whitespace(hypothesis)

    if unit == "character":
        reference_units = list(reference)
        hypothesis_units = list(hypothesis)
    elif unit == "word":
        reference_units = reference.split()
        hypothesis_units = hypothesis.split()
    else:
        raise ValueError("unit must be 'character' or 'word'")

    if not reference_units:
        return 0.0 if not hypothesis_units else float("inf")

    return edit_distance(reference_units, hypothesis_units) / len(
        reference_units
    )


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    """Validate and return a [left, top, right, bottom] bounding box."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None

    try:
        left, top, right, bottom = (float(item) for item in value)
    except (TypeError, ValueError):
        return None

    if right <= left or bottom <= top:
        return None

    return left, top, right, bottom


def _word_center(
    box: tuple[float, float, float, float],
) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _valid_order_metadata(
    word: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    """Return Tesseract's reading-order metadata when it is valid."""
    keys = ("block_num", "par_num", "line_num", "word_num")
    values: list[int] = []

    for key in keys:
        value = word.get(key)

        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None

        values.append(value)

    return values[0], values[1], values[2], values[3]


def _intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Calculate the area shared by two bounding boxes."""
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _area(box: tuple[float, float, float, float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def extract_entity_text(
    entity: dict[str, Any],
    words: list[dict[str, Any]],
    *,
    padding: float = 8.0,
) -> tuple[str, int]:
    """Extract OCR words assigned to an annotated entity.

    First select words whose centers fall inside the padded entity box.
    If no words meet that rule, use a conservative overlap fallback: an OCR
    box must cover at least 50% of the entity box and contain the entity
    center. This handles oversized OCR boxes without accepting every word
    whose box merely touches the annotation.

    When all selected tokens have valid Tesseract ordering metadata, preserve
    that reading order. Otherwise, use geometric top-to-bottom, left-to-right
    ordering.

    This remains a geometric baseline; overlap matching cannot guarantee
    semantic assignment when OCR boxes span multiple fields.
    """
    entity_box = _bbox(entity.get("bbox"))
    if entity_box is None:
        raise ValueError("Entity has an invalid bounding box")

    if padding < 0:
        raise ValueError("padding must be non-negative")

    left, top, right, bottom = entity_box
    expanded = (
        left - padding,
        top - padding,
        right + padding,
        bottom + padding,
    )
    entity_center_x, entity_center_y = _word_center(entity_box)
    entity_area = _area(entity_box)

    valid_words: list[
        tuple[int, dict[str, Any], tuple[float, float, float, float]]
    ] = []

    for index, word in enumerate(words):
        if not isinstance(word, dict):
            continue

        text = word.get("text")
        word_box = _bbox(word.get("bbox"))

        if (
            not isinstance(text, str)
            or not text.strip()
            or word_box is None
        ):
            continue

        valid_words.append((index, word, word_box))

    # Primary rule: use the OCR word center within the padded entity box.
    selected = []

    for index, word, word_box in valid_words:
        center_x, center_y = _word_center(word_box)

        if (
            expanded[0] <= center_x <= expanded[2]
            and expanded[1] <= center_y <= expanded[3]
        ):
            selected.append((index, word, word_box))

    # Fallback only when the primary rule found no words for this entity.
    if not selected:
        fallback = []

        for index, word, word_box in valid_words:
            overlap = _intersection_area(entity_box, word_box)

            if (
                overlap / entity_area >= 0.50
                and word_box[0] <= entity_center_x <= word_box[2]
                and word_box[1] <= entity_center_y <= word_box[3]
            ):
                fallback.append((index, word, word_box))

        # Prefer the candidate that covers the greatest fraction of the
        # entity. Preserve original order to make ties deterministic.
        if fallback:
            fallback.sort(
                key=lambda item: (
                    -_intersection_area(entity_box, item[2]) / entity_area,
                    item[0],
                )
            )
            selected = [fallback[0]]

    if selected and all(
        _valid_order_metadata(word) is not None
        for _, word, _ in selected
    ):
        selected.sort(
            key=lambda item: (
                *_valid_order_metadata(item[1]),  # type: ignore[arg-type]
                item[0],
            )
        )
    else:
        selected.sort(
            key=lambda item: (
                _word_center(item[2])[1],
                item[2][0],
                item[0],
            )
        )

    extracted = " ".join(word["text"].strip() for _, word, _ in selected)
    return normalize_whitespace(extracted), len(selected)


def evaluate_document(
    annotation: dict[str, Any],
    ocr_result: dict[str, Any],
    *,
    padding: float = 8.0,
) -> list[dict[str, Any]]:
    """Evaluate every present annotated entity value against OCR output."""
    document_id = annotation.get("document_id")

    if document_id != ocr_result.get("document_id"):
        raise ValueError(
            "Document ID mismatch: "
            f"{document_id!r} != {ocr_result.get('document_id')!r}"
        )

    entities = annotation.get("entities", [])
    words = ocr_result.get("words", [])

    if not isinstance(entities, list) or not isinstance(words, list):
        raise ValueError("Annotation entities and OCR words must be lists")

    rows: list[dict[str, Any]] = []

    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            continue

        reference = entity.get("text")
        if not isinstance(reference, str) or not reference.strip():
            continue

        prediction, matched_word_count = extract_entity_text(
            entity,
            words,
            padding=padding,
        )

        normalized_reference = normalize_whitespace(reference)
        normalized_prediction = normalize_whitespace(prediction)

        rows.append(
            {
                "document_id": document_id,
                "template_id": annotation.get("template_id"),
                "entity_index": index,
                "label": entity.get("label", "UNKNOWN"),
                "reference": reference,
                "prediction": prediction,
                "matched_word_count": matched_word_count,
                "detected": matched_word_count > 0,
                "exact_match": normalized_reference == normalized_prediction,
                "cer": error_rate(
                    normalized_reference,
                    normalized_prediction,
                    unit="character",
                ),
                "wer": error_rate(
                    normalized_reference,
                    normalized_prediction,
                    unit="word",
                ),
            }
        )

    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate field metrics and micro-average edit rates."""
    total = len(rows)
    detected = sum(bool(row["detected"]) for row in rows)
    exact = sum(bool(row["exact_match"]) for row in rows)

    reference_characters = 0
    character_errors = 0
    reference_words = 0
    word_errors = 0

    labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    templates: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        reference = normalize_whitespace(row["reference"])
        prediction = normalize_whitespace(row["prediction"])

        ref_chars = list(reference)
        pred_chars = list(prediction)

        if ref_chars:
            reference_characters += len(ref_chars)
            character_errors += edit_distance(ref_chars, pred_chars)

        ref_words = reference.split()
        pred_words = prediction.split()

        if ref_words:
            reference_words += len(ref_words)
            word_errors += edit_distance(ref_words, pred_words)

        labels[str(row["label"])].append(row)
        templates[str(row["template_id"])].append(row)

    def group_summary(
        groups: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for name, group in sorted(groups.items()):
            count = len(group)
            result[name] = {
                "field_count": count,
                "detected_fields": sum(
                    bool(row["detected"]) for row in group
                ),
                "exact_match_accuracy": (
                    sum(bool(row["exact_match"]) for row in group) / count
                    if count
                    else None
                ),
            }

        return result

    return {
        "field_count": total,
        "detected_fields": detected,
        "field_detection_coverage": detected / total if total else None,
        "exact_match_count": exact,
        "field_exact_match_accuracy": exact / total if total else None,
        "cer": (
            character_errors / reference_characters
            if reference_characters
            else None
        ),
        "wer": (
            word_errors / reference_words if reference_words else None
        ),
        "by_label": group_summary(labels),
        "by_template": group_summary(templates),
        "metric_notes": {
            "field_exact_match_accuracy": (
                "Case-sensitive comparison after whitespace normalization."
            ),
            "field_detection_coverage": (
                "Fraction of present annotated fields with at least one "
                "OCR token assigned by center matching or the overlap fallback."
            ),
            "cer": (
                "Micro-averaged character edit distance divided by "
                "reference characters."
            ),
            "wer": (
                "Micro-averaged word edit distance divided by reference words."
            ),
            "scope": (
                "Present annotated entity values only. Static printed labels "
                "and absent fields are excluded."
            ),
        },
    }
