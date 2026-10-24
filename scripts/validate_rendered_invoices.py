
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.data.annotation_validation import validate_source_alignment


SPLITS_DIR = ROOT / "datasets/splits/synthetic_invoices"
ANNOTATIONS_FILE = (
    ROOT / "datasets/processed/rendered_invoice_annotations.jsonl"
)
SOURCE_FILE = ROOT / "datasets/processed/synthetic_invoices.jsonl"
REPORT_FILE = ROOT / "datasets/reports/render_validation_report.json"

EXPECTED_SIZE = (1275, 1650)
EXPECTED_TEMPLATES = {"classic", "minimal", "sidebar"}


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL records, supporting an optional UTF-8 BOM."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")

    records = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}, line {line_number}"
                )

            records.append(record)

    return records


def validate() -> dict:
    """Validate rendered images, annotations, source alignment and splits."""
    errors: list[str] = []

    # ---------------------------------------------------------
    # 1. Load source records and rendered annotations
    # ---------------------------------------------------------
    source_records = read_jsonl(SOURCE_FILE)
    annotations = read_jsonl(ANNOTATIONS_FILE)

    source_ids = [
        record.get("document_id")
        for record in source_records
    ]
    annotation_ids = [
        record.get("document_id")
        for record in annotations
    ]

    source_id_counts = Counter(source_ids)
    annotation_id_counts = Counter(annotation_ids)

    duplicate_source_ids = sorted(
        str(doc_id)
        for doc_id, count in source_id_counts.items()
        if doc_id is None or count > 1
    )
    duplicate_annotation_ids = sorted(
        str(doc_id)
        for doc_id, count in annotation_id_counts.items()
        if doc_id is None or count > 1
    )

    if duplicate_source_ids:
        errors.append(
            f"Missing or duplicate source IDs: {duplicate_source_ids[:10]}"
        )

    if duplicate_annotation_ids:
        errors.append(
            "Missing or duplicate annotation IDs: "
            f"{duplicate_annotation_ids[:10]}"
        )

    source_by_id = {
        record["document_id"]: record
        for record in source_records
        if isinstance(record.get("document_id"), str)
        and record["document_id"]
    }

    annotation_by_id = {
        record["document_id"]: record
        for record in annotations
        if isinstance(record.get("document_id"), str)
        and record["document_id"]
    }

    # ---------------------------------------------------------
    # 2. Load train, validation and test split membership
    # ---------------------------------------------------------
    split_by_id: dict[str, str] = {}
    split_groups: dict[str, set[str]] = defaultdict(set)
    split_counts: dict[str, int] = {}
    split_errors: list[str] = []

    for split_name in ("train", "validation", "test"):
        split_path = SPLITS_DIR / f"{split_name}.jsonl"

        if not split_path.is_file():
            errors.append(f"Missing split file: {split_path}")
            continue

        records = read_jsonl(split_path)
        split_counts[split_name] = len(records)

        for record in records:
            doc_id = record.get("document_id")
            group_id = record.get("split_group_id")

            if not doc_id or not group_id:
                split_errors.append(
                    f"Missing document/group ID in {split_name}"
                )
                continue

            if doc_id in split_by_id:
                split_errors.append(
                    f"Document appears in multiple split records: {doc_id}"
                )
                continue

            split_by_id[doc_id] = split_name
            split_groups[group_id].add(split_name)

    if split_errors:
        errors.extend(split_errors)

    leakage = {
        group: sorted(split_names)
        for group, split_names in split_groups.items()
        if len(split_names) > 1
    }

    if leakage:
        errors.append(f"Split-group leakage: {leakage}")

    # ---------------------------------------------------------
    # 3. Check annotation coverage against split records
    # ---------------------------------------------------------
    missing_annotations = sorted(
        set(split_by_id) - set(annotation_by_id)
    )
    unexpected_annotations = sorted(
        set(annotation_by_id) - set(split_by_id)
    )

    if missing_annotations:
        errors.append(
            f"Missing annotations for {len(missing_annotations)} split records"
        )

    if unexpected_annotations:
        errors.append(
            f"{len(unexpected_annotations)} annotations have no split record"
        )

    # ---------------------------------------------------------
    # 4. Compare every annotation against its original source
    # ---------------------------------------------------------
    source_alignment_errors: list[str] = []

    for doc_id, annotation in annotation_by_id.items():
        source_record = source_by_id.get(doc_id)

        if source_record is None:
            source_alignment_errors.append(
                f"{doc_id}: annotation has no source record"
            )
            continue

        source_alignment_errors.extend(
            validate_source_alignment(source_record, annotation)
        )

    missing_source_annotations = sorted(
        set(source_by_id) - set(annotation_by_id)
    )

    if missing_source_annotations:
        source_alignment_errors.append(
            "Source records without annotations: "
            f"{len(missing_source_annotations)}"
        )

    if source_alignment_errors:
        errors.extend(
            f"Source alignment: {message}"
            for message in source_alignment_errors
        )

    # ---------------------------------------------------------
    # 5. Validate image files, dimensions, hashes and boxes
    # ---------------------------------------------------------
    template_counts: Counter = Counter()
    group_templates: dict[str, set[str]] = defaultdict(set)

    missing_images = 0
    bad_dimensions = 0
    bad_hashes = 0
    bad_boxes = 0
    unreadable_images = 0

    for doc_id, annotation in annotation_by_id.items():
        group_id = annotation.get("split_group_id")
        template_id = annotation.get("template_id")

        if group_id and template_id:
            group_templates[group_id].add(template_id)

        if template_id:
            template_counts[template_id] += 1

        raw_image_path = annotation.get("image_path")

        if not isinstance(raw_image_path, str) or not raw_image_path:
            missing_images += 1
            continue

        image_path = Path(raw_image_path)

        if not image_path.is_absolute():
            image_path = ROOT / image_path

        if not image_path.is_file():
            missing_images += 1
            continue

        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                image_format = image.format

        except Exception as exc:
            unreadable_images += 1
            errors.append(
                f"Cannot read image for {doc_id}: {exc}"
            )
            continue

        if (width, height) != EXPECTED_SIZE:
            bad_dimensions += 1

        if image_format != "PNG":
            errors.append(
                f"{doc_id}: expected PNG image, got {image_format!r}"
            )

        actual_hash = hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest()

        if actual_hash != annotation.get("image_sha256"):
            bad_hashes += 1

        entities = annotation.get("entities")

        if not isinstance(entities, list):
            bad_boxes += 1
            continue

        for entity in entities:
            if not isinstance(entity, dict):
                bad_boxes += 1
                continue

            bbox = entity.get("bbox")

            if not isinstance(bbox, list) or len(bbox) != 4:
                bad_boxes += 1
                continue

            x0, y0, x1, y1 = bbox

            valid_bbox = (
                all(type(value) is int for value in bbox)
                and 0 <= x0 < x1 <= width
                and 0 <= y0 < y1 <= height
                and isinstance(entity.get("text"), str)
                and bool(entity.get("text"))
                and entity.get("page_number") == 1
            )

            if not valid_bbox:
                bad_boxes += 1

    if missing_images:
        errors.append(f"Missing images: {missing_images}")

    if bad_dimensions:
        errors.append(f"Wrong image dimensions: {bad_dimensions}")

    if bad_hashes:
        errors.append(f"Image hash mismatches: {bad_hashes}")

    if bad_boxes:
        errors.append(f"Invalid entities or bounding boxes: {bad_boxes}")

    # ---------------------------------------------------------
    # 6. Check template consistency and coverage
    # ---------------------------------------------------------
    inconsistent_group_templates = {
        group: sorted(templates)
        for group, templates in group_templates.items()
        if len(templates) > 1
    }

    if inconsistent_group_templates:
        errors.append(
            "Template inconsistency within groups: "
            f"{inconsistent_group_templates}"
        )

    unexpected_templates = sorted(
        set(template_counts) - EXPECTED_TEMPLATES
    )
    missing_templates = sorted(
        EXPECTED_TEMPLATES - set(template_counts)
    )

    if unexpected_templates:
        errors.append(f"Unexpected templates: {unexpected_templates}")

    if missing_templates:
        errors.append(f"Templates not represented: {missing_templates}")

    # ---------------------------------------------------------
    # 7. Return report
    # ---------------------------------------------------------
    return {
        "validation_passed": not errors,
        "errors": errors,
        "source_record_count": len(source_records),
        "source_unique_document_ids": len(source_by_id),
        "annotation_count": len(annotations),
        "unique_annotation_ids": len(annotation_by_id),
        "split_counts": split_counts,
        "split_record_count": len(split_by_id),
        "missing_annotations": len(missing_annotations),
        "unexpected_annotations": len(unexpected_annotations),
        "source_alignment_error_count": len(source_alignment_errors),
        "source_alignment_errors": source_alignment_errors[:50],
        "missing_images": missing_images,
        "unreadable_images": unreadable_images,
        "wrong_image_dimensions": bad_dimensions,
        "image_hash_mismatches": bad_hashes,
        "invalid_entities_or_boxes": bad_boxes,
        "split_group_leakage": leakage,
        "inconsistent_group_templates": inconsistent_group_templates,
        "template_counts": dict(template_counts),
    }


def main() -> int:
    try:
        report = validate()
    except (OSError, ValueError, KeyError) as exc:
        report = {
            "validation_passed": False,
            "errors": [str(exc)],
        }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(
        "\nReport saved to: "
        f"{REPORT_FILE.relative_to(ROOT)}"
    )

    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
