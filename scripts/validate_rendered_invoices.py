
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPLITS_DIR = ROOT / "datasets/splits/synthetic_invoices"
ANNOTATIONS_FILE = ROOT / "datasets/processed/rendered_invoice_annotations.jsonl"
EXPECTED_SIZE = (1275, 1650)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def validate() -> dict:
    errors: list[str] = []
    annotations = read_jsonl(ANNOTATIONS_FILE)

    # 1. Unique annotation IDs.
    ids = [row.get("document_id") for row in annotations]
    id_counts = Counter(ids)
    duplicates = sorted(
        str(doc_id) for doc_id, count in id_counts.items()
        if count > 1
    )
    if duplicates:
        errors.append(f"Duplicate annotation IDs: {duplicates[:10]}")

    annotation_by_id = {
        row["document_id"]: row
        for row in annotations
        if row.get("document_id")
    }

    # 2. Load source split membership.
    split_by_id: dict[str, str] = {}
    split_groups: dict[str, set[str]] = defaultdict(set)
    split_counts: dict[str, int] = {}

    for split_name in ("train", "validation", "test"):
        path = SPLITS_DIR / f"{split_name}.jsonl"
        if not path.is_file():
            errors.append(f"Missing split file: {path}")
            continue

        records = read_jsonl(path)
        split_counts[split_name] = len(records)

        for record in records:
            doc_id = record.get("document_id")
            group_id = record.get("split_group_id")

            if not doc_id or not group_id:
                errors.append(
                    f"Missing document/group ID in {split_name}"
                )
                continue

            if doc_id in split_by_id:
                errors.append(f"Duplicate split document ID: {doc_id}")
                continue

            split_by_id[doc_id] = split_name
            split_groups[group_id].add(split_name)

    leakage = {
        group: sorted(names)
        for group, names in split_groups.items()
        if len(names) > 1
    }
    if leakage:
        errors.append(f"Split-group leakage: {leakage}")

    # 3. Annotation coverage and group/template consistency.
    missing_annotations = sorted(set(split_by_id) - set(annotation_by_id))
    unexpected_annotations = sorted(set(annotation_by_id) - set(split_by_id))

    if missing_annotations:
        errors.append(
            f"Missing annotations for {len(missing_annotations)} split records"
        )
    if unexpected_annotations:
        errors.append(
            f"{len(unexpected_annotations)} annotations have no split record"
        )

    group_templates: dict[str, set[str]] = defaultdict(set)
    template_counts: Counter = Counter()
    missing_images = 0
    bad_dimensions = 0
    bad_hashes = 0
    bad_boxes = 0

    for doc_id, annotation in annotation_by_id.items():
        group_id = annotation.get("split_group_id")
        template_id = annotation.get("template_id")

        if group_id and template_id:
            group_templates[group_id].add(template_id)
        if template_id:
            template_counts[template_id] += 1

        image_path = Path(annotation.get("image_path", ""))
        if not image_path.is_absolute():
            image_path = ROOT / image_path

        if not image_path.is_file():
            missing_images += 1
            continue

        try:
            with Image.open(image_path) as image:
                if image.size != EXPECTED_SIZE:
                    bad_dimensions += 1
                width, height = image.size
        except Exception as exc:
            errors.append(f"Cannot read image for {doc_id}: {exc}")
            continue

        actual_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if actual_hash != annotation.get("image_sha256"):
            bad_hashes += 1

        for entity in annotation.get("entities", []):
            bbox = entity.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                bad_boxes += 1
                continue

            x0, y0, x1, y1 = bbox
            if not (
                all(isinstance(v, int) for v in bbox)
                and 0 <= x0 < x1 <= width
                and 0 <= y0 < y1 <= height
                and bool(entity.get("text"))
                and entity.get("page_number") == 1
            ):
                bad_boxes += 1

    inconsistent_templates = {
        group: sorted(templates)
        for group, templates in group_templates.items()
        if len(templates) > 1
    }

    if missing_images:
        errors.append(f"Missing images: {missing_images}")
    if bad_dimensions:
        errors.append(f"Wrong image dimensions: {bad_dimensions}")
    if bad_hashes:
        errors.append(f"Image hash mismatches: {bad_hashes}")
    if bad_boxes:
        errors.append(f"Invalid bounding boxes/entities: {bad_boxes}")
    if inconsistent_templates:
        errors.append(
            f"Template inconsistency within groups: {inconsistent_templates}"
        )

    return {
        "validation_passed": not errors,
        "errors": errors,
        "annotation_count": len(annotations),
        "unique_annotation_ids": len(set(ids)),
        "split_counts": split_counts,
        "split_record_count": len(split_by_id),
        "missing_annotations": len(missing_annotations),
        "unexpected_annotations": len(unexpected_annotations),
        "missing_images": missing_images,
        "wrong_image_dimensions": bad_dimensions,
        "image_hash_mismatches": bad_hashes,
        "invalid_entities_or_boxes": bad_boxes,
        "split_group_leakage": leakage,
        "inconsistent_group_templates": inconsistent_templates,
        "template_counts": dict(template_counts),
    }


if __name__ == "__main__":
    report = validate()
    report_path = ROOT / "datasets/reports/render_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {report_path.relative_to(ROOT)}")
    raise SystemExit(0 if report["validation_passed"] else 1)
