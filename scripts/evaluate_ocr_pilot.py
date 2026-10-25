
"""Evaluate the Tesseract pilot against rendered invoice annotations."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ocr.evaluation import evaluate_document, summarize_rows


ANNOTATIONS_FILE = (
    ROOT / "datasets/processed/rendered_invoice_annotations.jsonl"
)
OCR_RESULTS_FILE = ROOT / "datasets/processed/ocr/tesseract_pilot.jsonl"
OUTPUT_DIR = ROOT / "datasets/processed/ocr"
DETAIL_FILE = OUTPUT_DIR / "tesseract_pilot_field_metrics.jsonl"
REPORT_FILE = OUTPUT_DIR / "tesseract_pilot_evaluation_report.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON Lines records with helpful validation errors."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}, line {line_number}"
                )

            records.append(value)

    return records


def unique_by_document_id(
    records: list[dict[str, Any]],
    *,
    source_name: str,
) -> dict[str, dict[str, Any]]:
    """Index records and reject missing or duplicate document IDs."""
    indexed: dict[str, dict[str, Any]] = {}

    for index, record in enumerate(records, start=1):
        document_id = record.get("document_id")

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                f"{source_name} record {index} has no valid document_id"
            )

        if document_id in indexed:
            raise ValueError(
                f"{source_name} contains duplicate document_id "
                f"{document_id!r}"
            )

        indexed[document_id] = record

    return indexed


def main() -> int:
    all_annotations = unique_by_document_id(
        read_jsonl(ANNOTATIONS_FILE),
        source_name="Annotations",
    )
    ocr_results = unique_by_document_id(
        read_jsonl(OCR_RESULTS_FILE),
        source_name="OCR results",
    )

    annotation_ids = set(all_annotations)
    ocr_ids = set(ocr_results)

    # Evaluate only documents included in the OCR pilot.
    missing_annotations = sorted(ocr_ids - annotation_ids)
    if missing_annotations:
        raise ValueError(
            "OCR pilot contains document IDs with no matching annotation. "
            f"Missing annotations: {missing_annotations[:10]}"
        )

    # Extra annotations are expected: the annotation file contains the
    # complete dataset, while the OCR output contains only the pilot.
    pilot_annotations = {
        document_id: all_annotations[document_id]
        for document_id in ocr_ids
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    document_summaries: list[dict[str, Any]] = []

    for document_id in sorted(ocr_ids):
        annotation = pilot_annotations[document_id]
        result = ocr_results[document_id]

        rows = evaluate_document(annotation, result)
        all_rows.extend(rows)

        document_summaries.append(
            {
                "document_id": document_id,
                "template_id": annotation.get("template_id"),
                "field_count": len(rows),
                "detected_fields": sum(
                    bool(row["detected"]) for row in rows
                ),
                "exact_matches": sum(
                    bool(row["exact_match"]) for row in rows
                ),
            }
        )

    with DETAIL_FILE.open("w", encoding="utf-8", newline="\n") as file:
        for row in all_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "status": "completed",
        "engine": "tesseract",
        "pilot_image_count": len(ocr_results),
        "annotation_dataset_count": len(all_annotations),
        "pilot_annotation_count": len(pilot_annotations),
        "missing_annotation_document_ids": missing_annotations,
        "unmatched_full_dataset_annotation_count": (
            len(annotation_ids - ocr_ids)
        ),
        "evaluation": summarize_rows(all_rows),
        "documents": document_summaries,
        "limitations": [
            "Metrics cover present annotated entity values, not all printed page text.",
            "Field assignment is based on bounding-box geometry and may include neighboring tokens.",
            "Missing source fields are excluded because they have no visible value annotation.",
            "Synthetic invoices do not establish accuracy on real-world invoices.",
        ],
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report["evaluation"], indent=2, ensure_ascii=False))
    print(f"\nPilot documents evaluated: {len(pilot_annotations)}")
    print(f"Full annotation dataset: {len(all_annotations)}")
    print(f"Detailed field metrics: {DETAIL_FILE.relative_to(ROOT)}")
    print(f"Evaluation report: {REPORT_FILE.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
