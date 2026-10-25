
"""Compare Tesseract PSM 4 and PSM 6 across all rendered invoices."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ocr.evaluation import evaluate_document, summarize_rows
from app.ocr.tesseract_engine import run_tesseract

ANNOTATIONS_FILE = (
    ROOT / "datasets/processed/rendered_invoice_annotations.jsonl"
)
OUTPUT_DIR = ROOT / "datasets/processed/ocr/full_variant_experiment"

VARIANTS = {
    "psm6_baseline": "--oem 3 --psm 6",
    "psm4_candidate": "--oem 3 --psm 4",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL and reject malformed records."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}, line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected an object at {path}, line {line_number}"
                )
            records.append(record)
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL without modifying source files."""
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    annotations = read_jsonl(ANNOTATIONS_FILE)

    ids = [row.get("document_id") for row in annotations]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("Every annotation must have a valid document_id.")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document IDs found in annotations.")

    annotations.sort(key=lambda row: str(row["document_id"]))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison = {
        "status": "completed",
        "image_count": len(annotations),
        "template_counts": {},
        "variants": {},
        "scope": (
            "All synthetic rendered invoices across all dataset splits. "
            "This is a full synthetic-dataset comparison, not a held-out "
            "test-set score or evidence of real-world generalization."
        ),
    }

    for variant_name, config in VARIANTS.items():
        print(f"\n=== {variant_name}: {config} ===")
        ocr_results = []
        started_all = time.perf_counter()

        for index, annotation in enumerate(annotations, start=1):
            print(
                f"[{index}/{len(annotations)}] "
                f"{annotation['document_id']} | "
                f"{annotation.get('template_id')}"
            )

            image_path = Path(annotation["image_path"])
            if not image_path.is_absolute():
                image_path = ROOT / image_path

            result = run_tesseract(
                image_path,
                document_id=annotation["document_id"],
                config=config,
            )
            result["template_id"] = annotation.get("template_id")
            result["split_group_id"] = annotation.get("split_group_id")
            ocr_results.append(result)

        total_elapsed = time.perf_counter() - started_all
        ocr_by_id = {
            row["document_id"]: row for row in ocr_results
        }

        if len(ocr_by_id) != len(annotations):
            raise ValueError("OCR result IDs are missing or duplicated.")

        field_rows = []
        for annotation in annotations:
            field_rows.extend(
                evaluate_document(
                    annotation,
                    ocr_by_id[annotation["document_id"]],
                )
            )

        metrics = summarize_rows(field_rows)
        runtimes = [
            row["runtime_seconds"] for row in ocr_results
        ]

        report = {
            "variant": variant_name,
            "config": config,
            "image_count": len(ocr_results),
            "elapsed_wall_seconds": round(total_elapsed, 3),
            "runtime_seconds": {
                "sum_per_image": round(sum(runtimes), 3),
                "mean": round(statistics.mean(runtimes), 6),
                "median": round(statistics.median(runtimes), 6),
            },
            "word_count": sum(row["word_count"] for row in ocr_results),
            "empty_ocr_results": sum(
                not row["text"].strip() for row in ocr_results
            ),
            "metrics": metrics,
        }

        write_jsonl(
            OUTPUT_DIR / f"{variant_name}_ocr.jsonl",
            ocr_results,
        )
        write_jsonl(
            OUTPUT_DIR / f"{variant_name}_field_metrics.jsonl",
            field_rows,
        )
        (OUTPUT_DIR / f"{variant_name}_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        comparison["variants"][variant_name] = {
            "config": config,
            "elapsed_wall_seconds": report["elapsed_wall_seconds"],
            "mean_runtime_seconds": report["runtime_seconds"]["mean"],
            "field_count": metrics["field_count"],
            "field_detection_coverage": metrics[
                "field_detection_coverage"
            ],
            "field_exact_match_accuracy": metrics[
                "field_exact_match_accuracy"
            ],
            "cer": metrics["cer"],
            "wer": metrics["wer"],
            "by_label": metrics["by_label"],
            "by_template": metrics["by_template"],
        }

        print(json.dumps(
            comparison["variants"][variant_name],
            indent=2,
            ensure_ascii=False,
        ))

    comparison["template_counts"] = {
        template: sum(
            row.get("template_id") == template for row in annotations
        )
        for template in sorted({
            str(row.get("template_id")) for row in annotations
        })
    }

    (OUTPUT_DIR / "comparison_report.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nFull comparison saved to: {OUTPUT_DIR}")
    print("Review comparison_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
