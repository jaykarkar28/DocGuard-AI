
"""Compare Tesseract configurations on the existing 30-image OCR pilot."""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ocr.evaluation import evaluate_document, summarize_rows
from app.ocr.tesseract_engine import (
    configure_tesseract,
    extract_word_records,
    reconstruct_text,
)

ANNOTATIONS_FILE = (
    ROOT / "datasets/processed/rendered_invoice_annotations.jsonl"
)
PILOT_FILE = ROOT / "datasets/processed/ocr/tesseract_pilot.jsonl"
OUTPUT_DIR = ROOT / "datasets/processed/ocr/variant_experiment"

# Keep the existing baseline, then compare one change at a time.
VARIANTS = [
    {"name": "baseline_psm6", "psm": 6, "preprocess": "none"},
    {"name": "psm4", "psm": 4, "preprocess": "none"},
    {"name": "psm11", "psm": 11, "preprocess": "none"},
    {"name": "gray_autocontrast_psm6", "psm": 6, "preprocess": "gray_autocontrast"},
    {"name": "upscale2x_gray_psm6", "psm": 6, "preprocess": "upscale2x_gray"},
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL and validate that every row is an object."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}, line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}, line {line_number}"
                )
            records.append(row)
    return records


def unique_by_id(
    records: list[dict[str, Any]], source: str
) -> dict[str, dict[str, Any]]:
    """Index records and reject missing or duplicate IDs."""
    indexed = {}
    for row in records:
        document_id = row.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"{source} contains a missing document_id")
        if document_id in indexed:
            raise ValueError(
                f"{source} contains duplicate ID: {document_id}"
            )
        indexed[document_id] = row
    return indexed


def resolve_image_path(raw_path: str) -> Path:
    """Resolve an annotation image path."""
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Invoice image not found: {path}")
    return path


def preprocess_image(
    image: Image.Image, method: str
) -> tuple[Image.Image, float]:
    """Return processed image and coordinate scale relative to source."""
    if method == "none":
        return image.copy(), 1.0

    gray = ImageOps.grayscale(image)

    if method == "gray_autocontrast":
        return ImageOps.autocontrast(gray), 1.0

    if method == "upscale2x_gray":
        enlarged = gray.resize(
            (gray.width * 2, gray.height * 2),
            Image.Resampling.LANCZOS,
        )
        return ImageOps.autocontrast(enlarged), 2.0

    raise ValueError(f"Unknown preprocessing method: {method}")


def restore_original_coordinates(
    words: list[dict[str, Any]], scale: float
) -> list[dict[str, Any]]:
    """Map OCR boxes back to the original invoice coordinate system."""
    if scale == 1.0:
        return words

    restored = []
    for word in words:
        item = dict(word)
        left, top, right, bottom = item["bbox"]
        item["bbox"] = [
            round(left / scale),
            round(top / scale),
            round(right / scale),
            round(bottom / scale),
        ]
        restored.append(item)
    return restored


def run_variant(
    annotation: dict[str, Any], variant: dict[str, Any]
) -> dict[str, Any]:
    """Run one OCR configuration on one image."""
    image_path = resolve_image_path(annotation["image_path"])
    started = time.perf_counter()

    with Image.open(image_path) as opened:
        opened.load()
        original = opened.convert("RGB")
        original_width, original_height = original.size
        processed, scale = preprocess_image(
            original, variant["preprocess"]
        )

        words = extract_word_records(
            processed,
            language="eng",
            config=f"--oem 3 --psm {variant['psm']}",
        )
        words = restore_original_coordinates(words, scale)

    runtime = time.perf_counter() - started

    return {
        "document_id": annotation["document_id"],
        "image_path": str(image_path),
        "image_width": original_width,
        "image_height": original_height,
        "engine": "tesseract",
        "engine_version": str(
            __import__("pytesseract").get_tesseract_version()
        ),
        "language": "eng",
        "config": f"--oem 3 --psm {variant['psm']}",
        "preprocessing": variant["preprocess"],
        "runtime_seconds": round(runtime, 6),
        "word_count": len(words),
        "mean_word_confidence": (
            round(
                statistics.mean(w["confidence"] for w in words), 4
            )
            if words
            else None
        ),
        "text": reconstruct_text(words),
        "words": words,
        "template_id": annotation.get("template_id"),
        "split_group_id": annotation.get("split_group_id"),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL records."""
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    configure_tesseract()

    annotations = unique_by_id(
        read_jsonl(ANNOTATIONS_FILE), "Annotations"
    )
    pilot_records = read_jsonl(PILOT_FILE)
    pilot_ids = [row.get("document_id") for row in pilot_records]

    if not pilot_ids or any(not isinstance(item, str) for item in pilot_ids):
        raise ValueError("Pilot file is empty or contains invalid IDs.")
    if len(set(pilot_ids)) != len(pilot_ids):
        raise ValueError("Pilot file contains duplicate document IDs.")

    missing = sorted(set(pilot_ids) - set(annotations))
    if missing:
        raise ValueError(f"Pilot IDs missing from annotations: {missing[:10]}")

    pilot_annotations = [annotations[doc_id] for doc_id in pilot_ids]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiment_summary = {
        "status": "completed",
        "image_count": len(pilot_annotations),
        "template_counts": {},
        "variants": [],
        "notes": [
            "Same pilot images are used for every variant.",
            "OCR bounding boxes are mapped back to original image coordinates.",
            "Metrics cover present annotated entity values only.",
            "Pilot results are exploratory; validate a winner on the full dataset.",
        ],
    }

    for variant in VARIANTS:
        name = variant["name"]
        print(f"\n=== {name} ===")
        results = []

        for index, annotation in enumerate(pilot_annotations, start=1):
            print(
                f"[{index}/{len(pilot_annotations)}] "
                f"{annotation['document_id']} | "
                f"{annotation.get('template_id')}"
            )
            results.append(run_variant(annotation, variant))

        all_field_rows = []
        for annotation, result in zip(pilot_annotations, results):
            all_field_rows.extend(evaluate_document(annotation, result))

        metrics = summarize_rows(all_field_rows)
        runtimes = [row["runtime_seconds"] for row in results]

        report = {
            "variant": variant,
            "image_count": len(results),
            "runtime_seconds": {
                "total": round(sum(runtimes), 6),
                "mean": round(statistics.mean(runtimes), 6),
                "median": round(statistics.median(runtimes), 6),
            },
            "metrics": metrics,
        }

        write_jsonl(OUTPUT_DIR / f"{name}.jsonl", results)
        write_jsonl(
            OUTPUT_DIR / f"{name}_field_metrics.jsonl",
            all_field_rows,
        )
        (OUTPUT_DIR / f"{name}_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        experiment_summary["variants"].append({
            "name": name,
            "preprocess": variant["preprocess"],
            "config": f"--oem 3 --psm {variant['psm']}",
            "runtime_seconds": report["runtime_seconds"],
            "field_exact_match_accuracy": metrics[
                "field_exact_match_accuracy"
            ],
            "field_detection_coverage": metrics[
                "field_detection_coverage"
            ],
            "cer": metrics["cer"],
            "wer": metrics["wer"],
            "by_template": metrics["by_template"],
        })

        print(json.dumps(
            experiment_summary["variants"][-1],
            indent=2,
            ensure_ascii=False,
        ))

    (OUTPUT_DIR / "comparison_report.json").write_text(
        json.dumps(experiment_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nExperiment complete: {OUTPUT_DIR}")
    print("Comparison report: comparison_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
