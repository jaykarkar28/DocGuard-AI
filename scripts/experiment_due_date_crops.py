
"""Compare crop-level Tesseract configurations for invoice due dates.

Runs on all annotated sidebar due-date fields by default. The experiment
preserves raw OCR predictions and never corrects dates using ground truth.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ocr.tesseract_engine import configure_tesseract
import pytesseract


ANNOTATIONS_FILE = (
    ROOT / "datasets/processed/rendered_invoice_annotations.jsonl"
)
OUTPUT_DIR = ROOT / "datasets/processed/ocr/due_date_crop_experiment"

# Four independent crop configurations. "crop_psm4" is the crop baseline,
# not the existing whole-page PSM 4 baseline.
CONFIGURATIONS = {
    "crop_psm4": {
        "preprocess": "none",
        "scale": 1,
        "config": "--oem 3 --psm 4",
    },
    "crop_psm7": {
        "preprocess": "none",
        "scale": 1,
        "config": "--oem 3 --psm 7",
    },
    "gray_autocontrast_psm7": {
        "preprocess": "gray_autocontrast",
        "scale": 1,
        "config": "--oem 3 --psm 7",
    },
    "upscale2_gray_psm7": {
        "preprocess": "gray_autocontrast",
        "scale": 2,
        "config": "--oem 3 --psm 7",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON Lines records and report malformed lines clearly."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

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
                    f"Expected a JSON object on line {line_number}"
                )
            records.append(record)
    return records


def resolve_image_path(raw_path: str) -> Path:
    """Resolve an annotation path, supporting absolute and project paths."""
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Annotated image does not exist: {path}")
    return path


def select_sidebar_due_dates(
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select every sidebar annotation containing a due-date entity."""
    selected = []

    for record in annotations:
        if record.get("template_id") != "sidebar":
            continue

        entities = record.get("entities", [])
        if not isinstance(entities, list):
            continue

        for entity in entities:
            if entity.get("label") != "DUE_DATE":
                continue

            if not isinstance(entity.get("text"), str):
                continue

            if not isinstance(entity.get("bbox"), list):
                continue

            selected.append({
                "document_id": record.get("document_id"),
                "template_id": record.get("template_id"),
                "image_path": record.get("image_path"),
                "reference": entity["text"],
                "bbox": entity["bbox"],
            })

    selected.sort(key=lambda row: str(row["document_id"]))

    if not selected:
        raise ValueError(
            "No sidebar DUE_DATE annotations found. Check the annotations "
            "path, template_id and entity label."
        )

    return selected


def make_crop(
    image: Image.Image,
    bbox: list[Any],
    padding: int = 12,
) -> Image.Image:
    """Crop a field with padding, clipping coordinates to the image."""
    if len(bbox) != 4:
        raise ValueError(f"Invalid bounding box: {bbox}")

    left, top, right, bottom = (int(value) for value in bbox)

    if right <= left or bottom <= top:
        raise ValueError(f"Invalid bounding box dimensions: {bbox}")

    width, height = image.size
    crop_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )

    crop = image.crop(crop_box)
    if crop.width == 0 or crop.height == 0:
        raise ValueError(f"Empty crop produced for bbox: {bbox}")
    return crop


def preprocess_crop(
    crop: Image.Image,
    *,
    preprocess: str,
    scale: int,
) -> Image.Image:
    """Apply a named, reproducible image preprocessing configuration."""
    if preprocess == "none":
        processed = crop.convert("RGB")
    elif preprocess == "gray_autocontrast":
        gray = ImageOps.grayscale(crop)
        processed = ImageOps.autocontrast(gray)
    else:
        raise ValueError(f"Unknown preprocessing mode: {preprocess}")

    if scale > 1:
        processed = processed.resize(
            (processed.width * scale, processed.height * scale),
            Image.Resampling.LANCZOS,
        )

    return processed


def run_crop_ocr(
    crop: Image.Image,
    *,
    config: str,
) -> tuple[str, float]:
    """Run OCR on a crop and return raw text plus elapsed seconds."""
    started = time.perf_counter()
    raw_text = pytesseract.image_to_string(
        crop,
        lang="eng",
        config=config,
    )
    elapsed = time.perf_counter() - started
    return raw_text.strip(), elapsed


def normalize_for_exact_match(value: str) -> str:
    """Normalize whitespace only; preserve punctuation and character errors."""
    return " ".join(value.split())


def evaluate_configuration(
    samples: list[dict[str, Any]],
    *,
    name: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one crop configuration on the same samples."""
    rows = []
    runtimes = []

    for index, sample in enumerate(samples, start=1):
        image_path = resolve_image_path(sample["image_path"])

        with Image.open(image_path) as source:
            source.load()
            crop = make_crop(source.convert("RGB"), sample["bbox"])

        prepared = preprocess_crop(
            crop,
            preprocess=settings["preprocess"],
            scale=settings["scale"],
        )

        prediction, elapsed = run_crop_ocr(
            prepared,
            config=settings["config"],
        )
        runtimes.append(elapsed)

        reference = sample["reference"]
        exact_match = (
            normalize_for_exact_match(reference)
            == normalize_for_exact_match(prediction)
        )

        row = {
            "configuration": name,
            "document_id": sample["document_id"],
            "template_id": sample["template_id"],
            "reference": reference,
            "raw_prediction": prediction,
            "exact_match": exact_match,
            "runtime_seconds": round(elapsed, 6),
            "preprocess": settings["preprocess"],
            "scale": settings["scale"],
            "config": settings["config"],
        }
        rows.append(row)

        if index % 25 == 0 or index == len(samples):
            print(f"  {name}: {index}/{len(samples)} fields")

    correct = sum(row["exact_match"] for row in rows)
    total = len(rows)

    report = {
        "configuration": name,
        "preprocess": settings["preprocess"],
        "scale": settings["scale"],
        "config": settings["config"],
        "field_count": total,
        "exact_match_count": correct,
        "exact_match_accuracy": correct / total if total else None,
        "mean_runtime_seconds": (
            statistics.mean(runtimes) if runtimes else None
        ),
        "median_runtime_seconds": (
            statistics.median(runtimes) if runtimes else None
        ),
        "mismatches": [
            {
                "document_id": row["document_id"],
                "reference": row["reference"],
                "prediction": row["raw_prediction"],
            }
            for row in rows
            if not row["exact_match"]
        ],
        "results": rows,
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark crop-level OCR on sidebar invoice due dates."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=ANNOTATIONS_FILE,
        help="Path to rendered invoice annotations JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for experiment results.",
    )
    args = parser.parse_args()

    configure_tesseract()

    annotations = read_jsonl(args.annotations)
    samples = select_sidebar_due_dates(annotations)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected sidebar due-date fields: {len(samples)}")
    print("Running crop-level OCR configurations...")

    summary = {
        "experiment": "sidebar_due_date_crop_ocr",
        "scope": (
            "All annotated sidebar due-date fields. Synthetic invoice "
            "dataset only; results do not establish real-world accuracy."
        ),
        "sample_count": len(samples),
        "configurations": {},
    }

    for name, settings in CONFIGURATIONS.items():
        print(f"\n=== {name} ===")
        report = evaluate_configuration(
            samples,
            name=name,
            settings=settings,
        )

        result_path = args.output_dir / f"{name}_results.json"
        result_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        summary["configurations"][name] = {
            key: value
            for key, value in report.items()
            if key not in {"results", "mismatches"}
        }
        summary["configurations"][name]["mismatches"] = report["mismatches"]

        print(
            f"Exact match: {report['exact_match_count']}/"
            f"{report['field_count']} "
            f"({report['exact_match_accuracy']:.2%})"
        )
        print(
            "Mean runtime: "
            f"{report['mean_runtime_seconds']:.4f} seconds/field"
        )
        print(f"Mismatches: {len(report['mismatches'])}")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===")
    for name, result in summary["configurations"].items():
        print(
            f"{name:26} "
            f"accuracy={result['exact_match_accuracy']:.2%} "
            f"mean_runtime={result['mean_runtime_seconds']:.4f}s"
        )

    print(f"\nDetailed results: {args.output_dir}")
    print(f"Summary report: {summary_path}")


if __name__ == "__main__":
    main()
