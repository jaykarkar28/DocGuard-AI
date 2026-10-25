
"""Run a reproducible Tesseract OCR pilot on rendered invoices."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ocr.tesseract_engine import (  # noqa: E402
    DEFAULT_OCR_CONFIG,
    run_tesseract,
)


ANNOTATIONS_FILE = (
    ROOT / "datasets/processed/rendered_invoice_annotations.jsonl"
)
OUTPUT_DIR = ROOT / "datasets/processed/ocr"
RESULTS_FILE = OUTPUT_DIR / "tesseract_pilot.jsonl"
REPORT_FILE = OUTPUT_DIR / "tesseract_pilot_report.json"

EXPECTED_TEMPLATES = ("classic", "minimal", "sidebar")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file and validate that each row is an object."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")

    records: list[dict[str, Any]] = []

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
                    f"Expected an object in {path}, line {line_number}"
                )

            records.append(record)

    return records


def select_pilot(
    annotations: list[dict[str, Any]],
    *,
    samples_per_template: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select a reproducible sample balanced across known templates."""
    if samples_per_template < 1:
        raise ValueError("samples_per_template must be at least 1")

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []

    for template in EXPECTED_TEMPLATES:
        candidates = [
            record
            for record in annotations
            if record.get("template_id") == template
            and isinstance(record.get("image_path"), str)
            and record.get("image_path")
        ]

        candidates.sort(
            key=lambda record: str(record.get("document_id", ""))
        )

        if len(candidates) < samples_per_template:
            raise ValueError(
                f"Template {template!r} has {len(candidates)} usable "
                f"annotations; need {samples_per_template}."
            )

        selected.extend(
            rng.sample(candidates, samples_per_template)
        )

    return selected


def resolve_image_path(raw_path: str) -> Path:
    """Resolve an annotation image path relative to the project root."""
    path = Path(raw_path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def build_report(
    results: list[dict[str, Any]],
    *,
    seed: int,
    samples_per_template: int,
    config: str = DEFAULT_OCR_CONFIG,
) -> dict[str, Any]:
    """Build a descriptive report without claiming accuracy metrics."""
    runtimes = [row["runtime_seconds"] for row in results]
    word_counts = [row["word_count"] for row in results]
    template_counts = Counter(row["template_id"] for row in results)

    return {
        "status": "completed",
        "engine": "tesseract",
        "engine_version": (
            results[0]["engine_version"] if results else None
        ),
        "language": "eng",
        "config": config,
        "sampling_seed": seed,
        "samples_per_template": samples_per_template,
        "image_count": len(results),
        "template_counts": dict(sorted(template_counts.items())),
        "runtime_seconds": {
            "total": round(sum(runtimes), 6),
            "mean": (
                round(statistics.mean(runtimes), 6)
                if runtimes
                else None
            ),
            "median": (
                round(statistics.median(runtimes), 6)
                if runtimes
                else None
            ),
        },
        "word_count": {
            "total": sum(word_counts),
            "mean": (
                round(statistics.mean(word_counts), 3)
                if word_counts
                else None
            ),
        },
        "empty_ocr_results": sum(
            1 for row in results if not row["text"].strip()
        ),
        "accuracy_metrics": {
            "cer": None,
            "wer": None,
            "field_exact_match": None,
            "note": (
                "Not measured by this pilot runner. Use the dedicated "
                "OCR evaluation script with the corresponding annotations "
                "to calculate field-level and text-level metrics."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a template-balanced Tesseract OCR pilot."
    )
    parser.add_argument(
        "--samples-per-template",
        type=int,
        default=10,
        help="Number of images selected per template (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic sampling seed (default: 42).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_OCR_CONFIG,
        help=(
            "Tesseract configuration. Default: "
            f"{DEFAULT_OCR_CONFIG!r}. "
            "Use '--config \"--oem 3 --psm 6\"' to run the baseline."
        ),
    )
    args = parser.parse_args()

    annotations = read_jsonl(ANNOTATIONS_FILE)
    pilot = select_pilot(
        annotations,
        samples_per_template=args.samples_per_template,
        seed=args.seed,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    for index, annotation in enumerate(pilot, start=1):
        document_id = annotation.get("document_id")
        raw_image_path = annotation["image_path"]
        image_path = resolve_image_path(raw_image_path)

        print(
            f"[{index}/{len(pilot)}] "
            f"{document_id} | {annotation.get('template_id')}"
        )

        result = run_tesseract(
            image_path,
            document_id=str(document_id),
            config=args.config,
        )
        result["template_id"] = annotation.get("template_id")
        result["split_group_id"] = annotation.get("split_group_id")
        results.append(result)

    with RESULTS_FILE.open("w", encoding="utf-8", newline="\n") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    report = build_report(
        results,
        seed=args.seed,
        samples_per_template=args.samples_per_template,
        config=args.config,
    )
    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nPilot report:")
    print(json.dumps(report, indent=2))
    print(f"\nOCR results: {RESULTS_FILE.relative_to(ROOT)}")
    print(f"Summary report: {REPORT_FILE.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
