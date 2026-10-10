
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.data.invoice_renderer import render_jsonl_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render synthetic invoice JSONL records into PNG images."
    )
    parser.add_argument(
        "--input",
        default="datasets/processed/synthetic_invoices.jsonl",
    )
    parser.add_argument(
        "--images",
        default="datasets/processed/invoice_images",
    )
    parser.add_argument(
        "--annotations",
        default="datasets/processed/rendered_invoice_annotations.jsonl",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    image_dir = PROJECT_ROOT / args.images
    annotation_path = PROJECT_ROOT / args.annotations

    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")

    result = render_jsonl_dataset(input_path, image_dir, annotation_path)

    report = {
        **result,
        "input": str(input_path.relative_to(PROJECT_ROOT)),
        "image_directory": str(image_dir.relative_to(PROJECT_ROOT)),
        "annotations": str(annotation_path.relative_to(PROJECT_ROOT)),
        "image_size": [1275, 1650],
        "templates": ["classic", "sidebar", "minimal"],
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
