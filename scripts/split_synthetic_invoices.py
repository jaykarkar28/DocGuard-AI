"""Create train/validation/test splits for the synthetic invoice pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data.invoice_splitting import (  # noqa: E402
    build_invoice_split_report,
    split_invoice_records,
    write_invoice_splits,
)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with Path(path).open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            records.append(record)

    return records


def create_split_artifacts(
    records: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    report_path: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    ratios = {
        "train": train_ratio,
        "validation": val_ratio,
        "test": test_ratio,
    }

    splits = split_invoice_records(
        records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    report = build_invoice_split_report(
        splits,
        seed=seed,
        ratios=ratios,
    )

    if not report["leakage_check_passed"]:
        raise ValueError(f"Invoice-family leakage detected: {report['group_leakage']}")

    if not report["document_id_check_passed"]:
        raise ValueError("Duplicate document IDs detected across splits.")

    # Write artifacts only after all split-quality checks have passed.
    write_invoice_splits(splits, output_dir)

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split synthetic invoice records by invoice family."
    )
    parser.add_argument(
        "--input",
        default="datasets/processed/synthetic_invoices.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/splits/synthetic_invoices",
    )
    parser.add_argument(
        "--report",
        default="datasets/reports/synthetic_invoice_split_report.json",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_jsonl(args.input)

    report = create_split_artifacts(
        records,
        output_dir=args.output_dir,
        report_path=args.report,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(json.dumps(report, indent=2))
    print("\nSynthetic invoice splits and report created successfully.")


if __name__ == "__main__":
    main()