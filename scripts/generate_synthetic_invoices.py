
"""Generate and validate a structured synthetic invoice dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Support running this script directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data.synthetic_invoice_dataset import (
    DEFECTS,
    generate_invoice_dataset,
)


def validate_invoice_dataset(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate dataset identity, provenance, labels, and group integrity."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    group_counts: Counter[str] = Counter()
    defect_counts: Counter[str] = Counter()
    valid_count = 0
    defective_count = 0

    expected_rule_ids = set(DEFECTS)

    for index, record in enumerate(records):
        prefix = f"Record {index}"

        document_id = record.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            errors.append(f"{prefix}: missing or invalid document_id")
        elif document_id in seen_ids:
            errors.append(f"{prefix}: duplicate document_id {document_id!r}")
        else:
            seen_ids.add(document_id)

        if record.get("document_type") != "invoice":
            errors.append(f"{prefix}: document_type must be 'invoice'")

        source = record.get("source")
        if not isinstance(source, dict):
            errors.append(f"{prefix}: source must be an object")
        else:
            if source.get("dataset") != "synthetic":
                errors.append(f"{prefix}: unexpected dataset provenance")
            if source.get("license") != "CC0-1.0":
                errors.append(f"{prefix}: unexpected license declaration")

        group_id = record.get("split_group_id")
        if not isinstance(group_id, str) or not group_id.strip():
            errors.append(f"{prefix}: missing split_group_id")
        else:
            group_counts[group_id] += 1

        invoice = record.get("invoice")
        if not isinstance(invoice, dict):
            errors.append(f"{prefix}: invoice payload must be an object")

        expected_risks = record.get("expected_risks")
        if not isinstance(expected_risks, list):
            errors.append(f"{prefix}: expected_risks must be a list")
            continue

        if not expected_risks:
            valid_count += 1
            continue

        defective_count += 1
        record_labels: set[str] = set()

        for risk in expected_risks:
            if not isinstance(risk, dict):
                errors.append(f"{prefix}: each expected risk must be an object")
                continue

            rule_id = risk.get("risk_type")

            if rule_id not in expected_rule_ids:
                errors.append(f"{prefix}: unknown risk type {rule_id!r}")
                continue

            if rule_id in record_labels:
                errors.append(
                    f"{prefix}: duplicate risk label {rule_id!r}"
                )
            record_labels.add(rule_id)
            defect_counts[rule_id] += 1

            if risk.get("label_source") != "synthetic_ground_truth":
                errors.append(
                    f"{prefix}: risk label must use "
                    "'synthetic_ground_truth' provenance"
                )

            expected_severity = DEFECTS[rule_id]["severity"]
            if risk.get("severity") != expected_severity:
                errors.append(
                    f"{prefix}: incorrect severity for {rule_id}"
                )

            if not isinstance(risk.get("evidence"), str) or not risk[
                "evidence"
            ].strip():
                errors.append(
                    f"{prefix}: missing evidence for {rule_id}"
                )

    expected_defect_ids = sorted(expected_rule_ids)
    missing_defects = [
        rule_id
        for rule_id in expected_defect_ids
        if defect_counts[rule_id] == 0
    ]

    if missing_defects:
        errors.append(
            f"Missing defect coverage: {', '.join(missing_defects)}"
        )

    return {
        "validation_passed": not errors,
        "errors": errors,
        "total_records": len(records),
        "unique_document_ids": len(seen_ids),
        "valid_records": valid_count,
        "defective_records": defective_count,
        "unique_split_groups": len(group_counts),
        "records_per_split_group": dict(sorted(group_counts.items())),
        "defect_label_counts": dict(sorted(defect_counts.items())),
        "missing_defect_types": missing_defects,
    }


def write_jsonl(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Write structured invoice records without DocumentRecord coercion."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a validated synthetic invoice dataset."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of base invoices; each creates 11 records.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--output",
        default="datasets/processed/synthetic_invoices.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--report",
        default="datasets/reports/synthetic_invoice_report.json",
        help="Validation report path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output files to be overwritten.",
    )
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")

    output_path = Path(args.output)
    report_path = Path(args.report)

    if output_path.resolve() == report_path.resolve():
        parser.error("--output and --report must be different paths")

    if not args.overwrite:
        existing = [
            str(path)
            for path in (output_path, report_path)
            if path.exists()
        ]
        if existing:
            parser.error(
                "Output already exists: "
                + ", ".join(existing)
                + ". Choose another path or pass --overwrite."
            )

    records = generate_invoice_dataset(
        base_count=args.count,
        seed=args.seed,
    )
    report = validate_invoice_dataset(records)
    report["seed"] = args.seed
    report["base_invoice_count"] = args.count
    report["expected_records"] = args.count * 11
    report["output_path"] = str(output_path)
    report["license_declaration"] = "CC0-1.0"
    report["note"] = (
        "Synthetic structured records; not rendered invoice images. "
        "Confirm project release terms before external distribution."
    )

    if len(records) != args.count * 11:
        report["validation_passed"] = False
        report["errors"].append("Unexpected generated record count")

    if not report["validation_passed"]:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    # Validate first, then write both artifacts.
    write_jsonl(records, output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nDataset and validation report created successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
