
import json

from app.data.synthetic_invoice_dataset import generate_invoice_dataset
from scripts.generate_synthetic_invoices import (
    validate_invoice_dataset,
    write_jsonl,
)


def test_generated_dataset_passes_validation():
    records = generate_invoice_dataset(base_count=3, seed=42)

    report = validate_invoice_dataset(records)

    assert report["validation_passed"] is True
    assert report["total_records"] == 33
    assert report["valid_records"] == 3
    assert report["defective_records"] == 30
    assert report["unique_document_ids"] == 33
    assert report["unique_split_groups"] == 3


def test_duplicate_document_ids_fail_validation():
    records = generate_invoice_dataset(base_count=1)
    records.append(dict(records[0]))

    report = validate_invoice_dataset(records)

    assert report["validation_passed"] is False
    assert any("duplicate document_id" in error for error in report["errors"])


def test_missing_provenance_fails_validation():
    records = generate_invoice_dataset(base_count=1)
    records[0]["source"]["dataset"] = "unknown"

    report = validate_invoice_dataset(records)

    assert report["validation_passed"] is False
    assert any("provenance" in error for error in report["errors"])


def test_missing_defect_coverage_fails_validation():
    records = generate_invoice_dataset(base_count=1)
    records = [
        record for record in records
        if not any(
            risk["risk_type"] == "INV-007"
            for risk in record["expected_risks"]
        )
    ]

    report = validate_invoice_dataset(records)

    assert report["validation_passed"] is False
    assert "INV-007" in report["missing_defect_types"]


def test_jsonl_writer_preserves_nested_invoice_payload(tmp_path):
    records = generate_invoice_dataset(base_count=1)
    output_path = tmp_path / "invoices.jsonl"

    write_jsonl(records, output_path)

    loaded = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert loaded == records
