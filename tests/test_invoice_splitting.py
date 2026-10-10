import copy
import json

import pytest

from app.data.invoice_splitting import (
    build_invoice_split_report,
    split_invoice_records,
)
from scripts.split_synthetic_invoices import create_split_artifacts


def make_records(family_count=10):
    records = []

    for family in range(family_count):
        group_id = f"family-{family:03d}"

        records.append({
            "document_id": f"INV-{family:03d}-VALID",
            "document_type": "invoice",
            "split_group_id": group_id,
            "invoice": {"invoice_number": f"INV-{family:03d}"},
            "expected_risks": [],
        })

        for defect_number in range(1, 11):
            risks = [{
                "risk_type": f"INV-{defect_number:03d}",
                "severity": "high",
                "evidence": "Synthetic test defect",
                "label_source": "synthetic_ground_truth",
            }]

            # Model the intentional INV-008 -> INV-010 cascade.
            if defect_number == 8:
                risks.append({
                    "risk_type": "INV-010",
                    "severity": "high",
                    "evidence": "Synthetic total reconciliation defect",
                    "label_source": "synthetic_ground_truth",
                })

            records.append({
                "document_id": f"INV-{family:03d}-DEF-{defect_number:03d}",
                "document_type": "invoice",
                "split_group_id": group_id,
                "invoice": {"invoice_number": f"INV-{family:03d}"},
                "expected_risks": risks,
            })

    return records


def test_all_records_are_preserved():
    records = make_records()
    splits = split_invoice_records(records)

    original_ids = {r["document_id"] for r in records}
    split_ids = {
        r["document_id"]
        for split_records in splits.values()
        for r in split_records
    }

    assert split_ids == original_ids
    assert sum(len(items) for items in splits.values()) == len(records)


def test_invoice_families_never_cross_splits():
    splits = split_invoice_records(make_records())

    owners = {}
    for split_name, records in splits.items():
        for record in records:
            group_id = record["split_group_id"]
            assert group_id not in owners or owners[group_id] == split_name
            owners[group_id] = split_name


def test_same_seed_produces_same_assignments():
    records = make_records()

    def assignments(seed):
        splits = split_invoice_records(records, seed=seed)
        return {
            record["document_id"]: split_name
            for split_name, items in splits.items()
            for record in items
        }

    assert assignments(42) == assignments(42)


def test_different_seed_changes_assignments():
    records = make_records()

    def assignments(seed):
        splits = split_invoice_records(records, seed=seed)
        return {
            record["document_id"]: split_name
            for split_name, items in splits.items()
            for record in items
        }

    assert assignments(42) != assignments(7)


def test_split_ratios_allocate_whole_groups():
    splits = split_invoice_records(make_records(30))

    group_counts = {
        name: len({record["split_group_id"] for record in records})
        for name, records in splits.items()
    }

    assert group_counts == {
        "train": 21,
        "validation": 5,
        "test": 4,
    }


def test_report_includes_defect_counts_and_no_leakage():
    splits = split_invoice_records(make_records())
    report = build_invoice_split_report(
        splits,
        seed=42,
        ratios={"train": 0.70, "validation": 0.15, "test": 0.15},
    )

    assert report["leakage_check_passed"] is True
    assert report["document_id_check_passed"] is True
    assert report["total_records"] == 110
    assert report["unique_document_ids"] == 110

    for split_info in report["splits"].values():
        assert split_info["group_count"] > 0
        assert split_info["defect_label_counts"]


def test_duplicate_document_ids_are_rejected():
    records = make_records()
    records.append(copy.deepcopy(records[0]))

    with pytest.raises(ValueError, match="Duplicate document_id"):
        split_invoice_records(records)


def test_missing_group_id_is_rejected():
    records = make_records()
    records[0]["split_group_id"] = ""

    with pytest.raises(ValueError, match="split_group_id"):
        split_invoice_records(records)


def test_invalid_ratios_are_rejected():
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_invoice_records(
            make_records(),
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.2,
        )


def test_artifacts_preserve_nested_invoice_data(tmp_path):
    records = make_records(10)
    output_dir = tmp_path / "splits"
    report_path = tmp_path / "report.json"

    report = create_split_artifacts(
        records,
        output_dir=output_dir,
        report_path=report_path,
    )

    saved_records = []
    for split_name in ("train", "validation", "test"):
        path = output_dir / f"{split_name}.jsonl"
        assert path.exists()
        saved_records.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )

    assert len(saved_records) == len(records)
    assert saved_records[0]["invoice"]["invoice_number"]
    assert report_path.exists()
    assert report["leakage_check_passed"] is True