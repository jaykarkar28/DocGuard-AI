
from collections import Counter

import pytest

from app.data.synthetic_invoice_dataset import generate_invoice_dataset
from app.risk.invoice_rules import evaluate_invoice


def test_dataset_contains_one_valid_and_ten_defective_variants():
    records = generate_invoice_dataset(base_count=3, seed=42)

    assert len(records) == 3 * 11


def test_dataset_generation_is_reproducible():
    assert generate_invoice_dataset(2, seed=17) == generate_invoice_dataset(
        2, seed=17
    )


def test_document_ids_are_unique():
    records = generate_invoice_dataset(5)

    ids = [record["document_id"] for record in records]
    assert len(ids) == len(set(ids))


def test_all_variants_from_one_base_share_a_split_group():
    records = generate_invoice_dataset(3)

    groups = {}
    for record in records:
        groups.setdefault(record["split_group_id"], []).append(record)

    assert len(groups) == 3
    assert all(len(group) == 11 for group in groups.values())


def test_ground_truth_labels_are_not_created_by_the_evaluator():
    records = generate_invoice_dataset(2)

    for record in records:
        expected = {
            risk["risk_type"]
            for risk in record["expected_risks"]
        }
        predicted = {
            risk["risk_type"]
            for risk in evaluate_invoice(record["invoice"])
        }

        assert expected == predicted, (
            f"{record['document_id']}: "
            f"expected {expected}, predicted {predicted}"
        )


def test_each_defect_type_is_represented():
    records = generate_invoice_dataset(1)

    counts = Counter(
        risk["risk_type"]
        for record in records
        for risk in record["expected_risks"]
    )

    assert set(counts) == {
        f"INV-{number:03d}" for number in range(1, 11)
    }


def test_valid_examples_have_no_expected_or_detected_risks():
    records = generate_invoice_dataset(3)

    valid_records = [
        record for record in records
        if not record["expected_risks"]
    ]

    assert len(valid_records) == 3

    for record in valid_records:
        assert evaluate_invoice(record["invoice"]) == []


def test_zero_base_count_is_rejected():
    with pytest.raises(ValueError):
        generate_invoice_dataset(base_count=0)
