
from copy import deepcopy

from app.data.annotation_validation import validate_source_alignment
from app.data.invoice_renderer import render_invoice
from app.data.synthetic_invoice_dataset import generate_invoice_dataset


def make_pair(tmp_path, record_index=0):
    record = deepcopy(
        generate_invoice_dataset(base_count=1, seed=42)[record_index]
    )
    annotation = render_invoice(record, tmp_path)
    return record, annotation


def test_valid_invoice_annotations_match_source(tmp_path):
    record, annotation = make_pair(tmp_path, 0)

    assert validate_source_alignment(record, annotation) == []


def test_defective_invoice_annotations_match_mutated_source(tmp_path):
    records = generate_invoice_dataset(base_count=1, seed=42)

    # Check each of the ten defective variants against its own source record.
    for record in records[1:]:
        annotation = render_invoice(record, tmp_path)
        assert validate_source_alignment(record, annotation) == []


def test_wrong_entity_text_is_detected(tmp_path):
    record, annotation = make_pair(tmp_path)
    annotation = deepcopy(annotation)

    target = next(
        entity for entity in annotation["entities"]
        if entity["label"] == "INVOICE_NUMBER"
    )
    target["text"] = "WRONG-INVOICE-NUMBER"

    errors = validate_source_alignment(record, annotation)

    assert any("missing annotation INVOICE_NUMBER" in e for e in errors)
    assert any("unexpected annotation INVOICE_NUMBER" in e for e in errors)


def test_missing_entity_is_detected(tmp_path):
    record, annotation = make_pair(tmp_path)
    annotation = deepcopy(annotation)

    annotation["entities"] = [
        entity for entity in annotation["entities"]
        if entity["label"] != "SUPPLIER_NAME"
    ]

    errors = validate_source_alignment(record, annotation)

    assert any("missing annotation SUPPLIER_NAME" in e for e in errors)


def test_missing_source_field_has_no_expected_entity(tmp_path):
    record, annotation = make_pair(tmp_path, 1)

    # INV-001 is the missing-invoice-number variant.
    assert record["invoice"]["invoice_number"] is None
    assert validate_source_alignment(record, annotation) == []


def test_split_group_mismatch_is_detected(tmp_path):
    record, annotation = make_pair(tmp_path)
    annotation = deepcopy(annotation)
    annotation["split_group_id"] = "incorrect-family"

    errors = validate_source_alignment(record, annotation)

    assert any("split_group_id mismatch" in e for e in errors)
