import pytest

from app.data.annotation_quality import (
    bbox_iou,
    compare_rendered_annotations,
    find_suspicious_overlaps,
    validate_annotation_entity_pairs,
    validate_annotation_geometry,
)


def test_valid_geometry_passes():
    annotation = {
        "document_id": "DOC-1", "width": 100, "height": 100,
        "entities": [{"label": "TOTAL_AMOUNT", "text": "10.00", "bbox": [10, 10, 40, 20]}],
    }
    assert validate_annotation_geometry(annotation) == []


@pytest.mark.parametrize("bbox", [[0, 0, 0, 10], [1, 2, 101, 20], [-1, 0, 5, 5], None, [1, 2, 3]])
def test_invalid_geometry_is_reported(bbox):
    annotation = {
        "document_id": "DOC-2", "width": 100, "height": 100,
        "entities": [{"label": "X", "text": "x", "bbox": bbox}],
    }
    assert validate_annotation_geometry(annotation)


def test_iou_for_half_overlap():
    assert bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_overlap_finding_is_warning_with_entity_context():
    annotation = {
        "document_id": "DOC-3",
        "entities": [
            {"label": "A", "text": "one", "bbox": [0, 0, 10, 10]},
            {"label": "B", "text": "two", "bbox": [5, 0, 15, 10]},
        ],
    }
    findings = find_suspicious_overlaps(annotation, iou_threshold=0.1)
    assert len(findings) == 1
    assert findings[0]["severity"] == "warning"
    assert findings[0]["left_label"] == "A"
    assert findings[0]["right_label"] == "B"


def test_non_overlapping_boxes_do_not_trigger_finding():
    annotation = {
        "document_id": "DOC-4",
        "entities": [
            {"label": "A", "text": "one", "bbox": [0, 0, 10, 10]},
            {"label": "B", "text": "two", "bbox": [10, 0, 20, 10]},
        ],
    }
    assert find_suspicious_overlaps(annotation) == []


def test_invalid_iou_threshold_rejected():
    with pytest.raises(ValueError):
        find_suspicious_overlaps({"entities": []}, iou_threshold=1.1)


def test_missing_source_value_is_expected_not_an_annotation_error():
    record = {
        "document_id": "INV-001",
        "invoice": {"invoice_number": None, "supplier_name": "Acme", "line_items": []},
    }
    annotation = {
        "entities": [{"label": "SUPPLIER_NAME", "text": "Acme", "bbox": [1, 1, 10, 10]}],
    }
    assert validate_annotation_entity_pairs(record, annotation) == []


def test_missing_and_extra_entity_pairs_are_reported():
    record = {
        "document_id": "DOC-5",
        "invoice": {"invoice_number": "INV-5", "supplier_name": "Acme", "line_items": []},
    }
    annotation = {
        "entities": [{"label": "SUPPLIER_NAME", "text": "Wrong", "bbox": [1, 1, 10, 10]}],
    }
    errors = validate_annotation_entity_pairs(record, annotation)
    assert any("missing annotation INVOICE_NUMBER" in error for error in errors)
    assert any("missing annotation SUPPLIER_NAME='Acme'" in error for error in errors)
    assert any("unexpected annotation SUPPLIER_NAME='Wrong'" in error for error in errors)


def test_rendered_annotation_difference_detected():
    expected = {"document_id": "DOC-6", "template_id": "classic", "width": 100, "height": 100, "entities": []}
    actual = {**expected, "entities": [{"label": "X"}]}
    assert compare_rendered_annotations(expected, actual) == [
        "DOC-6: rendered annotation mismatch for entities"
    ]
