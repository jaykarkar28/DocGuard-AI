import pytest
from pydantic import ValidationError

from app.data.schemas import (
    DocumentRecord,
    DocumentSource,
    EntityAnnotation,
    PageMetadata,
    RiskAnnotation,
)


@pytest.fixture
def valid_document_data():
    return {
        "document_id": "INV-0001",
        "document_type": "invoice",
        "source": {
            "dataset": "synthetic",
            "license": "internal_synthetic",
            "source_document_id": None,
        },
        "split_group_id": "invoice-template-001",
        "pages": [
            {
                "page_number": 1,
                "image_path": "datasets/processed/images/inv-0001.png",
                "width": 1000,
                "height": 1400,
            }
        ],
        "entities": [
            {
                "label": "INVOICE_NUMBER",
                "text": "INV-0001",
                "page_number": 1,
                "bbox": [100, 120, 300, 160],
            }
        ],
        "risks": [
            {
                "risk_type": "MISSING_REQUIRED_FIELD",
                "severity": "medium",
                "evidence": "Due date is missing",
                "label_source": "synthetic_ground_truth",
            }
        ],
        "annotation_version": "0.1",
    }


def test_valid_invoice_is_accepted(valid_document_data):
    document = DocumentRecord.model_validate(valid_document_data)

    assert document.document_id == "INV-0001"
    assert document.document_type == "invoice"
    assert len(document.pages) == 1
    assert len(document.entities) == 1
    assert len(document.risks) == 1


def test_valid_contract_is_accepted(valid_document_data):
    valid_document_data["document_type"] = "contract"

    document = DocumentRecord.model_validate(valid_document_data)

    assert document.document_type == "contract"


def test_invalid_document_type_is_rejected(valid_document_data):
    valid_document_data["document_type"] = "receipt"

    with pytest.raises(ValidationError):
        DocumentRecord.model_validate(valid_document_data)


def test_negative_page_number_is_rejected():
    with pytest.raises(ValidationError):
        PageMetadata(
            page_number=0,
            image_path="page.png",
            width=1000,
            height=1400,
        )


def test_invalid_bounding_box_is_rejected():
    with pytest.raises(ValidationError):
        EntityAnnotation(
            label="INVOICE_NUMBER",
            text="INV-0001",
            page_number=1,
            bbox=(200, 100, 100, 150),
        )


def test_entity_referencing_missing_page_is_rejected(valid_document_data):
    valid_document_data["entities"][0]["page_number"] = 2

    with pytest.raises(ValidationError):
        DocumentRecord.model_validate(valid_document_data)


def test_bounding_box_outside_page_is_rejected(valid_document_data):
    valid_document_data["entities"][0]["bbox"] = [900, 100, 1100, 150]

    with pytest.raises(ValidationError):
        DocumentRecord.model_validate(valid_document_data)


def test_duplicate_page_numbers_are_rejected(valid_document_data):
    valid_document_data["pages"].append(
        {
            "page_number": 1,
            "image_path": "another-page.png",
            "width": 1000,
            "height": 1400,
        }
    )

    with pytest.raises(ValidationError):
        DocumentRecord.model_validate(valid_document_data)


def test_invalid_risk_severity_is_rejected():
    with pytest.raises(ValidationError):
        RiskAnnotation(
            risk_type="MISSING_REQUIRED_FIELD",
            severity="urgent",
            evidence="Due date is missing",
            label_source="human",
        )


def test_unknown_document_field_is_rejected(valid_document_data):
    valid_document_data["unexpected_field"] = "not allowed"

    with pytest.raises(ValidationError):
        DocumentRecord.model_validate(valid_document_data)


def test_source_requires_dataset_and_license():
    with pytest.raises(ValidationError):
        DocumentSource(dataset="", license="internal_synthetic")
