import pytest

from app.data.normalizers import (
    normalize_document_type,
    normalize_record,
    normalize_severity,
)


def test_document_type_aliases():
    assert normalize_document_type(" INVOICES ") == "invoice"
    assert normalize_document_type("CONTRACTS") == "contract"


def test_unknown_document_type_is_rejected():
    with pytest.raises(ValueError, match="Unsupported document type"):
        normalize_document_type("receipt")


def test_severity_aliases():
    assert normalize_severity("MODERATE") == "medium"
    assert normalize_severity(" Critical ") == "critical"


def test_unknown_severity_is_rejected():
    with pytest.raises(ValueError, match="Unsupported severity"):
        normalize_severity("urgent")


def test_normalize_record_does_not_modify_input():
    original = {
        "document_id": "INV-001",
        "document_type": "INVOICES",
        "risks": [
            {
                "risk_type": "MISSING_REQUIRED_FIELD",
                "severity": "MODERATE",
                "evidence": "Due date missing",
                "label_source": "human",
            }
        ],
    }

    normalized = normalize_record(original)

    assert normalized["document_type"] == "invoice"
    assert normalized["risks"][0]["severity"] == "medium"
    assert original["document_type"] == "INVOICES"
    assert original["risks"][0]["severity"] == "MODERATE"


def test_normalize_record_preserves_unknown_fields():
    record = {
        "document_id": "INV-001",
        "custom_metadata": {"source_version": 2},
    }

    normalized = normalize_record(record)

    assert normalized["custom_metadata"] == {"source_version": 2}
