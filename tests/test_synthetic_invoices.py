
from copy import deepcopy

import pytest

from app.data.synthetic_invoices import generate_valid_invoices
from app.risk.invoice_rules import evaluate_invoice


def test_generation_is_reproducible():
    assert generate_valid_invoices(5, seed=42) == (
        generate_valid_invoices(5, seed=42)
    )


def test_generated_invoices_are_valid():
    records = generate_valid_invoices(10, seed=7)

    assert len(records) == 10
    assert len({record["document_id"] for record in records}) == 10

    for record in records:
        assert record["document_type"] == "invoice"
        assert record["expected_risks"] == []
        assert evaluate_invoice(record["invoice"]) == []


def test_zero_count_is_rejected():
    with pytest.raises(ValueError):
        generate_valid_invoices(0)


def test_missing_invoice_number_is_detected():
    invoice = deepcopy(generate_valid_invoices(1)[0]["invoice"])
    invoice["invoice_number"] = ""

    risks = evaluate_invoice(invoice)

    assert any(risk["risk_type"] == "INV-001" for risk in risks)


def test_due_date_before_invoice_date_is_detected():
    invoice = deepcopy(generate_valid_invoices(1)[0]["invoice"])
    invoice["due_date"] = "2020-01-01"
    invoice["invoice_date"] = "2026-01-01"

    risks = evaluate_invoice(invoice)

    assert any(risk["risk_type"] == "INV-004" for risk in risks)


def test_total_mismatch_is_detected():
    invoice = deepcopy(generate_valid_invoices(1)[0]["invoice"])
    invoice["total_amount"] = "1.00"

    risks = evaluate_invoice(invoice)

    assert any(risk["risk_type"] == "INV-010" for risk in risks)


def test_negative_amount_is_detected():
    invoice = deepcopy(generate_valid_invoices(1)[0]["invoice"])
    invoice["tax_amount"] = "-1.00"

    risks = evaluate_invoice(invoice)

    assert any(risk["risk_type"] == "INV-008" for risk in risks)


def test_invalid_currency_is_detected():
    invoice = deepcopy(generate_valid_invoices(1)[0]["invoice"])
    invoice["currency"] = "XYZ"

    risks = evaluate_invoice(invoice)

    assert any(risk["risk_type"] == "INV-007" for risk in risks)
