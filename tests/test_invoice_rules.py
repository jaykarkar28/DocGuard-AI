
from copy import deepcopy

from app.data.synthetic_invoices import generate_valid_invoices
from app.risk.invoice_rules import evaluate_invoice


def test_missing_supplier_is_detected():
    invoice = deepcopy(generate_valid_invoices(1)[0]["invoice"])
    invoice["supplier_name"] = None

    risks = evaluate_invoice(invoice)

    assert any(risk["risk_type"] == "INV-005" for risk in risks)


def test_subtotal_mismatch_is_detected():
    invoice = deepcopy(generate_valid_invoices(1)[0]["invoice"])
    invoice["subtotal"] = "0.01"

    risks = evaluate_invoice(invoice)

    assert any(risk["risk_type"] == "INV-009" for risk in risks)


def test_invalid_date_is_detected():
    invoice = deepcopy(generate_valid_invoices(1)[0]["invoice"])
    invoice["invoice_date"] = "2026-02-30"

    risks = evaluate_invoice(invoice)

    assert any(risk["risk_type"] == "INV-002" for risk in risks)
