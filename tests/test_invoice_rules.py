
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





from copy import deepcopy
from decimal import Decimal

from app.data.synthetic_invoices import generate_valid_invoices
from app.risk.invoice_rules import evaluate_invoice


def make_invoice():
    return deepcopy(generate_valid_invoices(1, seed=42)[0]["invoice"])


def risk_types(invoice):
    return {risk["risk_type"] for risk in evaluate_invoice(invoice)}


def test_missing_due_date_is_detected():
    invoice = make_invoice()
    invoice["due_date"] = None

    assert "INV-003" in risk_types(invoice)


def test_invalid_currency_is_detected_case_sensitively():
    invoice = make_invoice()
    invoice["currency"] = "usd"

    assert "INV-007" in risk_types(invoice)


def test_missing_subtotal_does_not_pass_silently():
    invoice = make_invoice()
    invoice["subtotal"] = None

    assert "INV-009" in risk_types(invoice)


def test_malformed_line_item_is_detected():
    invoice = make_invoice()
    invoice["line_items"][0]["unit_price"] = "not-a-number"

    assert "INV-009" in risk_types(invoice)


def test_negative_line_item_quantity_is_detected():
    invoice = make_invoice()
    invoice["line_items"][0]["quantity"] = -2

    assert "INV-009" in risk_types(invoice)


def test_non_list_line_items_are_detected():
    invoice = make_invoice()
    invoice["line_items"] = "invalid"

    assert "INV-009" in risk_types(invoice)


def test_risk_annotations_have_required_fields():
    invoice = make_invoice()
    invoice["total_amount"] = "1.00"

    risks = evaluate_invoice(invoice)

    assert risks
    for risk in risks:
        assert risk["risk_type"]
        assert risk["severity"] in {"low", "medium", "high", "critical"}
        assert risk["evidence"]
        assert risk["label_source"] == "rule"


def test_valid_invoice_has_no_risks():
    invoice = make_invoice()

    assert evaluate_invoice(invoice) == []



from app.data.synthetic_invoices import generate_valid_invoices
from app.risk.invoice_rules import evaluate_invoice


def test_non_dictionary_invoice_does_not_crash():
    assert any(
        risk["risk_type"] == "INV-009"
        for risk in evaluate_invoice(None)
    )


def test_nan_subtotal_is_rejected():
    invoice = generate_valid_invoices(1)[0]["invoice"].copy()
    invoice["subtotal"] = "NaN"

    risks = evaluate_invoice(invoice)

    assert any(
        risk["risk_type"] == "INV-008"
        and "subtotal" in risk["evidence"]
        for risk in risks
    )


def test_infinite_total_is_rejected():
    invoice = generate_valid_invoices(1)[0]["invoice"].copy()
    invoice["total_amount"] = "Infinity"

    risks = evaluate_invoice(invoice)

    assert any(
        risk["risk_type"] == "INV-008"
        and "total_amount" in risk["evidence"]
        for risk in risks
    )


def test_whitespace_required_field_is_detected():
    invoice = generate_valid_invoices(1)[0]["invoice"].copy()
    invoice["supplier_name"] = "   "

    risks = evaluate_invoice(invoice)

    assert any(
        risk["risk_type"] == "INV-005"
        for risk in risks
    )


def test_valid_invoice_still_has_no_risks():
    invoice = generate_valid_invoices(1)[0]["invoice"]

    assert evaluate_invoice(invoice) == []
