
from copy import deepcopy
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.data.invoice_schemas import InvoicePayload
from app.data.synthetic_invoices import generate_valid_invoices
from app.risk.invoice_rules import evaluate_invoice


def make_invoice():
    return deepcopy(generate_valid_invoices(1, seed=42)[0]["invoice"])


def test_valid_generated_invoice_passes_schema():
    invoice = InvoicePayload.model_validate(make_invoice())

    assert invoice.invoice_number
    assert invoice.total_amount is not None


def test_invalid_currency_fails_schema():
    payload = make_invoice()
    payload["currency"] = "XYZ"

    with pytest.raises(ValidationError):
        InvoicePayload.model_validate(payload)


def test_negative_amount_fails_schema():
    payload = make_invoice()
    payload["tax_amount"] = "-10.00"

    with pytest.raises(ValidationError):
        InvoicePayload.model_validate(payload)


def test_inconsistent_line_total_fails_schema():
    payload = make_invoice()
    payload["line_items"][0]["line_total"] = "0.01"

    with pytest.raises(ValidationError):
        InvoicePayload.model_validate(payload)


def test_inconsistent_subtotal_fails_schema():
    payload = make_invoice()
    payload["subtotal"] = "0.01"

    with pytest.raises(ValidationError):
        InvoicePayload.model_validate(payload)


def test_total_mismatch_fails_schema():
    payload = make_invoice()
    payload["total_amount"] = "1.00"

    with pytest.raises(ValidationError):
        InvoicePayload.model_validate(payload)


def test_due_date_before_invoice_date_fails_schema():
    payload = make_invoice()
    payload["invoice_date"] = "2026-03-15"
    payload["due_date"] = "2026-03-14"

    with pytest.raises(ValidationError):
        InvoicePayload.model_validate(payload)


def test_missing_subtotal_is_preserved_for_risk_detection():
    payload = make_invoice()
    payload["subtotal"] = None

    # Missing data can be represented by the schema so that risk rules
    # can identify it instead of losing the example during validation.
    invoice = InvoicePayload.model_validate(payload)
    assert invoice.subtotal is None

    risks = evaluate_invoice(payload)
    assert any(r["risk_type"] == "INV-009" for r in risks)


def test_missing_required_key_fails_schema():
    payload = make_invoice()
    del payload["invoice_number"]

    with pytest.raises(ValidationError):
        InvoicePayload.model_validate(payload)


def test_decimal_values_are_parsed_exactly():
    invoice = InvoicePayload.model_validate(make_invoice())

    assert isinstance(invoice.subtotal, Decimal)
    assert invoice.subtotal >= Decimal("0.00")
