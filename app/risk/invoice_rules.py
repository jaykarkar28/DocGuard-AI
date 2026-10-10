
"""Deterministic validation rules for structured synthetic invoices."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from app.data.synthetic_invoices import SUPPORTED_CURRENCIES


def _parse_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def evaluate_invoice(invoice: dict[str, Any]) -> list[dict[str, str]]:
    """Return risk annotations for detectable invoice defects."""
    risks: list[dict[str, str]] = []

    def add(
        risk_type: str,
        severity: str,
        evidence: str,
    ) -> None:
        risks.append(
            {
                "risk_type": risk_type,
                "severity": severity,
                "evidence": evidence,
                "label_source": "rule",
            }
        )

    required_fields = {
        "invoice_number": "INV-001",
        "invoice_date": "INV-002",
        "due_date": "INV-003",
        "supplier_name": "INV-005",
        "customer_name": "INV-006",
    }

    for field, rule_id in required_fields.items():
        value = invoice.get(field)
        if value is None or (
            isinstance(value, str) and not value.strip()
        ):
            add(
                rule_id,
                "high",
                f"Required field '{field}' is missing or empty.",
            )

    parsed_dates: dict[str, date] = {}
    for field in ("invoice_date", "due_date"):
        value = invoice.get(field)
        if value is None or value == "":
            continue

        try:
            parsed_dates[field] = date.fromisoformat(str(value))
        except ValueError:
            add(
                f"{'INV-002' if field == 'invoice_date' else 'INV-003'}",
                "high",
                f"Field '{field}' is not a valid ISO calendar date.",
            )

    if (
        "invoice_date" in parsed_dates
        and "due_date" in parsed_dates
        and parsed_dates["due_date"] < parsed_dates["invoice_date"]
    ):
        add(
            "INV-004",
            "medium",
            "Due date precedes invoice date.",
        )

    currency = invoice.get("currency")
    if currency not in SUPPORTED_CURRENCIES:
        add(
            "INV-007",
            "medium",
            f"Currency '{currency}' is outside the configured pilot allowlist.",
        )

    monetary_fields = ("subtotal", "tax_amount", "total_amount")
    amounts: dict[str, Decimal] = {}

    for field in monetary_fields:
        value = invoice.get(field)
        amount = _parse_decimal(value)

        if amount is None:
            if value is not None and value != "":
                add(
                    "INV-008",
                    "high",
                    f"Field '{field}' is not a valid decimal amount.",
                )
            continue

        amounts[field] = amount
        if amount < 0:
            add(
                "INV-008",
                "high",
                f"Field '{field}' contains a negative amount.",
            )

    line_items = invoice.get("line_items", [])
    line_totals: list[Decimal] = []

    if not isinstance(line_items, list):
        add(
            "INV-009",
            "high",
            "Field 'line_items' must be a list.",
        )
        line_items = []

    for index, item in enumerate(line_items):
        if not isinstance(item, dict):
            add(
                "INV-009",
                "high",
                f"Line item {index} is not an object.",
            )
            continue

        quantity = _parse_decimal(item.get("quantity"))
        unit_price = _parse_decimal(item.get("unit_price"))
        line_total = _parse_decimal(item.get("line_total"))

        if (
            quantity is None
            or unit_price is None
            or line_total is None
            or quantity <= 0
            or unit_price < 0
            or line_total < 0
        ):
            add(
                "INV-009",
                "high",
                f"Line item {index} has invalid quantity or monetary values.",
            )
            continue

        expected_line_total = (quantity * unit_price).quantize(
            Decimal("0.01")
        )
        if expected_line_total != line_total:
            add(
                "INV-009",
                "medium",
                f"Line item {index} total does not match quantity × unit price.",
            )

        line_totals.append(line_total)

    if line_items and len(line_totals) == len(line_items):
        if "subtotal" in amounts:
            if sum(line_totals, Decimal("0.00")) != amounts["subtotal"]:
                add(
                    "INV-009",
                    "medium",
                    "Subtotal does not equal the sum of line-item totals.",
                )
    elif not line_items:
        add(
            "INV-009",
            "medium",
            "Invoice contains no line items.",
        )

    if "subtotal" in amounts and "tax_amount" in amounts:
        if "total_amount" in amounts:
            expected_total = amounts["subtotal"] + amounts["tax_amount"]
            if expected_total != amounts["total_amount"]:
                add(
                    "INV-010",
                    "medium",
                    "Total does not equal subtotal plus tax.",
                )

    return risks
