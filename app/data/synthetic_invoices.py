
"""Reproducible synthetic invoice generation for DocGuard AI."""

from __future__ import annotations

import random
from decimal import Decimal
from datetime import date, timedelta
from typing import Any


SCHEMA_VERSION = "0.1"

# Small, explicit allowlist for the initial pilot.
# Expand from a maintained ISO 4217 source before broader use.
SUPPORTED_CURRENCIES = {"USD", "INR", "EUR", "GBP"}


def _money(value: Decimal) -> str:
    """Return a two-decimal-place monetary string."""
    return str(value.quantize(Decimal("0.01")))


def generate_valid_invoices(
    count: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate valid structured invoice records reproducibly."""
    if count < 1:
        raise ValueError("count must be at least 1")

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    suppliers = [
        "Northstar Supplies",
        "Blue River Services",
        "Summit Office Goods",
        "Greenfield Systems",
        "Silverline Trading",
    ]
    customers = [
        "Aster Technologies",
        "Cedar Works",
        "Pioneer Analytics",
        "Brightpath Industries",
        "Maple Consulting",
    ]
    descriptions = [
        "Office supplies",
        "Software services",
        "Equipment",
        "Consulting services",
        "Maintenance",
    ]

    base_date = date(2026, 1, 1)

    for index in range(count):
        invoice_date = base_date + timedelta(days=rng.randint(0, 300))
        due_date = invoice_date + timedelta(days=rng.choice([15, 30, 45]))
        item_count = rng.randint(1, 4)

        line_items = []
        subtotal = Decimal("0.00")

        for item_index in range(item_count):
            quantity = rng.randint(1, 10)
            unit_price = Decimal(rng.randint(100, 50000)) / Decimal("100")
            line_total = (unit_price * quantity).quantize(
                Decimal("0.01")
            )
            subtotal += line_total

            line_items.append(
                {
                    "description": rng.choice(descriptions),
                    "quantity": quantity,
                    "unit_price": _money(unit_price),
                    "line_total": _money(line_total),
                }
            )

        tax_amount = (subtotal * Decimal("0.18")).quantize(
            Decimal("0.01")
        )
        total_amount = subtotal + tax_amount
        currency = rng.choice(sorted(SUPPORTED_CURRENCIES))

        records.append(
            {
                "document_id": f"SYN-INV-{index + 1:05d}",
                "document_type": "invoice",
                "source": {
                    "dataset": "synthetic",
                    "license": "CC0-1.0",
                    "source_document_id": None,
                },
                "split_group_id": f"invoice-template-{index % 5:02d}",
                "schema_version": SCHEMA_VERSION,
                "generation_seed": seed,
                "invoice": {
                    "invoice_number": f"INV-{index + 1:05d}",
                    "invoice_date": invoice_date.isoformat(),
                    "due_date": due_date.isoformat(),
                    "supplier_name": rng.choice(suppliers),
                    "customer_name": rng.choice(customers),
                    "currency": currency,
                    "subtotal": _money(subtotal),
                    "tax_amount": _money(tax_amount),
                    "total_amount": _money(total_amount),
                    "line_items": line_items,
                },
                "expected_risks": [],
            }
        )

    return records
