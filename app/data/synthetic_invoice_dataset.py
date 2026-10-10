
"""Build synthetic invoices with independently defined defect labels."""

from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Callable

from app.data.synthetic_invoices import generate_valid_invoices


DEFECTS: dict[str, dict[str, str]] = {
    "INV-001": {
        "severity": "high",
        "description": "Missing invoice number",
    },
    "INV-002": {
        "severity": "high",
        "description": "Invalid invoice date",
    },
    "INV-003": {
        "severity": "high",
        "description": "Missing due date",
    },
    "INV-004": {
        "severity": "medium",
        "description": "Due date precedes invoice date",
    },
    "INV-005": {
        "severity": "high",
        "description": "Missing supplier name",
    },
    "INV-006": {
        "severity": "high",
        "description": "Missing customer name",
    },
    "INV-007": {
        "severity": "medium",
        "description": "Unsupported currency",
    },
    "INV-008": {
        "severity": "high",
        "description": "Negative total amount",
    },
    "INV-009": {
        "severity": "high",
        "description": "Missing subtotal",
    },
    "INV-010": {
        "severity": "medium",
        "description": "Incorrect invoice total",
    },
}


def _remove_invoice_number(invoice: dict[str, Any]) -> None:
    invoice["invoice_number"] = None


def _invalidate_invoice_date(invoice: dict[str, Any]) -> None:
    invoice["invoice_date"] = "not-a-valid-date"


def _remove_due_date(invoice: dict[str, Any]) -> None:
    invoice["due_date"] = None


def _reverse_dates(invoice: dict[str, Any]) -> None:
    invoice_date = date.fromisoformat(invoice["invoice_date"])
    invoice["due_date"] = (
        invoice_date - timedelta(days=1)
    ).isoformat()


def _remove_supplier(invoice: dict[str, Any]) -> None:
    invoice["supplier_name"] = None


def _remove_customer(invoice: dict[str, Any]) -> None:
    invoice["customer_name"] = None


def _invalidate_currency(invoice: dict[str, Any]) -> None:
    invoice["currency"] = "XYZ"


def _make_total_negative(invoice: dict[str, Any]) -> None:
    invoice["total_amount"] = "-1.00"


def _remove_subtotal(invoice: dict[str, Any]) -> None:
    invoice["subtotal"] = None


def _change_total(invoice: dict[str, Any]) -> None:
    total = Decimal(invoice["total_amount"])
    invoice["total_amount"] = str(
        total + Decimal("1.00")
    )


MUTATIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    "INV-001": _remove_invoice_number,
    "INV-002": _invalidate_invoice_date,
    "INV-003": _remove_due_date,
    "INV-004": _reverse_dates,
    "INV-005": _remove_supplier,
    "INV-006": _remove_customer,
    "INV-007": _invalidate_currency,
    "INV-008": _make_total_negative,
    "INV-009": _remove_subtotal,
    "INV-010": _change_total,
}


def _expected_risks_for_defect(
    rule_id: str,
) -> list[dict[str, str]]:
    """Return ground-truth labels from the mutation specification."""
    expected_ids = [rule_id]

    # A negative total also violates subtotal + tax reconciliation.
    if rule_id == "INV-008":
        expected_ids.append("INV-010")

    return [
        {
            "risk_type": expected_id,
            "severity": DEFECTS[expected_id]["severity"],
            "evidence": (
                "Synthetic defect injected: "
                f"{DEFECTS[expected_id]['description']}."
            ),
            "label_source": "synthetic_ground_truth",
        }
        for expected_id in expected_ids
    ]


def generate_invoice_dataset(
    base_count: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate valid invoices and controlled, labeled defect variants.

    Each base invoice produces:
    - One valid invoice.
    - Ten defective variants, one for each configured defect.

    All variants from the same base invoice share a split group to
    prevent related records from crossing train/validation/test splits.
    """
    if base_count < 1:
        raise ValueError("base_count must be at least 1")

    base_records = generate_valid_invoices(
        count=base_count,
        seed=seed,
    )
    dataset: list[dict[str, Any]] = []

    for base_record in base_records:
        base_id = base_record["document_id"]
        family_id = f"invoice-family-{base_id}"

        # Keep one untouched valid example for each base invoice.
        valid_record = deepcopy(base_record)
        valid_record["split_group_id"] = family_id
        valid_record["expected_risks"] = []
        dataset.append(valid_record)

        # Create one independently labeled variant per mutation.
        for rule_id, mutation in MUTATIONS.items():
            variant = deepcopy(base_record)
            variant["document_id"] = f"{base_id}-{rule_id}"
            variant["split_group_id"] = family_id

            mutation(variant["invoice"])

            variant["expected_risks"] = (
                _expected_risks_for_defect(rule_id)
            )

            dataset.append(variant)

    return dataset
