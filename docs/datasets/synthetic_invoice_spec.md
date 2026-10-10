# DocGuard AI — Synthetic Invoice Specification

**Version:** 0.1  
**Status:** Design  
**Scope:** Synthetic invoice records for data-pipeline and ground-truth validation

## 1. Objectives

Create reproducible synthetic invoices for structured field extraction, document classification, deterministic defect detection, and future OCR/layout experiments.

This initial specification covers structured records only. It does not claim OCR or image-extraction performance.

## 2. Required invoice fields

| Field | Type | Validation |
|---|---|---|
| invoice_number | String or null | Required and non-empty |
| invoice_date | ISO date or null | Required and valid |
| due_date | ISO date or null | Required for this pilot; must not precede invoice date |
| supplier_name | String or null | Required and non-empty |
| customer_name | String or null | Required and non-empty |
| currency | String or null | Accepted ISO 4217 currency code |
| subtotal | Decimal string or null | Non-negative |
| tax_amount | Decimal string or null | Non-negative |
| total_amount | Decimal string or null | Non-negative |
| line_items | Array | Description, quantity, unit price and line total |

Use decimal arithmetic for monetary calculations, not binary floating-point arithmetic.

## 3. Initial defect taxonomy

| Risk ID | Defect | Expected rule |
|---|---|---|
| INV-001 | Missing invoice number | Number is absent or empty |
| INV-002 | Missing invoice date | Date is absent or empty |
| INV-003 | Missing due date | Due date is absent or empty |
| INV-004 | Invalid date order | Due date precedes invoice date |
| INV-005 | Missing supplier | Supplier is absent or empty |
| INV-006 | Missing customer | Customer is absent or empty |
| INV-007 | Invalid currency | Currency is not in the configured ISO 4217 allowlist |
| INV-008 | Negative amount | A monetary value is below zero |
| INV-009 | Subtotal mismatch | Subtotal differs from line-item totals beyond currency rounding tolerance |
| INV-010 | Total mismatch | Total differs from subtotal plus tax beyond currency rounding tolerance |

These rules identify defined data defects. They do not establish fraud, legal non-compliance, or financial loss.

## 4. Ground-truth policy

Each record must include a unique document ID, template-family ID, reproducible generation seed or run ID, document type, synthetic source metadata, schema version, expected field values, and expected risk labels with evidence.

Valid invoices must have an empty risk-label list. Intentionally missing fields must be distinguishable from accidental generation errors.

## 5. Generation rules

- Generate valid invoices first.
- Apply one controlled defect at a time.
- Add multi-defect cases only after individual rules pass.
- Keep fixed seeds for reproducibility.
- Keep variants from the same template family in one split.
- Never use real customer or financial information.

## 6. Validation requirements

- Schema validation succeeds for valid records.
- Document IDs are unique.
- Monetary arithmetic uses exact decimal values.
- Valid invoice totals reconcile within a documented rounding tolerance.
- Defective examples trigger their intended rules.
- Unrelated rules do not trigger unexpectedly.
- Repeated runs with the same seed produce equivalent records.
- Template families do not cross train, validation, and test splits.

## 7. Out of scope

OCR accuracy, bounding-box quality, fraud prediction, legal compliance decisions, and claims about real-world invoice performance require separate datasets and evaluation.
