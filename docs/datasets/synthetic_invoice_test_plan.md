# Synthetic Invoice Test Plan

## Schema and integrity

- [ ] Valid invoice passes validation.
- [ ] Missing required fields are handled consistently.
- [ ] Invalid calendar dates are rejected or labelled correctly.
- [ ] Invalid currency codes are detected.
- [ ] Negative monetary values are detected.
- [ ] Duplicate document IDs are rejected.

## Financial consistency

- [ ] Line-item totals reconcile with subtotal.
- [ ] Subtotal plus tax reconciles with total.
- [ ] Decimal rounding follows the documented currency policy.
- [ ] Missing monetary values do not silently become zero.

## Ground truth and risk rules

- [ ] Each single-defect example triggers its intended risk label.
- [ ] Valid invoices produce no risk labels.
- [ ] Risk labels contain a rule ID and evidence.
- [ ] Multi-defect examples produce the complete expected label set.
- [ ] Unrelated risk rules do not trigger unexpectedly.

## Reproducibility and leakage

- [ ] A fixed seed produces reproducible output.
- [ ] Variants preserve their template-family ID.
- [ ] Template families do not cross dataset splits.
- [ ] Source and schema versions are recorded.

## Acceptance gate

Do not generate the full dataset until schema, arithmetic, risk-rule, and reproducibility tests pass.
