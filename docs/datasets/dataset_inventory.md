# DocGuard AI — Dataset Inventory

**Status:** Research in progress  
**Last reviewed:** 2026-10-10  
**Project scope:** Invoice and contract document intelligence  
**Policy:** Do not download large datasets until suitability and acquisition decisions are documented.

## 1. Dataset selection criteria

Evaluate each source against:

- Relevance to invoices or contracts
- Availability of original images/PDFs and text
- Availability and granularity of annotations
- License and attribution obligations
- Dataset size and local/GPU resource requirements
- Suitability for classification, entity extraction, OCR, or layout understanding
- Ability to prevent train/validation/test leakage
- Reproducibility and provenance

## 2. Candidate inventory

| Dataset | Intended use | Annotation coverage | License / terms | Decision |
|---|---|---|---|---|
| CORD | Receipt-like field extraction pilot | Receipt images, text/boxes, semantic parsing labels | CC BY 4.0; preserve attribution | Pilot only |
| CUAD | Contract clause extraction pilot | 510 contracts; 41 clause types; expert annotations | CC BY 4.0; preserve attribution | Candidate for contract-text experiments |
| FUNSD | OCR/layout/form-understanding experiment | 199 scanned forms; word boxes, entities and relationships | Original dataset described as non-commercial research/educational use | Optional; verify intended use before acquisition |
| DocLayNet | General document layout segmentation | 80,863 pages; bounding-box layout annotations | Community Data License Agreement – Permissive 1.0 | Defer due to download size |
| Synthetic invoices | Invoice-specific field extraction and defect detection | Generated fields, boxes, and controlled risk labels | Original project-generated content; document generation method and dependencies | Required |
| Synthetic contracts | Contract-specific missing clauses and risk scenarios | Generated clauses, entities, and controlled risk labels | Original project-generated content; document generation method and dependencies | Required |

## 3. Official sources

- CORD: https://github.com/clovaai/cord
- CUAD: https://github.com/The-Atticus-Project/cuad
- CUAD legal information: https://www.atticusprojectai.org/legal/
- FUNSD: https://github.com/crcresearch/FUNSD
- DocLayNet: https://github.com/DS4SD/DocLayNet

Before acquisition, record the exact download URL, retrieval date, release/version or commit, license URL, checksum where practical, and any restrictions applying to the actual files.

## 4. Intended model tasks

1. Document classification: invoice vs. contract.
2. Invoice entity extraction: invoice number, dates, supplier, totals, tax, and currency.
3. Contract entity/clause extraction: parties, dates, renewal, termination, and selected clause types.
4. Layout-aware extraction: use image coordinates and OCR tokens where suitable annotations exist.
5. Risk detection: identify defined defects using deterministic rules and annotated ground truth.

Do not treat receipt fields as equivalent to invoice fields. Do not treat clause extraction accuracy as proof that a model can detect legal risk correctly.

## 5. Synthetic-data policy

- Generate valid and intentionally defective examples from documented templates.
- Keep generation parameters and random seeds reproducible.
- Store expected field values and risk labels separately from model outputs.
- Include defect categories such as missing invoice number, invalid date order, inconsistent totals, missing required clause, and ambiguous renewal language.
- Do not create contradictory ground truth.
- Keep all variants derived from the same template family in the same data split.
- Clearly label synthetic data in every record and report.

## 6. Split and leakage policy

- Preserve official benchmark splits when using them for benchmark evaluation.
- For custom data, group by source document, contract family, or synthetic template family before splitting.
- Never split near-duplicate pages or generated variants across train, validation, and test.
- Record split assignments in versioned JSONL artifacts.
- Do not use test-set results to tune training or thresholds.

## 7. Acquisition gate

A dataset may be acquired only after recording:

- [ ] Intended task and limitations
- [ ] Exact source and version
- [ ] License and attribution requirements
- [ ] Download size and storage requirements
- [ ] Annotation format and mapping plan
- [ ] Provenance fields
- [ ] Split/leakage strategy
- [ ] Validation procedure

## 8. Current decision

Start with a small, controlled pilot. Use CORD for receipt-like parsing experiments, evaluate CUAD for contract-text extraction, and create synthetic invoice examples for the project's actual invoice requirements. Defer DocLayNet until a concrete layout experiment justifies its storage cost. Keep FUNSD optional and respect its stated usage restrictions.

**Research limitation:** This inventory is an engineering decision record, not legal advice. Re-check the exact dataset terms before downloading, redistributing, or using any data in a public portfolio.



# Synthetic Invoice Test Plan

## Schema and data integrity

- [ ] Valid invoice passes validation.
- [ ] Missing required field is represented consistently.
- [ ] Invalid dates are rejected or labelled correctly.
- [ ] Unsupported currency code is handled correctly.
- [ ] Negative monetary values are detected.
- [ ] Duplicate document IDs are rejected.

## Financial consistency

- [ ] Line-item totals reconcile with subtotal.
- [ ] Subtotal plus tax reconciles with total.
- [ ] Decimal rounding follows the documented currency policy.
- [ ] Missing monetary values do not silently become zero.

## Ground truth and risk rules

- [ ] Every single-defect example triggers its intended risk label.
- [ ] Valid invoices produce no risk labels.
- [ ] Each risk label includes evidence and a rule identifier.
- [ ] Multiple-defect examples produce the complete expected label set.
- [ ] No unexpected risk labels appear.

## Reproducibility and leakage

- [ ] Fixed seed produces reproducible output.
- [ ] Generated variants preserve their template-family ID.
- [ ] Template families do not cross dataset splits.
- [ ] Source and schema versions are recorded.

## Acceptance gate

Do not generate the full dataset until all initial schema, arithmetic, and risk-rule tests pass.

