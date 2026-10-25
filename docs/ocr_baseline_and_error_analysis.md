# DocGuard AI — OCR Baseline and Error Analysis

## 1. Objective

Evaluate a lightweight OCR baseline for rendered invoice documents before integrating OCR into the wider document intelligence pipeline.

The experiment evaluates Tesseract OCR, compares page segmentation configurations, investigates field-level errors, and tests targeted OCR on due-date crops.

The goal is to establish a reproducible baseline and document limitations—not to claim production accuracy on real-world invoices.

## 2. Environment

- Operating system: Windows
- Python: 3.12.10
- OCR engine: Tesseract OCR 5.5.3
- Python wrapper: `pytesseract`
- Image processing: Pillow
- Primary document domain: invoices
- Dataset: controlled synthetic invoices
- Hardware constraint: resource-conscious execution on a machine without a dedicated GPU

Tesseract is configured through the project's OCR wrapper. The executable path can be overridden using `DOCGUARD_TESSERACT_CMD`.

## 3. Dataset and evaluation scope

The rendered synthetic invoice dataset contains:

- 330 invoice images
- 30 base invoice groups, each with a valid invoice and ten defective variants
- Three templates: `classic`, `minimal`, and `sidebar`
- 110 images per template
- Image dimensions of 1275 × 1650 pixels

Related variants from each base invoice remain within the same data split to prevent variants of the same base document from crossing splits.

The OCR benchmark covered all 330 rendered images across the dataset's train, validation, and test partitions. These results are descriptive measurements on the synthetic dataset, not a held-out-only performance estimate.

Field-level evaluation compares OCR text assigned geometrically to annotated entity boxes. It evaluates present annotated entity values; static printed labels and absent fields are excluded.

## 4. Whole-page OCR baseline

The primary comparison used Tesseract's default OCR engine mode with two page segmentation modes:

- PSM 6: assume a uniform block of text
- PSM 4: assume a single column of text with variable-sized text blocks

### Results across 330 images

| Metric | PSM 6 | PSM 4 |
|---|---:|---:|
| Wall-clock runtime | 170.563 s | 174.087 s |
| Mean runtime per image | 0.484812 s | 0.496516 s |
| Field exact-match accuracy | 98.6074% | 99.4529% |
| Character error rate (CER) | 0.2293% | 0.0764% |
| Word error rate (WER) | 1.4002% | 0.4575% |
| Field detection coverage | 100% | 100% |
| Evaluated fields | 6032 | 6032 |
| Exact field matches | — | 5999 |
| Incorrect fields | — | 33 |

PSM 4 improved field exact-match accuracy by approximately 0.85 percentage points and reduced CER and WER substantially, with a small increase in mean runtime.

**Selected whole-page baseline: `--oem 3 --psm 4`.**

These measurements apply to the current synthetic invoice dataset and annotation-based evaluation method. They should not be interpreted as expected accuracy on arbitrary scanned or photographed invoices.

## 5. Error analysis

The PSM 4 whole-page benchmark produced 33 field mismatches across 6032 evaluated fields.

| Field label | Incorrect fields |
|---|---:|
| `INVOICE_NUMBER` | 29 |
| `DUE_DATE` | 4 |
| Other evaluated labels | 0 |

### 5.1 Invoice-number punctuation

The 29 invoice-number mismatches came from variants of three base invoices:

- `SYN-INV-00004`: OCR returned `INV-00004.`
- `SYN-INV-00014`: OCR returned `INV-00014.`
- `SYN-INV-00024`: OCR returned `INV-00024.`

The visible invoice numbers match the references. The additional terminal period is consistent with an OCR punctuation artifact.

The same artifact repeats across defective variants of each base invoice. These errors are therefore not 29 independent examples of visual recognition difficulty.

**Planned handling:** preserve the raw OCR output and apply conservative normalization only to extracted `INVOICE_NUMBER` values. Do not remove punctuation indiscriminately from arbitrary document text.

### 5.2 Sidebar due dates

Four due-date mismatches were found, all in the sidebar template.

| Document | Reference | Whole-page OCR |
|---|---|---|
| `SYN-INV-00010-INV-005` | `2026-04-26` | `3026.04.26` |
| `SYN-INV-00016-INV-005` | `2026-05-28` | `3026-05-28` |
| `SYN-INV-00025-INV-005` | `2026-06-25` | `3026-08-25` |
| `SYN-INV-00028-INV-005` | `2026-06-30` | `3026-08-30` |

Visual inspection showed that the rendered dates matched their reference annotations. The errors involved character and separator recognition, with lower OCR confidence around the affected sidebar fields.

The findings suggested evaluating OCR on a focused crop of the due-date value rather than relying exclusively on whole-page recognition.

No hard-coded replacement of `3026` with `2026`, or `08` with `06`, is justified. Such corrections could conceal genuine recognition errors in other documents.

## 6. Targeted due-date crop experiment

A controlled experiment evaluated all 100 annotated sidebar due-date fields using the same crop-generation procedure and four Tesseract configurations.

Each crop included padding around the annotated field. Predictions were evaluated using exact-match comparison after whitespace normalization only. Punctuation and digit errors were not ignored.

| Configuration | Exact matches | Accuracy | Mean runtime per field |
|---|---:|---:|---:|
| Crop + PSM 4 | 100/100 | 100.00% | 0.2147 s |
| Crop + PSM 7 | 100/100 | 100.00% | 0.1781 s |
| Grayscale/autocontrast + PSM 7 | 100/100 | 100.00% | 0.1905 s |
| 2× upscale + grayscale + PSM 7 | 100/100 | 100.00% | 0.1871 s |

PSM 7 treats the crop as a single text line. It achieved the same observed accuracy as the other tested configurations and had the lowest mean runtime.

**Selected candidate for sidebar due-date crops: `--oem 3 --psm 7`, without additional preprocessing.**

This result supports a targeted crop-level OCR strategy for the current synthetic sidebar dates. It does not establish universal date recognition accuracy, and it does not prove that crop-level OCR will improve every real-world invoice.

The experiment should be reproducible using:

`python scripts/experiment_due_date_crops.py`

Its output is stored under:

`datasets/processed/ocr/due_date_crop_experiment/`

The experiment's detailed JSON results should be retained as generated evidence, rather than manually editing reported values.

## 7. Invoice-number normalization

A separate normalization module provides field-specific processing:

- `normalize_invoice_number(value)` removes terminal `.`, `,`, `;`, or `:` punctuation and surrounding whitespace.
- `normalize_field_value(label, value)` applies that behavior only when the field label is `INVOICE_NUMBER`.
- Other field values are returned unchanged.
- Misread letters and digits are not guessed or corrected.
- Raw OCR predictions remain available for audit and evaluation.

Unit tests cover terminal punctuation, internal punctuation, whitespace, unchanged non-invoice-number fields, invalid argument types, and preservation of misread characters.

Normalization is not a substitute for OCR evaluation. Raw and normalized metrics should be reported separately if normalization is integrated into the extraction pipeline.

## 8. Limitations

1. **Synthetic-data scope:** the images are generated from controlled templates and do not represent the full variation of real invoices.
2. **Related examples:** defective variants share a base invoice. The 330 images are not 330 independent document designs.
3. **Annotation-based evaluation:** entity matching relies on annotated bounding boxes and a geometric assignment heuristic. It may misassign words when boxes overlap or word boxes span multiple fields.
4. **Partial text coverage:** field metrics cover annotated entity values, not all static printed labels or absent fields.
5. **Crop experiment scope:** the 100 sidebar due-date fields demonstrate performance on this dataset only. The same result should not be assumed for other templates, OCR engines, image quality levels, or document sources.
6. **No automatic date repair:** OCR output that resembles an invalid or unexpected date must be validated downstream rather than silently rewritten.
7. **Runtime variability:** timings depend on the local machine, process state, and image processing overhead.

## 9. Decisions and follow-up

### Decisions

- Retain Tesseract as the lightweight OCR baseline.
- Use PSM 4 for the current whole-page benchmark.
- Keep crop + PSM 7 as the candidate configuration for sidebar due-date extraction.
- Preserve raw OCR output.
- Apply invoice-number punctuation normalization only to the corresponding extracted field.
- Keep date validation and compliance checks separate from OCR recognition.

### Follow-up work

1. Integrate field-level OCR as a separate extraction step without changing the raw whole-page result.
2. Add tests that compare raw and normalized invoice-number values.
3. Evaluate the integrated extraction pipeline on all templates and inspect regressions.
4. Retain the existing synthetic dataset limitations in project documentation.
5. Continue to the next DocGuard AI roadmap phase only after testing and evaluation are complete.

## 10. Reproduction commands

Run from the project root with the virtual environment activated.

```powershell
python scripts/benchmark_ocr_full.py
python scripts/benchmark_ocr_variants.py
python scripts/experiment_due_date_crops.py
python -m pytest -v
```

Some benchmark scripts can take several minutes because OCR is run across many images.

## 11. Current validation status

The reported test run completed with **179 tests passed**.

The pytest output includes a warning that pytest configuration in `pyproject.toml` is being ignored in favor of `pytest.ini`. It did not prevent this test run from passing; the configuration discrepancy should be reviewed separately.

The OCR experiments and their test suite provide a documented baseline for subsequent DocGuard AI extraction, classification, and risk-analysis work.
