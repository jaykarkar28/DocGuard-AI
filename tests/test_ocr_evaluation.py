
import math

import pytest

from app.ocr.evaluation import (
    edit_distance,
    error_rate,
    evaluate_document,
    extract_entity_text,
    summarize_rows,
)


def test_edit_distance_counts_substitution_insertion_and_deletion():
    assert edit_distance(list("cat"), list("cut")) == 1
    assert edit_distance(list("cat"), list("cats")) == 1
    assert edit_distance(list("cats"), list("cat")) == 1


def test_cer_is_character_based():
    assert error_rate("cat", "cut", unit="character") == pytest.approx(
        1 / 3
    )


def test_wer_is_word_based():
    assert error_rate(
        "invoice total due", "invoice amount due", unit="word"
    ) == pytest.approx(1 / 3)


def test_whitespace_is_normalized_but_punctuation_is_preserved():
    assert error_rate(
        "INV-001", "INV-001", unit="character"
    ) == 0.0
    assert error_rate(
        "INV-001", "INV001", unit="character"
    ) > 0.0


def test_empty_reference_is_handled_explicitly():
    assert error_rate("", "", unit="character") == 0.0
    assert math.isinf(error_rate("", "unexpected", unit="word"))


def test_invalid_error_rate_unit_is_rejected():
    with pytest.raises(ValueError, match="unit"):
        error_rate("a", "b", unit="sentence")


def test_extract_entity_text_selects_words_inside_entity_box():
    entity = {"bbox": [100, 100, 220, 130]}
    words = [
        {"text": "Invoice", "bbox": [105, 105, 155, 120]},
        {"text": "001", "bbox": [160, 105, 190, 120]},
        {"text": "Outside", "bbox": [300, 105, 350, 120]},
    ]

    text, count = extract_entity_text(entity, words)

    assert text == "Invoice 001"
    assert count == 2


def test_extract_entity_text_rejects_invalid_entity_box():
    with pytest.raises(ValueError, match="bounding box"):
        extract_entity_text({"bbox": [10, 10, 10, 20]}, [])


def test_negative_padding_is_rejected():
    with pytest.raises(ValueError, match="padding"):
        extract_entity_text({"bbox": [0, 0, 10, 10]}, [], padding=-1)


def test_evaluate_document_reports_exact_and_failed_fields():
    annotation = {
        "document_id": "doc-1",
        "template_id": "classic",
        "entities": [
            {
                "label": "INVOICE_NUMBER",
                "text": "INV-001",
                "bbox": [100, 100, 200, 130],
            },
            {
                "label": "TOTAL_AMOUNT",
                "text": "100.00",
                "bbox": [300, 100, 380, 130],
            },
        ],
    }
    ocr_result = {
        "document_id": "doc-1",
        "words": [
            {"text": "INV-001", "bbox": [105, 105, 180, 120]},
            {"text": "100.0O", "bbox": [305, 105, 365, 120]},
        ],
    }

    rows = evaluate_document(annotation, ocr_result)

    assert len(rows) == 2
    assert rows[0]["exact_match"] is True
    assert rows[1]["exact_match"] is False
    assert rows[1]["detected"] is True


def test_no_ocr_tokens_means_not_detected_and_not_exact():
    annotation = {
        "document_id": "doc-2",
        "template_id": "minimal",
        "entities": [
            {
                "label": "TOTAL_AMOUNT",
                "text": "100.00",
                "bbox": [300, 100, 380, 130],
            }
        ],
    }

    rows = evaluate_document(
        annotation,
        {"document_id": "doc-2", "words": []},
    )

    assert rows[0]["detected"] is False
    assert rows[0]["prediction"] == ""
    assert rows[0]["exact_match"] is False


def test_document_id_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Document ID mismatch"):
        evaluate_document(
            {"document_id": "doc-a", "entities": []},
            {"document_id": "doc-b", "words": []},
        )


def test_summary_uses_micro_averaged_cer_and_wer():
    rows = [
        {
            "label": "INVOICE_NUMBER",
            "template_id": "classic",
            "reference": "abc",
            "prediction": "abc",
            "detected": True,
            "exact_match": True,
        },
        {
            "label": "TOTAL_AMOUNT",
            "template_id": "classic",
            "reference": "xy",
            "prediction": "xz",
            "detected": True,
            "exact_match": False,
        },
    ]

    summary = summarize_rows(rows)

    assert summary["field_count"] == 2
    assert summary["detected_fields"] == 2
    assert summary["field_exact_match_accuracy"] == pytest.approx(0.5)
    assert summary["cer"] == pytest.approx(1 / 5)
    assert summary["wer"] == pytest.approx(0.5)
    assert summary["by_template"]["classic"]["field_count"] == 2


def test_empty_summary_has_null_rates():
    summary = summarize_rows([])

    assert summary["field_count"] == 0
    assert summary["field_detection_coverage"] is None
    assert summary["field_exact_match_accuracy"] is None
    assert summary["cer"] is None
    assert summary["wer"] is None


def test_extract_entity_text_preserves_tesseract_reading_order():
    from app.ocr.evaluation import extract_entity_text

    entity = {
        "bbox": [0, 0, 200, 50],
        "text": "Consulting services",
    }
    words = [
        {
            "text": "Consulting",
            "bbox": [10, 10, 80, 25],
            "block_num": 1,
            "par_num": 1,
            "line_num": 1,
            "word_num": 1,
        },
        {
            "text": "services",
            "bbox": [90, 5, 150, 20],
            "block_num": 1,
            "par_num": 1,
            "line_num": 1,
            "word_num": 2,
        },
    ]

    prediction, count = extract_entity_text(entity, words, padding=0)

    assert prediction == "Consulting services"
    assert count == 2



def test_extract_entity_text_overlap_fallback_recovers_wide_quantity_box():
    entity = {
        "label": "LINE_ITEM_QUANTITY",
        "text": "10",
        "bbox": [700, 480, 724, 495],
    }
    words = [
        {
            "text": "10",
            "bbox": [542, 471, 744, 506],
            "block_num": 1,
            "par_num": 1,
            "line_num": 9,
            "word_num": 2,
        }
    ]

    prediction, count = extract_entity_text(entity, words, padding=8)

    assert prediction == "10"
    assert count == 1


def test_extract_entity_text_does_not_select_word_with_small_overlap():
    entity = {
        "bbox": [100, 100, 120, 120],
    }
    words = [
        {
            "text": "unrelated",
            "bbox": [118, 118, 150, 150],
        }
    ]

    prediction, count = extract_entity_text(entity, words, padding=0)

    assert prediction == ""
    assert count == 0
