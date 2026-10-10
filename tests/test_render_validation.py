
import json
from pathlib import Path

from scripts.validate_rendered_invoices import ROOT, validate


def test_rendered_dataset_passes_validation():
    report = validate()

    assert report["validation_passed"] is True
    assert report["errors"] == []
    assert report["annotation_count"] == 330
    assert report["unique_annotation_ids"] == 330
    assert report["missing_annotations"] == 0
    assert report["unexpected_annotations"] == 0
    assert report["missing_images"] == 0
    assert report["wrong_image_dimensions"] == 0
    assert report["image_hash_mismatches"] == 0
    assert report["invalid_entities_or_boxes"] == 0
    assert report["split_group_leakage"] == {}
    assert report["inconsistent_group_templates"] == {}


def test_split_counts_match_expected_pilot_dataset():
    report = validate()

    assert report["split_counts"] == {
        "train": 231,
        "validation": 55,
        "test": 44,
    }


def test_all_three_templates_are_represented():
    report = validate()

    assert report["template_counts"] == {
        "sidebar": 110,
        "minimal": 110,
        "classic": 110,
    }


def test_saved_report_matches_validation_result():
    report_path = ROOT / "datasets/reports/render_validation_report.json"

    assert report_path.is_file()

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    current_report = validate()

    assert saved_report == current_report
