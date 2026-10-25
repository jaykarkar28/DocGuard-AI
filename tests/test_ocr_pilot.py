
from __future__ import annotations

import pytest

from scripts.run_ocr_pilot import build_report, select_pilot


def make_annotations() -> list[dict]:
    records = []

    for template in ("classic", "minimal", "sidebar"):
        for index in range(12):
            records.append(
                {
                    "document_id": f"{template}-{index:03d}",
                    "template_id": template,
                    "image_path": f"datasets/images/{template}-{index:03d}.png",
                    "split_group_id": f"group-{template}-{index:03d}",
                }
            )

    return records


def test_pilot_is_balanced_across_templates():
    selected = select_pilot(
        make_annotations(),
        samples_per_template=4,
        seed=42,
    )

    counts = {
        template: sum(
            record["template_id"] == template for record in selected
        )
        for template in ("classic", "minimal", "sidebar")
    }

    assert len(selected) == 12
    assert counts == {
        "classic": 4,
        "minimal": 4,
        "sidebar": 4,
    }


def test_pilot_selection_is_reproducible():
    annotations = make_annotations()

    first = select_pilot(
        annotations,
        samples_per_template=5,
        seed=123,
    )
    second = select_pilot(
        annotations,
        samples_per_template=5,
        seed=123,
    )

    assert [
        row["document_id"] for row in first
    ] == [
        row["document_id"] for row in second
    ]


def test_pilot_rejects_insufficient_template_records():
    annotations = [
        row
        for row in make_annotations()
        if not (
            row["template_id"] == "sidebar"
            and row["document_id"].endswith(("010", "011"))
        )
    ]

    with pytest.raises(ValueError, match="sidebar"):
        select_pilot(
            annotations,
            samples_per_template=11,
            seed=42,
        )


def test_report_does_not_claim_unmeasured_accuracy():
    results = [
        {
            "document_id": "classic-001",
            "template_id": "classic",
            "engine_version": "5.5.3",
            "runtime_seconds": 0.25,
            "word_count": 30,
            "text": "Example invoice",
        }
    ]

    report = build_report(
        results,
        seed=42,
        samples_per_template=1,
    )

    assert report["image_count"] == 1
    assert report["accuracy_metrics"]["cer"] is None
    assert report["accuracy_metrics"]["wer"] is None
    assert report["accuracy_metrics"]["field_exact_match"] is None
