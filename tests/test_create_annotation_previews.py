from pathlib import Path

import pytest
from PIL import Image

from scripts.create_annotation_previews import (
    _valid_bbox,
    create_preview,
    select_template_samples,
)


def test_valid_bbox_returns_integer_coordinates():
    assert _valid_bbox([1, 2, 10, 20], 100, 100) == (1, 2, 10, 20)


@pytest.mark.parametrize(
    "bbox",
    [
        None,
        [1, 2, 3],
        [1, 2, "x", 4],
        [-1, 2, 10, 20],
        [1, 2, 101, 20],
        [10, 10, 10, 20],
        [10, 20, 5, 15],
    ],
)
def test_invalid_bbox_is_rejected(bbox):
    with pytest.raises(ValueError):
        _valid_bbox(bbox, 100, 100)


def test_select_template_samples_is_deterministic_and_grouped():
    annotations = [
        {"document_id": "b", "template_id": "classic"},
        {"document_id": "a", "template_id": "classic"},
        {"document_id": "c", "template_id": "sidebar"},
    ]
    selected = select_template_samples(annotations, 1)
    assert [(a["template_id"], a["document_id"]) for a in selected] == [
        ("classic", "a"),
        ("sidebar", "c"),
    ]


def test_select_template_samples_rejects_zero():
    with pytest.raises(ValueError):
        select_template_samples([], 0)


def test_create_preview_adds_legend_and_writes_png(tmp_path: Path):
    image_path = tmp_path / "invoice.png"
    Image.new("RGB", (200, 120), "white").save(image_path)
    annotation = {
        "document_id": "DOC-1",
        "template_id": "classic",
        "image_path": str(image_path),
        "entities": [
            {"label": "INVOICE_NUMBER", "text": "INV-001", "bbox": [10, 10, 80, 25]}
        ],
    }
    result = create_preview(annotation, tmp_path, tmp_path / "out")
    assert result.is_file()
    with Image.open(result) as preview:
        assert preview.width > 200
        assert preview.height >= 120
        assert preview.format == "PNG"


def test_create_preview_raises_for_missing_image(tmp_path: Path):
    annotation = {
        "document_id": "DOC-1",
        "template_id": "classic",
        "image_path": "missing.png",
        "entities": [],
    }
    with pytest.raises(FileNotFoundError):
        create_preview(annotation, tmp_path, tmp_path / "out")
