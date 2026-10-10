
import json

from PIL import Image

from app.data.invoice_renderer import (
    PAGE_HEIGHT,
    PAGE_WIDTH,
    render_invoice,
    render_jsonl_dataset,
)
from app.data.synthetic_invoice_dataset import generate_invoice_dataset


def test_renderer_creates_image_and_valid_annotations(tmp_path):
    record = generate_invoice_dataset(base_count=1, seed=42)[0]

    annotation = render_invoice(record, tmp_path)

    image_path = tmp_path / f"{record['document_id']}.png"
    assert image_path.is_file()
    assert annotation["width"] == PAGE_WIDTH
    assert annotation["height"] == PAGE_HEIGHT
    assert annotation["document_id"] == record["document_id"]

    with Image.open(image_path) as image:
        assert image.size == (PAGE_WIDTH, PAGE_HEIGHT)
        assert image.format == "PNG"

    for entity in annotation["entities"]:
        x0, y0, x1, y1 = entity["bbox"]
        assert 0 <= x0 < x1 <= PAGE_WIDTH
        assert 0 <= y0 < y1 <= PAGE_HEIGHT
        assert entity["text"]


def test_missing_field_is_not_rendered_or_annotated(tmp_path):
    record = generate_invoice_dataset(base_count=1, seed=42)[1]
    assert record["expected_risks"]

    record["invoice"]["invoice_number"] = None
    annotation = render_invoice(record, tmp_path)

    labels = [entity["label"] for entity in annotation["entities"]]
    assert "INVOICE_NUMBER" not in labels


def test_related_variants_use_same_template(tmp_path):
    records = generate_invoice_dataset(base_count=1, seed=42)
    annotations = [
        render_invoice(record, tmp_path)
        for record in records
    ]

    assert len({item["template_id"] for item in annotations}) == 1


def test_jsonl_pipeline_writes_one_annotation_per_record(tmp_path):
    records = generate_invoice_dataset(base_count=2, seed=42)
    source = tmp_path / "input.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    images_dir = tmp_path / "images"

    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    result = render_jsonl_dataset(source, images_dir, annotations_path)

    lines = [
        json.loads(line)
        for line in annotations_path.read_text(encoding="utf-8").splitlines()
    ]
    assert result["rendered_images"] == len(records)
    assert len(lines) == len(records)
    assert len({item["document_id"] for item in lines}) == len(records)
    assert len(list(images_dir.glob("*.png"))) == len(records)
