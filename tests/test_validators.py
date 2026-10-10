import json

import pytest

from app.data.loaders import (
    DatasetLoadError,
    load_jsonl,
    save_jsonl,
)


def make_record(document_id: str) -> dict:
    return {
        "document_id": document_id,
        "document_type": "invoice",
        "source": {
            "dataset": "synthetic",
            "license": "internal_synthetic",
            "source_document_id": None,
        },
        "split_group_id": f"template-{document_id}",
        "pages": [
            {
                "page_number": 1,
                "image_path": f"images/{document_id}.png",
                "width": 1000,
                "height": 1400,
            }
        ],
        "entities": [],
        "risks": [],
        "annotation_version": "0.1",
    }


def test_save_and_load_jsonl_round_trip(tmp_path):
    path = tmp_path / "documents.jsonl"
    records = [make_record("INV-001"), make_record("INV-002")]

    assert save_jsonl(records, path) == 2

    loaded = list(load_jsonl(path))

    assert len(loaded) == 2
    assert loaded[0].document_id == "INV-001"
    assert loaded[1].document_id == "INV-002"


def test_loader_skips_blank_lines(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text(
        json.dumps(make_record("INV-001")) + "\n\n",
        encoding="utf-8",
    )

    assert len(list(load_jsonl(path))) == 1


def test_loader_reports_invalid_json_with_line_number(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text('{"document_id":\n', encoding="utf-8")

    with pytest.raises(DatasetLoadError, match=r":1: invalid JSON"):
        list(load_jsonl(path))


def test_loader_rejects_non_object_json(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text('"not a document"\n', encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="JSON object"):
        list(load_jsonl(path))


def test_loader_can_return_raw_dictionaries(tmp_path):
    path = tmp_path / "documents.jsonl"
    path.write_text(
        json.dumps(make_record("INV-001")) + "\n",
        encoding="utf-8",
    )

    loaded = list(load_jsonl(path, validate=False))

    assert isinstance(loaded[0], dict)


def test_loader_rejects_invalid_document_record(tmp_path):
    record = make_record("INV-001")
    record["document_type"] = "receipt"

    path = tmp_path / "documents.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="invalid document record"):
        list(load_jsonl(path))


def test_save_rejects_invalid_records(tmp_path):
    record = make_record("INV-001")
    record["document_type"] = "receipt"

    with pytest.raises(DatasetLoadError, match="Record 1"):
        save_jsonl([record], tmp_path / "invalid.jsonl")
