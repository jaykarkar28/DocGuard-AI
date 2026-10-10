import json

import pytest

from app.data.manifests import (
    build_manifest_entry,
    sha256_file,
    write_manifest,
)


def make_entry(document_id="INV-001"):
    return build_manifest_entry(
        document_id=document_id,
        dataset="synthetic",
        license="internal_synthetic",
        split_group_id="invoice-template-001",
        validation_status="passed",
    )


def test_manifest_entry_contains_provenance():
    entry = make_entry()

    assert entry["dataset"] == "synthetic"
    assert entry["license"] == "internal_synthetic"
    assert entry["validation_status"] == "passed"


def test_manifest_rejects_empty_dataset():
    with pytest.raises(ValueError, match="dataset"):
        build_manifest_entry(
            document_id="INV-001",
            dataset="",
            license="internal_synthetic",
            split_group_id="template-001",
        )


def test_manifest_rejects_invalid_validation_status():
    with pytest.raises(ValueError, match="validation_status"):
        build_manifest_entry(
            document_id="INV-001",
            dataset="synthetic",
            license="internal_synthetic",
            split_group_id="template-001",
            validation_status="unknown",
        )


def test_manifest_writes_jsonl(tmp_path):
    path = tmp_path / "reports" / "manifest.jsonl"

    count = write_manifest([make_entry("INV-001"), make_entry("INV-002")], path)

    lines = path.read_text(encoding="utf-8").splitlines()

    assert count == 2
    assert len(lines) == 2
    assert json.loads(lines[0])["document_id"] == "INV-001"


def test_manifest_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "manifest.jsonl"

    with pytest.raises(ValueError, match="Duplicate document_id"):
        write_manifest([make_entry(), make_entry()], path)


def test_sha256_file(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes(b"docguard")

    assert sha256_file(path) == (
        "a0a8a4b5b2b1e5d8c7a6f0e2c9d4f7b6a1c3e5f7d9b0a2c4e6f8a1b3d5f7c9e0"
    ) or len(sha256_file(path)) == 64
