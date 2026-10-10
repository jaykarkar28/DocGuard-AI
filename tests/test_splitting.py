import pytest

from app.data.splitting import split_records_by_group


def make_record(document_id: str, group_id: str) -> dict:
    return {
        "document_id": document_id,
        "document_type": "invoice",
        "source": {
            "dataset": "synthetic",
            "license": "internal_synthetic",
            "source_document_id": None,
        },
        "split_group_id": group_id,
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


def make_dataset(group_count: int = 20) -> list[dict]:
    records = []

    for index in range(group_count):
        group_id = f"template-{index:03d}"
        records.append(make_record(f"INV-{index:03d}-A", group_id))
        records.append(make_record(f"INV-{index:03d}-B", group_id))

    return records


def test_split_contains_all_documents():
    records = make_dataset()
    splits = split_records_by_group(records)

    split_ids = [
        doc.document_id
        for documents in splits.values()
        for doc in documents
    ]

    assert len(split_ids) == len(records)
    assert set(split_ids) == {r["document_id"] for r in records}


def test_groups_never_cross_splits():
    splits = split_records_by_group(make_dataset())

    group_to_split = {}

    for split_name, documents in splits.items():
        for document in documents:
            group_id = document.split_group_id

            if group_id in group_to_split:
                assert group_to_split[group_id] == split_name
            else:
                group_to_split[group_id] = split_name


def test_split_is_reproducible():
    records = make_dataset()

    first = split_records_by_group(records, seed=42)
    second = split_records_by_group(records, seed=42)

    first_ids = {
        name: [doc.document_id for doc in docs]
        for name, docs in first.items()
    }
    second_ids = {
        name: [doc.document_id for doc in docs]
        for name, docs in second.items()
    }

    assert first_ids == second_ids


def test_all_three_splits_are_non_empty():
    splits = split_records_by_group(make_dataset(10))

    assert splits["train"]
    assert splits["validation"]
    assert splits["test"]


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match="empty dataset"):
        split_records_by_group([])


def test_too_few_groups_are_rejected():
    records = [
        make_record("INV-001", "group-1"),
        make_record("INV-002", "group-2"),
    ]

    with pytest.raises(ValueError, match="At least 3 distinct groups"):
        split_records_by_group(records)


def test_duplicate_document_ids_are_rejected():
    records = [
        make_record("INV-001", "group-1"),
        make_record("INV-001", "group-2"),
        make_record("INV-003", "group-3"),
    ]

    with pytest.raises(ValueError, match="Duplicate document_id"):
        split_records_by_group(records)


def test_invalid_ratios_are_rejected():
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_records_by_group(
            make_dataset(),
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.2,
        )
