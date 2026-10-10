import random
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from app.data.schemas import DocumentRecord


def split_records_by_group(
    records: Iterable[dict[str, Any] | DocumentRecord],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[DocumentRecord]]:
    """Split documents by group to prevent group-level data leakage.

    Documents sharing a split_group_id always remain in the same split.
    Ratios are approximate because groups may have different sizes.
    """
    ratios = (train_ratio, val_ratio, test_ratio)

    if any(ratio <= 0 or ratio >= 1 for ratio in ratios):
        raise ValueError("Each split ratio must be greater than 0 and less than 1")

    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    documents = [
        record
        if isinstance(record, DocumentRecord)
        else DocumentRecord.model_validate(record)
        for record in records
    ]

    if not documents:
        raise ValueError("Cannot split an empty dataset")

    seen_ids: set[str] = set()
    grouped: dict[str, list[DocumentRecord]] = defaultdict(list)

    for document in documents:
        if document.document_id in seen_ids:
            raise ValueError(
                f"Duplicate document_id: {document.document_id}"
            )
        seen_ids.add(document.document_id)
        grouped[document.split_group_id].append(document)

    group_ids = list(grouped)

    if len(group_ids) < 3:
        raise ValueError(
            "At least 3 distinct groups are required to create "
            "non-empty train, validation, and test splits"
        )

    random.Random(seed).shuffle(group_ids)

    # Allocate groups approximately according to the requested ratios.
    group_count = len(group_ids)
    train_count = max(1, round(group_count * train_ratio))
    val_count = max(1, round(group_count * val_ratio))

    # Reserve at least one group for each split.
    if train_count + val_count >= group_count:
        train_count = max(1, group_count - 2)
        val_count = 1

    train_groups = set(group_ids[:train_count])
    val_groups = set(group_ids[train_count:train_count + val_count])
    test_groups = set(group_ids[train_count + val_count:])

    result = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for document in documents:
        group_id = document.split_group_id

        if group_id in train_groups:
            result["train"].append(document)
        elif group_id in val_groups:
            result["validation"].append(document)
        else:
            result["test"].append(document)

    return result
