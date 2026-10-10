"""Leakage-safe splitting for structured synthetic invoice records."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLIT_NAMES = ("train", "validation", "test")


def _allocate_group_counts(
    group_count: int,
    ratios: dict[str, float],
) -> dict[str, int]:
    """Allocate whole groups using the largest-remainder method."""
    counts = {
        name: int(group_count * ratios[name])
        for name in SPLIT_NAMES
    }
    remaining = group_count - sum(counts.values())

    remainders = {
        name: group_count * ratios[name] - counts[name]
        for name in SPLIT_NAMES
    }
    priority = sorted(
        SPLIT_NAMES,
        key=lambda name: (-remainders[name], SPLIT_NAMES.index(name)),
    )

    for name in priority[:remaining]:
        counts[name] += 1

    if any(counts[name] == 0 for name in SPLIT_NAMES):
        raise ValueError(
            "Not enough invoice families to populate all three splits."
        )

    return counts


def split_invoice_records(
    records: list[dict[str, Any]],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split records by split_group_id, never by individual record."""
    ratios = {
        "train": train_ratio,
        "validation": val_ratio,
        "test": test_ratio,
    }

    if any(ratio <= 0 for ratio in ratios.values()):
        raise ValueError("All split ratios must be greater than zero.")

    if abs(sum(ratios.values()) - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0.")

    if not records:
        raise ValueError("Cannot split an empty dataset.")

    seen_ids: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {index} must be a dictionary.")

        document_id = record.get("document_id")
        group_id = record.get("split_group_id")

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"Record at index {index} has no valid document_id.")

        if document_id in seen_ids:
            raise ValueError(f"Duplicate document_id: {document_id}")
        seen_ids.add(document_id)

        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(
                f"Record {document_id} has no valid split_group_id."
            )

        if not isinstance(record.get("expected_risks"), list):
            raise ValueError(
                f"Record {document_id} must have an expected_risks list."
            )

        grouped[group_id].append(record)

    group_ids = list(grouped)
    if len(group_ids) < 3:
        raise ValueError("At least three distinct invoice families are required.")

    rng = random.Random(seed)
    rng.shuffle(group_ids)

    group_counts = _allocate_group_counts(len(group_ids), ratios)
    assignments: dict[str, str] = {}
    cursor = 0

    for split_name in SPLIT_NAMES:
        next_cursor = cursor + group_counts[split_name]
        for group_id in group_ids[cursor:next_cursor]:
            assignments[group_id] = split_name
        cursor = next_cursor

    splits: dict[str, list[dict[str, Any]]] = {
        name: [] for name in SPLIT_NAMES
    }

    for group_id, group_records in grouped.items():
        split_name = assignments[group_id]
        splits[split_name].extend(group_records)

    return splits


def build_invoice_split_report(
    splits: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    ratios: dict[str, float],
) -> dict[str, Any]:
    """Summarize split sizes, defect labels, and group leakage."""
    report: dict[str, Any] = {
        "seed": seed,
        "requested_ratios": ratios,
        "splits": {},
    }
    group_owners: dict[str, set[str]] = defaultdict(set)
    all_document_ids: list[str] = []

    for split_name in SPLIT_NAMES:
        records = splits[split_name]
        label_counts: Counter[str] = Counter()
        group_ids: set[str] = set()
        valid_records = 0
        defective_records = 0

        for record in records:
            document_id = record["document_id"]
            group_id = record["split_group_id"]
            all_document_ids.append(document_id)
            group_ids.add(group_id)
            group_owners[group_id].add(split_name)

            risks = record["expected_risks"]
            if risks:
                defective_records += 1
            else:
                valid_records += 1

            for risk in risks:
                label_counts[risk["risk_type"]] += 1

        report["splits"][split_name] = {
            "record_count": len(records),
            "group_count": len(group_ids),
            "valid_records": valid_records,
            "defective_records": defective_records,
            "defect_label_counts": dict(sorted(label_counts.items())),
            "document_ids_unique_within_split": (
                len({r["document_id"] for r in records}) == len(records)
            ),
        }

    leakage = {
        group_id: sorted(owners)
        for group_id, owners in group_owners.items()
        if len(owners) > 1
    }

    report["total_records"] = len(all_document_ids)
    report["unique_document_ids"] = len(set(all_document_ids))
    report["duplicate_document_ids"] = (
        len(all_document_ids) - len(set(all_document_ids))
    )
    report["group_leakage"] = leakage
    report["leakage_check_passed"] = not leakage
    report["document_id_check_passed"] = (
        len(all_document_ids) == len(set(all_document_ids))
    )

    return report


def write_invoice_splits(
    splits: dict[str, list[dict[str, Any]]],
    output_dir: str | Path,
) -> None:
    """Write structured records to JSONL without changing nested payloads."""
    import json

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    for split_name in SPLIT_NAMES:
        output_path = directory / f"{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8", newline="\n") as file:
            for record in splits[split_name]:
                file.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True)
                    + "\n"
                )