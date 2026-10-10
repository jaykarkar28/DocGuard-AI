import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.data.schemas import DocumentRecord


class DatasetLoadError(ValueError):
    """Raised when a JSONL dataset contains invalid records."""


def load_jsonl(
    path: str | Path,
    *,
    validate: bool = True,
) -> Iterator[dict[str, Any] | DocumentRecord]:
    """Read JSONL records one at a time.

    When validate=True, each record is returned as a DocumentRecord.
    When validate=False, each record is returned as a dictionary.
    """
    dataset_path = Path(path)

    with dataset_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetLoadError(
                    f"{dataset_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise DatasetLoadError(
                    f"{dataset_path}:{line_number}: "
                    "each JSONL record must be a JSON object"
                )

            if validate:
                try:
                    yield DocumentRecord.model_validate(record)
                except ValidationError as exc:
                    raise DatasetLoadError(
                        f"{dataset_path}:{line_number}: "
                        f"invalid document record: {exc}"
                    ) from exc
            else:
                yield record


def save_jsonl(
    records: Iterable[dict[str, Any] | DocumentRecord],
    path: str | Path,
) -> int:
    """Validate and write records as JSONL. Return the number written."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for index, record in enumerate(records, start=1):
            try:
                document = (
                    record
                    if isinstance(record, DocumentRecord)
                    else DocumentRecord.model_validate(record)
                )
            except ValidationError as exc:
                raise DatasetLoadError(
                    f"Record {index}: invalid document record: {exc}"
                ) from exc

            file.write(
                document.model_dump_json(exclude_none=False) + "\n"
            )
            count += 1

    return count
