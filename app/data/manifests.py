import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path) -> str:
    """Calculate a file's SHA-256 hash without loading it all into memory."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def build_manifest_entry(
    *,
    document_id: str,
    dataset: str,
    license: str,
    split_group_id: str,
    source_document_id: str | None = None,
    content_hash: str | None = None,
    validation_status: str = "pending",
) -> dict[str, Any]:
    """Create a reproducible metadata entry for a document."""
    required = {
        "document_id": document_id,
        "dataset": dataset,
        "license": license,
        "split_group_id": split_group_id,
    }

    for field, value in required.items():
        if not value or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")

    if validation_status not in {"pending", "passed", "failed"}:
        raise ValueError("validation_status must be pending, passed, or failed")

    return {
        "document_id": document_id,
        "dataset": dataset,
        "license": license,
        "source_document_id": source_document_id,
        "split_group_id": split_group_id,
        "content_hash": content_hash,
        "validation_status": validation_status,
    }


def write_manifest(
    entries: Iterable[dict[str, Any]],
    path: str | Path,
) -> int:
    """Write manifest entries to JSONL and return the number written."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_ids: set[str] = set()

    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for index, entry in enumerate(entries, start=1):
            document_id = entry.get("document_id")

            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError(
                    f"Manifest entry {index} has an invalid document_id"
                )

            if document_id in seen_ids:
                raise ValueError(
                    f"Duplicate document_id in manifest: {document_id}"
                )

            seen_ids.add(document_id)
            file.write(
                json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
            )
            count += 1

    return count
