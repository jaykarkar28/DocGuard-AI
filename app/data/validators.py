from collections import Counter
from collections.abc import Iterable

from pydantic import ValidationError

from app.data.schemas import DocumentRecord


def validate_document_records(
    records: Iterable[dict | DocumentRecord],
) -> list[str]:
    """Validate document records and return actionable error messages."""
    errors: list[str] = []
    seen_ids: set[str] = set()

    for index, record in enumerate(records):
        try:
            document = (
                record
                if isinstance(record, DocumentRecord)
                else DocumentRecord.model_validate(record)
            )
        except ValidationError as exc:
            for issue in exc.errors():
                location = ".".join(str(part) for part in issue["loc"])
                errors.append(
                    f"Record {index}: {location}: {issue['msg']}"
                )
            continue

        if document.document_id in seen_ids:
            errors.append(
                f"Record {index}: duplicate document_id "
                f"'{document.document_id}'"
            )
        else:
            seen_ids.add(document.document_id)

    return errors


def summarize_document_records(
    records: Iterable[dict | DocumentRecord],
) -> dict:
    """Summarize document counts by type and annotation totals."""
    documents = [
        record
        if isinstance(record, DocumentRecord)
        else DocumentRecord.model_validate(record)
        for record in records
    ]

    type_counts = Counter(doc.document_type for doc in documents)

    return {
        "total_documents": len(documents),
        "document_types": dict(type_counts),
        "total_pages": sum(len(doc.pages) for doc in documents),
        "total_entities": sum(len(doc.entities) for doc in documents),
        "total_risks": sum(len(doc.risks) for doc in documents),
        "unique_document_ids": len(
            {doc.document_id for doc in documents}
        ),
    }
