from typing import Any


DOCUMENT_TYPE_ALIASES = {
    "invoice": "invoice",
    "invoices": "invoice",
    "contract": "contract",
    "contracts": "contract",
}

SEVERITY_ALIASES = {
    "low": "low",
    "medium": "medium",
    "moderate": "medium",
    "high": "high",
    "critical": "critical",
}


def normalize_document_type(value: str) -> str:
    """Normalize known document type aliases."""
    normalized = value.strip().lower()

    if normalized not in DOCUMENT_TYPE_ALIASES:
        raise ValueError(f"Unsupported document type: {value!r}")

    return DOCUMENT_TYPE_ALIASES[normalized]


def normalize_severity(value: str) -> str:
    """Normalize known risk severity aliases."""
    normalized = value.strip().lower()

    if normalized not in SEVERITY_ALIASES:
        raise ValueError(f"Unsupported severity: {value!r}")

    return SEVERITY_ALIASES[normalized]


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized copy of a document record.

    Does not modify the input dictionary or invent missing metadata.
    """
    normalized = record.copy()

    if isinstance(normalized.get("document_type"), str):
        normalized["document_type"] = normalize_document_type(
            normalized["document_type"]
        )

    if isinstance(normalized.get("risks"), list):
        normalized["risks"] = [
            {
                **risk,
                "severity": normalize_severity(risk["severity"]),
            }
            if isinstance(risk, dict)
            and isinstance(risk.get("severity"), str)
            else risk
            for risk in normalized["risks"]
        ]

    return normalized
