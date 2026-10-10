import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.data.loaders import load_jsonl, save_jsonl
from app.data.schemas import DocumentRecord
from app.data.splitting import split_records_by_group


def build_split_report(
    splits: dict[str, list[DocumentRecord]],
    *,
    seed: int,
    ratios: dict[str, float],
) -> dict[str, Any]:
    """Build split statistics and verify group separation."""
    group_sets = {
        split_name: {doc.split_group_id for doc in documents}
        for split_name, documents in splits.items()
    }

    leakage: dict[str, list[str]] = {}
    split_names = list(group_sets)

    for index, first_name in enumerate(split_names):
        for second_name in split_names[index + 1:]:
            overlap = group_sets[first_name] & group_sets[second_name]
            if overlap:
                leakage[f"{first_name}__{second_name}"] = sorted(overlap)

    document_type_counts = {
        name: dict(Counter(doc.document_type for doc in documents))
        for name, documents in splits.items()
    }

    report = {
        "seed": seed,
        "requested_ratios": ratios,
        "total_documents": sum(len(docs) for docs in splits.values()),
        "total_unique_groups": len(set().union(*group_sets.values())),
        "splits": {
            name: {
                "document_count": len(documents),
                "group_count": len(group_sets[name]),
                "document_types": document_type_counts[name],
            }
            for name, documents in splits.items()
        },
        "group_leakage": leakage,
        "leakage_check_passed": not leakage,
    }

    if not report["leakage_check_passed"]:
        raise ValueError(f"Group leakage detected: {leakage}")

    return report


def create_split_artifacts(
    records: list[dict[str, Any] | DocumentRecord],
    *,
    output_dir: str | Path,
    report_path: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Any]:
    """Split records and save JSONL files plus a JSON quality report."""
    splits = split_records_by_group(
        records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    ratios = {
        "train": train_ratio,
        "validation": val_ratio,
        "test": test_ratio,
    }

    report = build_split_report(splits, seed=seed, ratios=ratios)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, documents in splits.items():
        save_jsonl(documents, output_path / f"{split_name}.jsonl")

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leakage-safe DocGuard AI dataset splits."
    )
    parser.add_argument(
        "--input",
        default="datasets/processed/documents.jsonl",
        help="Validated JSONL document records.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/splits",
        help="Directory for train/validation/test JSONL files.",
    )
    parser.add_argument(
        "--report",
        default="datasets/reports/split_report.json",
        help="Path to the JSON split report.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = list(load_jsonl(args.input))

    report = create_split_artifacts(
        records,
        output_dir=args.output_dir,
        report_path=args.report,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(json.dumps(report, indent=2))
    print("\nDataset split files and report created successfully.")


if __name__ == "__main__":
    main()
