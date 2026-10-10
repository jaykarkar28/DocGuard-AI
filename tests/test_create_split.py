import json

from app.data.loaders import load_jsonl
from scripts.create_splits import create_split_artifacts


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


def test_create_split_artifacts_writes_all_files(tmp_path):
    output_dir = tmp_path / "splits"
    report_path = tmp_path / "reports" / "split_report.json"

    report = create_split_artifacts(
        make_dataset(),
        output_dir=output_dir,
        report_path=report_path,
        seed=42,
    )

    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
    assert report_path.exists()
    assert report["leakage_check_passed"] is True


def test_split_files_preserve_all_documents(tmp_path):
    records = make_dataset()
    output_dir = tmp_path / "splits"

    create_split_artifacts(
        records,
        output_dir=output_dir,
        report_path=tmp_path / "report.json",
    )

    loaded_ids = set()

    for split_name in ("train", "validation", "test"):
        documents = load_jsonl(output_dir / f"{split_name}.jsonl")
        loaded_ids.update(document.document_id for document in documents)

    assert loaded_ids == {record["document_id"] for record in records}
    assert len(loaded_ids) == len(records)


def test_report_has_document_and_group_counts(tmp_path):
    report_path = tmp_path / "report.json"

    report = create_split_artifacts(
        make_dataset(),
        output_dir=tmp_path / "splits",
        report_path=report_path,
    )

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["total_documents"] == 40
    assert report["total_unique_groups"] == 20
    assert saved_report["leakage_check_passed"] is True

    for split_name in ("train", "validation", "test"):
        assert saved_report["splits"][split_name]["document_count"] > 0
        assert saved_report["splits"][split_name]["group_count"] > 0


def test_groups_do_not_cross_split_files(tmp_path):
    output_dir = tmp_path / "splits"

    create_split_artifacts(
        make_dataset(),
        output_dir=output_dir,
        report_path=tmp_path / "report.json",
    )

    group_to_split = {}

    for split_name in ("train", "validation", "test"):
        for document in load_jsonl(output_dir / f"{split_name}.jsonl"):
            group_id = document.split_group_id

            # Multiple documents may share a group, but only within
            # the same split.
            if group_id in group_to_split:
                assert group_to_split[group_id] == split_name, (
                    f"Group {group_id} appears in multiple splits"
                )
            else:
                group_to_split[group_id] = split_name

    assert len(group_to_split) == 20



def test_same_seed_produces_same_split_assignments(tmp_path):
    records = make_dataset()

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    create_split_artifacts(
        records,
        output_dir=first_dir,
        report_path=tmp_path / "first_report.json",
        seed=42,
    )
    create_split_artifacts(
        records,
        output_dir=second_dir,
        report_path=tmp_path / "second_report.json",
        seed=42,
    )

    for split_name in ("train", "validation", "test"):
        first_ids = {
            doc.document_id
            for doc in load_jsonl(first_dir / f"{split_name}.jsonl")
        }
        second_ids = {
            doc.document_id
            for doc in load_jsonl(second_dir / f"{split_name}.jsonl")
        }

        assert first_ids == second_ids
