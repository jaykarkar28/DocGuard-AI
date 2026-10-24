"""Create readable visual previews of invoice entity bounding boxes.

Bounding boxes are drawn on the invoice. Entity details are placed in a side
legend so long labels do not overlap neighboring invoice fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COLORS = [
    "#D32F2F", "#1565C0", "#2E7D32", "#6A1B9A", "#EF6C00",
    "#00838F", "#AD1457", "#4E342E", "#4527A0", "#558B2F",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(record)
    return records


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _valid_bbox(bbox: Any, width: int, height: int) -> tuple[int, int, int, int]:
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or not all(isinstance(v, (int, float)) for v in bbox)
    ):
        raise ValueError(f"Invalid bbox: {bbox!r}")
    x0, y0, x1, y1 = (int(round(v)) for v in bbox)
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height or x1 <= x0 or y1 <= y0:
        raise ValueError(f"Bbox is outside image bounds or has no area: {bbox!r}")
    return x0, y0, x1, y1


def create_preview(
    annotation: dict[str, Any],
    project_root: Path,
    output_dir: Path,
) -> Path:
    """Save a preview with outlined entities and a non-overlapping side legend."""
    if "document_id" not in annotation:
        raise ValueError("Annotation is missing document_id")
    image_path = Path(str(annotation.get("image_path", "")))
    if not image_path.is_absolute():
        image_path = project_root / image_path
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found for {annotation.get('document_id')}: {image_path}"
        )

    with Image.open(image_path) as source:
        invoice = source.convert("RGB")

    entities = annotation.get("entities", [])
    if not isinstance(entities, list):
        raise ValueError("entities must be a list")

    # Add a separate legend area; annotation text never covers invoice content.
    legend_width = max(430, min(680, invoice.width // 2))
    row_height = 42
    top_padding = 78
    legend_height = max(invoice.height, top_padding + len(entities) * row_height + 24)
    canvas = Image.new("RGB", (invoice.width + legend_width, legend_height), "white")
    canvas.paste(invoice, (0, 0))

    draw = ImageDraw.Draw(canvas)
    title_font = load_font(23)
    body_font = load_font(15)
    small_font = load_font(13)
    draw.rectangle((invoice.width, 0, canvas.width, canvas.height), fill="#F5F7FA")
    draw.text((invoice.width + 18, 18), "ANNOTATION LEGEND", fill="#111827", font=title_font)
    draw.text(
        (invoice.width + 18, 49),
        f"Document: {annotation['document_id']} | Entities: {len(entities)}",
        fill="#4B5563",
        font=small_font,
    )

    for index, entity in enumerate(entities, start=1):
        if not isinstance(entity, dict):
            raise ValueError(f"Entity {index} must be an object")
        x0, y0, x1, y1 = _valid_bbox(entity.get("bbox"), invoice.width, invoice.height)
        color = COLORS[(index - 1) % len(COLORS)]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)

        # Numbered marker keeps the visual association without long overlapping tags.
        marker = str(index)
        marker_box = draw.textbbox((0, 0), marker, font=small_font)
        marker_w = marker_box[2] - marker_box[0] + 8
        marker_h = marker_box[3] - marker_box[1] + 4
        marker_x = min(max(0, x0), max(0, invoice.width - marker_w))
        marker_y = max(0, y0 - marker_h)
        draw.rectangle(
            (marker_x, marker_y, marker_x + marker_w, marker_y + marker_h),
            fill=color,
        )
        draw.text((marker_x + 4, marker_y + 1), marker, fill="white", font=small_font)

        row_y = top_padding + (index - 1) * row_height
        label = str(entity.get("label", "UNKNOWN"))
        value = str(entity.get("text", ""))
        draw.ellipse(
            (invoice.width + 18, row_y + 3, invoice.width + 32, row_y + 17),
            fill=color,
        )
        draw.text(
            (invoice.width + 40, row_y),
            f"{index}. {label}",
            fill="#111827",
            font=body_font,
        )
        available_width = legend_width - 62
        # Wrap long entity values so the legend remains readable.
        words = value.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if draw.textlength(candidate, font=small_font) <= available_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        if not lines:
            lines = ["(empty)"]
        draw.text(
            (invoice.width + 40, row_y + 19),
            lines[0],
            fill="#374151",
            font=small_font,
        )
        if len(lines) > 1:
            # Preserve readability while keeping one row per entity.
            draw.text(
                (invoice.width + 40, row_y + 32),
                lines[1] + (" …" if len(lines) > 2 else ""),
                fill="#374151",
                font=small_font,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    template_id = str(annotation.get("template_id", "unknown"))
    output_path = output_dir / f"{annotation['document_id']}_{template_id}_preview.png"
    canvas.save(output_path, format="PNG")
    return output_path


def select_template_samples(
    annotations: list[dict[str, Any]],
    samples_per_template: int,
) -> list[dict[str, Any]]:
    if samples_per_template < 1:
        raise ValueError("samples_per_template must be at least 1")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for annotation in annotations:
        template = annotation.get("template_id")
        if not template:
            raise ValueError(
                f"Missing template_id for {annotation.get('document_id')}"
            )
        grouped.setdefault(str(template), []).append(annotation)
    selected: list[dict[str, Any]] = []
    for template in sorted(grouped):
        selected.extend(
            sorted(
                grouped[template],
                key=lambda item: str(item.get("document_id", "")),
            )[:samples_per_template]
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visual previews of invoice annotations."
    )
    parser.add_argument(
        "--annotations",
        default="datasets/processed/rendered_invoice_annotations.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/reports/annotation_previews",
    )
    parser.add_argument("--samples-per-template", type=int, default=1)
    args = parser.parse_args()
    project_root = Path.cwd()
    annotations = load_jsonl(project_root / args.annotations)
    selected = select_template_samples(annotations, args.samples_per_template)
    previews = []
    for annotation in selected:
        output_path = create_preview(annotation, project_root, project_root / args.output_dir)
        previews.append({
            "document_id": annotation["document_id"],
            "template_id": annotation["template_id"],
            "entity_count": len(annotation.get("entities", [])),
            "preview_path": str(output_path),
        })
    report = {
        "preview_count": len(previews),
        "templates": sorted({item["template_id"] for item in previews}),
        "previews": previews,
        "note": (
            "The legend improves readability but visual inspection is still "
            "required to confirm that each box surrounds the intended printed value."
        ),
    }
    report_path = project_root / args.output_dir / "preview_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nPreview report saved to: {report_path}")


if __name__ == "__main__":
    main()
