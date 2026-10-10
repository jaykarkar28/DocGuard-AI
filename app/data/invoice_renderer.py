
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PAGE_WIDTH = 1275
PAGE_HEIGHT = 1650
MARGIN = 80
DPI = 150

ENTITY_FIELDS = {
    "invoice_number": "INVOICE_NUMBER",
    "invoice_date": "INVOICE_DATE",
    "due_date": "DUE_DATE",
    "supplier_name": "SUPPLIER_NAME",
    "customer_name": "CUSTOMER_NAME",
    "currency": "CURRENCY",
    "subtotal": "SUBTOTAL",
    "tax_amount": "TAX_AMOUNT",
    "total_amount": "TOTAL_AMOUNT",
}

TEMPLATES = ("classic", "sidebar", "minimal")


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a common system font, falling back to Pillow's default."""
    candidates = (
        ["arialbd.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["arial.ttf", "DejaVuSans.ttf"]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _display(value: Any) -> str | None:
    """Convert a present value to display text; never render Python None."""
    if value is None or value == "":
        return None
    return str(value)


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _group_template(record: dict[str, Any]) -> str:
    """Keep related variants on the same layout."""
    group_id = str(record.get("split_group_id", record["document_id"]))
    match = re.search(r"(\d+)$", group_id)
    index = int(match.group(1)) if match else int(
        hashlib.sha256(group_id.encode()).hexdigest()[:8], 16
    )
    return TEMPLATES[index % len(TEMPLATES)]


def render_invoice(
    record: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Render one structured invoice and return pixel-based annotations."""
    if record.get("document_type") != "invoice":
        raise ValueError("Only invoice records can be rendered.")

    invoice = record.get("invoice")
    if not isinstance(invoice, dict):
        raise ValueError(f"{record.get('document_id')}: missing invoice object")

    document_id = str(record["document_id"])
    template_id = _group_template(record)

    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    entities: list[dict[str, Any]] = []

    title_font = _font(44, bold=True)
    heading_font = _font(23, bold=True)
    body_font = _font(21)
    small_font = _font(17)

    def text(
        x: int,
        y: int,
        value: Any,
        label: str | None = None,
        font: ImageFont.ImageFont = body_font,
    ) -> int:
        """Draw a value and capture its actual visible text bounds."""
        rendered = _display(value)
        if rendered is None:
            return y

        draw.text((x, y), rendered, fill="black", font=font)

        if label:
            left, top, right, bottom = draw.textbbox(
                (x, y), rendered, font=font
            )
            entities.append({
                "label": label,
                "text": rendered,
                "page_number": 1,
                "bbox": [left, top, right, bottom],
            })

        left, top, right, bottom = draw.textbbox((x, y), rendered, font=font)
        return y + (bottom - top) + 12

    def rule(y: int, x1: int = MARGIN, x2: int = PAGE_WIDTH - MARGIN):
        draw.line((x1, y, x2, y), fill="#777777", width=2)

    def field(label: str, value: Any, x: int, y: int) -> int:
        """Draw a field label and annotate the field value only."""
        text(x, y, f"{label}:", font=small_font)
        return text(x, y + 25, value, ENTITY_FIELDS.get(label), body_font)

    # Three distinct layouts; variants within a family share the same template.
    if template_id == "classic":
        text(MARGIN, 65, "INVOICE", font=title_font)
        text(MARGIN, 135, invoice.get("supplier_name"), "SUPPLIER_NAME")
        text(MARGIN, 190, "BILL TO", font=heading_font)
        text(MARGIN, 225, invoice.get("customer_name"), "CUSTOMER_NAME")
        field("invoice_number", invoice.get("invoice_number"), 770, 135)
        field("invoice_date", invoice.get("invoice_date"), 770, 220)
        field("due_date", invoice.get("due_date"), 770, 305)
        y = 410

    elif template_id == "sidebar":
        draw.rectangle((0, 0, 390, PAGE_HEIGHT), fill="#EAF0F5")
        text(55, 65, "INVOICE", font=title_font)
        text(55, 155, "SUPPLIER", font=heading_font)
        text(55, 195, invoice.get("supplier_name"), "SUPPLIER_NAME")
        text(55, 340, "CUSTOMER", font=heading_font)
        text(55, 380, invoice.get("customer_name"), "CUSTOMER_NAME")
        field("invoice_number", invoice.get("invoice_number"), 450, 150)
        field("invoice_date", invoice.get("invoice_date"), 450, 240)
        field("due_date", invoice.get("due_date"), 450, 330)
        y = 455

    else:  # minimal
        text(MARGIN, 65, invoice.get("supplier_name"), "SUPPLIER_NAME",
             title_font)
        text(MARGIN, 135, "INVOICE", font=heading_font)
        text(760, 135, invoice.get("invoice_number"), "INVOICE_NUMBER")
        rule(190)
        text(MARGIN, 220, "CUSTOMER", font=small_font)
        text(MARGIN, 250, invoice.get("customer_name"), "CUSTOMER_NAME")
        text(760, 220, "INVOICE DATE", font=small_font)
        text(760, 250, invoice.get("invoice_date"), "INVOICE_DATE")
        text(760, 300, "DUE DATE", font=small_font)
        text(760, 330, invoice.get("due_date"), "DUE_DATE")
        y = 405

    # Line-item table.
    rule(y)
    headers = [
        ("DESCRIPTION",  MARGIN),
        ("QTY",          700),
        ("UNIT PRICE",   820),
        ("LINE TOTAL",   1030),
    ]
    for header, x in headers:
        text(x, y + 15, header, font=heading_font)
    y += 65
    rule(y)

    line_items = invoice.get("line_items")
    if not isinstance(line_items, list):
        line_items = []

    for item in line_items:
        if y > PAGE_HEIGHT - 330:
            raise ValueError(
                f"{document_id}: invoice content exceeds one-page canvas"
            )

        if not isinstance(item, dict):
            continue

        text(MARGIN, y, item.get("description"), "LINE_ITEM_DESCRIPTION")
        text(700, y, item.get("quantity"), "LINE_ITEM_QUANTITY")
        text(820, y, item.get("unit_price"), "LINE_ITEM_UNIT_PRICE")
        text(1030, y, item.get("line_total"), "LINE_ITEM_TOTAL")
        y += 43

    y += 25
    rule(y)
    y += 25

    # Totals block; absent values remain absent rather than displaying "None".
    text(770, y, "CURRENCY", font=small_font)
    text(1030, y, invoice.get("currency"), "CURRENCY")
    y += 45

    text(770, y, "SUBTOTAL", font=small_font)
    text(1030, y, invoice.get("subtotal"), "SUBTOTAL")
    y += 45

    text(770, y, "TAX", font=small_font)
    text(1030, y, invoice.get("tax_amount"), "TAX_AMOUNT")
    y += 55

    text(770, y, "TOTAL", font=heading_font)
    text(1030, y, invoice.get("total_amount"), "TOTAL_AMOUNT",
         heading_font)

    # Explicit image metadata.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{_safe_id(document_id)}.png"
    image.save(image_path, format="PNG", dpi=(DPI, DPI))

    return {
        "document_id": document_id,
        "document_type": "invoice",
        "source": record.get("source", {}),
        "split_group_id": record.get("split_group_id"),
        "template_id": template_id,
        "image_path": image_path.as_posix(),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "width": PAGE_WIDTH,
        "height": PAGE_HEIGHT,
        "dpi": DPI,
        "page_number": 1,
        "entities": entities,
        "expected_risks": record.get("expected_risks", []),
        "annotation_version": "0.1",
    }


def render_jsonl_dataset(
    input_path: str | Path,
    image_dir: str | Path,
    annotation_path: str | Path,
) -> dict[str, int]:
    """Render JSONL records and write one annotation record per image."""
    input_path = Path(input_path)
    annotation_path = Path(annotation_path)
    annotation_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    seen_ids: set[str] = set()

    with input_path.open("r", encoding="utf-8-sig") as source, \
            annotation_path.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            document_id = str(record.get("document_id", ""))

            if not document_id or document_id in seen_ids:
                raise ValueError(
                    f"Missing or duplicate document_id at line {line_number}"
                )
            seen_ids.add(document_id)

            annotation = render_invoice(record, image_dir)
            target.write(json.dumps(annotation, ensure_ascii=False) + "\n")
            count += 1

    return {"rendered_images": count, "unique_document_ids": len(seen_ids)}
