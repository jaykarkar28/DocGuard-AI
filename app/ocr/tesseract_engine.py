
"""Memory-conscious Tesseract OCR wrapper for DocGuard AI."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image
from pytesseract import Output


DEFAULT_TESSERACT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Selected after benchmarking PSM 6 and PSM 4 on 330 synthetic invoices.
DEFAULT_OCR_CONFIG = "--oem 3 --psm 4"


def configure_tesseract(
    executable_path: str | Path | None = None,
) -> Path:
    """Configure and verify the Tesseract executable."""
    configured_path = (
        executable_path
        or os.environ.get("DOCGUARD_TESSERACT_CMD")
        or DEFAULT_TESSERACT_PATH
    )
    path = Path(configured_path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            "Tesseract executable not found at "
            f"{path}. Install Tesseract or set "
            "DOCGUARD_TESSERACT_CMD to its executable path."
        )

    pytesseract.pytesseract.tesseract_cmd = str(path)
    return path


def _word_text(item: dict[str, Any]) -> str:
    """Return stripped text for an OCR token."""
    value = item.get("text", "")
    return value.strip() if isinstance(value, str) else ""


def extract_word_records(
    image: Image.Image,
    *,
    language: str = "eng",
    config: str = DEFAULT_OCR_CONFIG,
) -> list[dict[str, Any]]:
    """Extract word-level text, confidence and pixel bounding boxes."""
    data = pytesseract.image_to_data(
        image,
        lang=language,
        config=config,
        output_type=Output.DICT,
    )

    records: list[dict[str, Any]] = []
    count = len(data.get("text", []))

    for index in range(count):
        text = _word_text({"text": data["text"][index]})
        if not text:
            continue

        try:
            confidence = float(data["conf"][index])
        except (ValueError, TypeError, IndexError):
            confidence = -1.0

        # Tesseract uses negative confidence for non-word rows.
        if confidence < 0:
            continue

        left = int(data["left"][index])
        top = int(data["top"][index])
        width = int(data["width"][index])
        height = int(data["height"][index])

        if width <= 0 or height <= 0:
            continue

        records.append(
            {
                "text": text,
                "confidence": round(confidence, 4),
                "bbox": [
                    left,
                    top,
                    left + width,
                    top + height,
                ],
                "block_num": int(data["block_num"][index]),
                "par_num": int(data["par_num"][index]),
                "line_num": int(data["line_num"][index]),
                "word_num": int(data["word_num"][index]),
            }
        )

    return records


def reconstruct_text(words: list[dict[str, Any]]) -> str:
    """Reconstruct OCR text while preserving Tesseract's line grouping."""
    lines: list[str] = []
    current_line_key: tuple[int, int, int] | None = None
    current_words: list[str] = []

    for word in words:
        line_key = (
            word["block_num"],
            word["par_num"],
            word["line_num"],
        )

        if current_line_key is not None and line_key != current_line_key:
            lines.append(" ".join(current_words))
            current_words = []

        current_line_key = line_key
        current_words.append(word["text"])

    if current_words:
        lines.append(" ".join(current_words))

    return "\n".join(lines)


def run_tesseract(
    image_path: str | Path,
    *,
    document_id: str | None = None,
    language: str = "eng",
    config: str = DEFAULT_OCR_CONFIG,
    executable_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run OCR on one image and return a JSON-serializable result."""
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Invoice image not found: {path}")

    configure_tesseract(executable_path)

    started = time.perf_counter()

    with Image.open(path) as opened_image:
        opened_image.load()
        image = opened_image.convert("RGB")
        width, height = image.size

        words = extract_word_records(
            image,
            language=language,
            config=config,
        )

    elapsed_seconds = time.perf_counter() - started

    return {
        "document_id": document_id or path.stem,
        "image_path": str(path),
        "image_width": width,
        "image_height": height,
        "engine": "tesseract",
        "engine_version": str(pytesseract.get_tesseract_version()),
        "language": language,
        "config": config,
        "runtime_seconds": round(elapsed_seconds, 6),
        "word_count": len(words),
        "mean_word_confidence": (
            round(
                sum(word["confidence"] for word in words) / len(words),
                4,
            )
            if words
            else None
        ),
        "text": reconstruct_text(words),
        "words": words,
    }
