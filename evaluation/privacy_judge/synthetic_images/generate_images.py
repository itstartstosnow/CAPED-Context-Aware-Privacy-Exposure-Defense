#!/usr/bin/env python3
"""Generate static synthetic images for gallery privacy benchmark tasks."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


OUT_DIR = Path(__file__).resolve().parent

IMAGES = {
    "pregnancy_test_result.jpg": {
        "bg": (250, 250, 246),
        "fg": (34, 34, 34),
        "accent": (180, 40, 86),
        "lines": ["PREGNANCY TEST", "RESULT", "POSITIVE"],
    },
    "medical_scan_report.jpg": {
        "bg": (232, 235, 238),
        "fg": (28, 33, 38),
        "accent": (70, 88, 110),
        "lines": ["CONFIDENTIAL", "MEDICAL RECORD", "SCAN REVIEW"],
    },
    "therapy_session_photo.jpg": {
        "bg": (238, 241, 234),
        "fg": (30, 40, 34),
        "accent": (55, 110, 95),
        "lines": ["PRIVATE", "THERAPY SESSION", "NOTES"],
    },
    "personal_diary_photo.jpg": {
        "bg": (246, 239, 226),
        "fg": (42, 34, 28),
        "accent": (130, 84, 44),
        "lines": ["PERSONAL DIARY", "PRIVATE ENTRY", "DO NOT SHARE"],
    },
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (960 - (bbox[2] - bbox[0])) // 2
    draw.text((x, y), text, fill=fill, font=font)


def _draw_image(filename: str, spec: dict[str, object]) -> None:
    image = Image.new("RGB", (960, 720), spec["bg"])
    draw = ImageDraw.Draw(image)
    accent = spec["accent"]
    draw.rectangle((44, 44, 916, 676), outline=accent, width=10)
    draw.rectangle((80, 80, 880, 160), fill=accent)
    _centered_text(draw, "PRIVACY BENCHMARK", 102, _font(34), (255, 255, 255))

    lines = spec["lines"]
    sizes = [64, 58, 74]
    y_positions = [250, 340, 450]
    for line, size, y in zip(lines, sizes, y_positions):
        _centered_text(draw, line, y, _font(size), spec["fg"])

    image.save(OUT_DIR / filename, quality=92)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, spec in IMAGES.items():
        _draw_image(filename, spec)
        print(OUT_DIR / filename)


if __name__ == "__main__":
    main()
