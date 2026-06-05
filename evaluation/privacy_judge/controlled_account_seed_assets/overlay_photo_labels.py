from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


PHOTO_DIR = Path(__file__).resolve().parent / "photos"

LABELS = {
    "fake_bill_envelope_counter.jpg": ["SAMPLE BILL", "Synthetic finance cue"],
    "fake_lease_folder_table.jpg": ["FICTIONAL LEASE DRAFT", "Sample tenant / address"],
    "fake_passport_cover_table.jpg": ["SAMPLE TRAVEL DOCUMENT", "No real country"],
    "synthetic_bank_card_blurred.jpg": ["SYNTHETIC BANK CARD", "0000 0000 0000 0000"],
    "synthetic_medical_bracelet_drawer.jpg": ["SYNTHETIC MEDICAL ID", "Fake condition cue"],
    "synthetic_prescription_bag_kitchen.jpg": ["SYNTHETIC PRESCRIPTION", "Alex Sample"],
    "target_receipt_cafe.jpg": ["CAFE RECEIPT", "Total 12.40"],
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay(path: Path, lines: list[str]) -> None:
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    w, _ = img.size
    box_h = 230
    draw.rounded_rectangle((40, 40, w - 40, 40 + box_h), radius=28, fill=(255, 255, 255, 232))
    draw.rounded_rectangle((40, 40, w - 40, 40 + box_h), radius=28, outline=(0, 0, 0, 210), width=6)

    title_font = font(64, bold=True)
    sub_font = font(40, bold=True)
    draw.text((80, 78), lines[0], fill=(0, 0, 0), font=title_font)
    if len(lines) > 1:
        draw.text((82, 164), lines[1], fill=(35, 35, 35), font=sub_font)
    img.save(path, quality=94)
    print(path)


def main() -> None:
    for name, lines in LABELS.items():
        overlay(PHOTO_DIR / name, lines)


if __name__ == "__main__":
    main()
