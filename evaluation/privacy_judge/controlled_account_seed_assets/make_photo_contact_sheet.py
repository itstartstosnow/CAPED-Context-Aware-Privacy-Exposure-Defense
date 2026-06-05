from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


PHOTO_DIR = Path(__file__).resolve().parent / "photos"
OUT = Path(__file__).resolve().parent / "photo_contact_sheet.jpg"
CELL_W, CELL_H = 380, 430


def main() -> None:
    files = sorted(p for p in PHOTO_DIR.glob("*.jpg") if not p.name.startswith("_"))
    sheet = Image.new("RGB", (CELL_W * 3, CELL_H * 3), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, path in enumerate(files):
        img = Image.open(path).convert("RGB")
        img.thumbnail((340, 340))
        x = (idx % 3) * CELL_W + 20
        y = (idx // 3) * CELL_H + 20
        sheet.paste(img, (x, y))
        draw.text((x, y + 350), path.name, fill=(0, 0, 0), font=font)

    sheet.save(OUT, quality=92)
    print(OUT)


if __name__ == "__main__":
    main()
