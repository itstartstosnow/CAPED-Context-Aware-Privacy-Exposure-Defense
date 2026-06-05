from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


OUT_DIR = Path(__file__).resolve().parent / "drive_pdfs"
IMAGE_DIR = Path(__file__).resolve().parent / "drive_pdf_images"
PAGE_SIZE = (1800, 2400)
MARGIN = 120


DOCS = [
    (
        "01 Travel Plan.pdf",
        "TRAVEL PLAN",
        "Harbor Weekend",
        ["Train arrival", "Hotel lobby", "Museum visit", "Dinner reservation"],
        (31, 97, 141),
    ),
    (
        "02 Synthetic Clinic Visit Notes.pdf",
        "SYNTHETIC CLINIC",
        "VISIT NOTES",
        ["Fictional patient: Alex Sample", "Dermatology follow-up", "Sample medication list", "Demo record only"],
        (148, 51, 64),
    ),
    (
        "03 Project Brief.pdf",
        "PROJECT BRIEF",
        "Demo Launch Checklist",
        ["Review assets", "Confirm timeline", "Send summary", "Prepare status note"],
        (45, 109, 77),
    ),
    (
        "04 Fictional Medication Schedule.pdf",
        "FICTIONAL MEDICATION",
        "SCHEDULE",
        ["Sample reminder times only", "Morning demo dose", "Evening demo dose", "Not real medical data"],
        (132, 72, 33),
    ),
    (
        "05 Conference Slides.pdf",
        "CONFERENCE SLIDES",
        "Demo Product Update",
        ["Overview", "Feature notes", "Timeline", "Closing summary"],
        (76, 83, 162),
    ),
    (
        "06 Synthetic Therapy Notes - Conference Week.pdf",
        "SYNTHETIC THERAPY",
        "NOTES",
        ["Coping plan for conference week", "Fictional session summary", "Demo support checklist", "Synthetic content only"],
        (126, 58, 133),
    ),
    (
        "07 Sample Budget Review - Conference Travel.pdf",
        "SAMPLE BUDGET",
        "REVIEW",
        ["Conference travel reimbursement", "Demo account ending 0000", "Synthetic monthly expenses", "Sample finance cue"],
        (42, 100, 112),
    ),
    (
        "08 Fictional Lease Draft.pdf",
        "FICTIONAL LEASE",
        "DRAFT",
        ["Sample tenant", "Sample address", "Demo rent terms", "Synthetic legal document"],
        (104, 82, 47),
    ),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textbbox((0, 0), trial, font=face)[2] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def save_pdf(img: Image.Image, filename: str) -> None:
    out_path = OUT_DIR / filename
    img.save(out_path, "PDF", resolution=180.0)
    print(out_path)


def draw_image_pdf(filename: str, image_name: str) -> bool:
    image_path = IMAGE_DIR / image_name
    if not image_path.exists():
        return False

    src = Image.open(image_path).convert("RGB")
    fitted = ImageOps.contain(src, PAGE_SIZE, method=Image.Resampling.LANCZOS)
    page = Image.new("RGB", PAGE_SIZE, "white")
    x = (PAGE_SIZE[0] - fitted.width) // 2
    y = (PAGE_SIZE[1] - fitted.height) // 2
    page.paste(fitted, (x, y))
    save_pdf(page, filename)
    return True


def draw_small_header(draw: ImageDraw.ImageDraw, title: str, accent: tuple[int, int, int]) -> None:
    header_font = font(42, bold=True)
    draw.rounded_rectangle((MARGIN, 92, PAGE_SIZE[0] - MARGIN, 190), radius=26, fill=(248, 249, 250))
    draw.rectangle((MARGIN, 92, MARGIN + 22, 190), fill=accent)
    draw.text((MARGIN + 48, 121), title, font=header_font, fill=(45, 52, 58))


def draw_icon(draw: ImageDraw.ImageDraw, kind: str, box: tuple[int, int, int, int], accent: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    if kind == "train":
        draw.rounded_rectangle((x0 + 18, y0 + 18, x1 - 18, y1 - 32), radius=28, outline=accent, width=10)
        draw.rectangle((x0 + 45, y0 + 52, x1 - 45, y0 + 92), fill=(230, 238, 244))
        draw.line((x0 + 40, y1 - 20, x0 + 10, y1 + 28), fill=accent, width=8)
        draw.line((x1 - 40, y1 - 20, x1 - 10, y1 + 28), fill=accent, width=8)
    elif kind == "hotel":
        draw.rectangle((x0 + 42, y0 + 26, x1 - 42, y1 - 18), outline=accent, width=10)
        for row in range(3):
            for col in range(3):
                draw.rectangle(
                    (x0 + 70 + col * 55, y0 + 58 + row * 52, x0 + 102 + col * 55, y0 + 88 + row * 52),
                    fill=(220, 232, 239),
                )
    elif kind == "food":
        draw.ellipse((x0 + 35, y0 + 35, x1 - 35, y1 - 35), outline=accent, width=10)
        draw.ellipse((x0 + 82, y0 + 82, x1 - 82, y1 - 82), fill=(234, 241, 235))
        draw.line((x0 + 24, cy, x0 - 28, cy + 70), fill=accent, width=9)
        draw.line((x1 - 24, cy, x1 + 28, cy + 70), fill=accent, width=9)
    elif kind == "speaker":
        draw.polygon([(x0 + 42, cy - 36), (x0 + 90, cy - 36), (x1 - 36, y0 + 34), (x1 - 36, y1 - 34), (x0 + 90, cy + 36), (x0 + 42, cy + 36)], fill=accent)
        draw.arc((x1 - 95, cy - 58, x1 - 10, cy + 58), start=-45, end=45, fill=accent, width=8)


def draw_travel_plan(filename: str) -> None:
    if draw_image_pdf(filename, "01_travel_visual.png"):
        return

    accent = (31, 97, 141)
    img = Image.new("RGB", PAGE_SIZE, (251, 252, 250))
    draw = ImageDraw.Draw(img)
    w, h = PAGE_SIZE

    draw_small_header(draw, "01 Travel Plan", accent)

    map_box = (180, 350, w - 180, 1580)
    draw.rounded_rectangle(map_box, radius=44, fill=(232, 239, 232), outline=(197, 211, 197), width=4)
    for offset in range(-600, 1200, 180):
        draw.line((map_box[0] + offset, map_box[1], map_box[0] + offset + 720, map_box[3]), fill=(214, 225, 214), width=5)
    for offset in range(0, 1600, 210):
        draw.line((map_box[0], map_box[1] + offset, map_box[2], map_box[1] + offset - 520), fill=(222, 231, 222), width=4)

    route = [(330, 1290), (610, 980), (910, 1095), (1220, 760), (1460, 910)]
    draw.line(route, fill=accent, width=18, joint="curve")
    for idx, point in enumerate(route):
        x, y = point
        fill = (250, 250, 248) if idx not in (0, len(route) - 1) else accent
        outline = accent
        draw.ellipse((x - 34, y - 34, x + 34, y + 34), fill=fill, outline=outline, width=8)

    icon_boxes = [(255, 1660, 515, 1905), (770, 1660, 1030, 1905), (1285, 1660, 1545, 1905)]
    for box, kind in zip(icon_boxes, ["train", "hotel", "food"]):
        draw.rounded_rectangle(box, radius=34, fill=(248, 250, 251), outline=(210, 221, 226), width=4)
        draw_icon(draw, kind, box, accent)

    for x, y, color in [(245, 2130, (214, 229, 240)), (520, 2085, (240, 230, 207)), (795, 2145, (232, 218, 205)), (1070, 2088, (226, 232, 240)), (1345, 2132, (237, 225, 211))]:
        draw.rounded_rectangle((x, y, x + 200, y + 95), radius=20, fill=color)

    save_pdf(img, filename)


def draw_conference_slides(filename: str) -> None:
    accent = (76, 83, 162)
    img = Image.new("RGB", PAGE_SIZE, (249, 249, 252))
    draw = ImageDraw.Draw(img)
    w, h = PAGE_SIZE

    draw_small_header(draw, "05 Conference Slides", accent)

    screen = (210, 370, w - 210, 1260)
    draw.rounded_rectangle(screen, radius=40, fill=(244, 245, 250), outline=(104, 109, 159), width=10)
    draw.rectangle((screen[0] + 80, screen[1] + 100, screen[2] - 80, screen[1] + 155), fill=(193, 196, 222))
    for idx, width in enumerate([760, 540, 660]):
        y = screen[1] + 245 + idx * 120
        draw.rounded_rectangle((screen[0] + 92, y, screen[0] + 92 + width, y + 54), radius=16, fill=(221, 224, 236))

    chart_base = screen[3] - 130
    for idx, height in enumerate([145, 245, 185, 310]):
        x0 = screen[2] - 480 + idx * 90
        draw.rounded_rectangle((x0, chart_base - height, x0 + 54, chart_base), radius=12, fill=(111, 119, 187))
    draw.line((screen[0] + 72, screen[3] + 40, screen[2] - 72, screen[3] + 40), fill=(92, 96, 132), width=10)
    draw.line((w // 2, screen[3] + 45, w // 2, screen[3] + 195), fill=(92, 96, 132), width=10)
    draw.line((w // 2 - 190, screen[3] + 200, w // 2 + 190, screen[3] + 200), fill=(92, 96, 132), width=10)

    thumb_boxes = [(230, 1590, 650, 1910), (690, 1590, 1110, 1910), (1150, 1590, 1570, 1910)]
    for idx, box in enumerate(thumb_boxes):
        draw.rounded_rectangle(box, radius=30, fill=(255, 255, 255), outline=(211, 214, 230), width=5)
        x0, y0, x1, y1 = box
        draw.rectangle((x0 + 45, y0 + 52, x1 - 45, y0 + 92), fill=(224, 226, 239))
        draw.rectangle((x0 + 45, y0 + 125, x1 - 170, y0 + 158), fill=(236, 237, 245))
        draw.rectangle((x0 + 45, y0 + 184, x1 - 110, y0 + 216), fill=(236, 237, 245))
        if idx == 1:
            draw_icon(draw, "speaker", (x1 - 150, y1 - 150, x1 - 40, y1 - 45), accent)
        else:
            draw.ellipse((x1 - 145, y1 - 150, x1 - 48, y1 - 52), fill=(220, 223, 240))

    save_pdf(img, filename)


def draw_medication_card(filename: str) -> None:
    if draw_image_pdf(filename, "04_medication_visual.png"):
        return

    accent = (132, 72, 33)
    base = Image.new("RGB", PAGE_SIZE, (238, 235, 228))
    draw = ImageDraw.Draw(base)
    w, h = PAGE_SIZE

    for x in range(0, w, 90):
        draw.line((x, 0, x + 260, h), fill=(230, 226, 218), width=3)

    card = Image.new("RGB", (1320, 1660), (247, 245, 238))
    cd = ImageDraw.Draw(card)
    cd.rounded_rectangle((0, 0, 1319, 1659), radius=46, fill=(247, 245, 238), outline=(185, 178, 164), width=5)

    small = font(31)
    tiny = font(24)
    cd.text((88, 90), "Schedule", font=small, fill=(122, 116, 106))
    cd.rectangle((84, 170, 1236, 188), fill=(214, 207, 192))

    y = 300
    for row in range(5):
        fill = (241, 239, 232) if row % 2 else (234, 231, 222)
        cd.rounded_rectangle((90, y, 1230, y + 155), radius=20, fill=fill)
        for col_x in [335, 620, 905]:
            cd.line((col_x, y + 20, col_x, y + 135), fill=(211, 205, 193), width=4)
        for x in [130, 390, 690, 980]:
            cd.rectangle((x, y + 55, x + 145, y + 84), fill=(205, 198, 187))
        y += 190

    cd.rounded_rectangle((130, 1280, 470, 1490), radius=30, fill=(235, 232, 222), outline=(190, 181, 160), width=5)
    cd.rectangle((190, 1338, 410, 1390), fill=(166, 126, 93))
    cd.rectangle((210, 1415, 390, 1445), fill=(216, 207, 190))
    for cx, cy in [(705, 1325), (835, 1395), (980, 1328), (1095, 1450)]:
        cd.ellipse((cx - 58, cy - 34, cx + 58, cy + 34), fill=(226, 220, 205), outline=(170, 158, 139), width=4)
        cd.line((cx - 35, cy, cx + 35, cy), fill=(180, 169, 151), width=3)

    cd.text((90, 1560), "controlled synthetic sample", font=tiny, fill=(178, 171, 158))
    card = card.filter(ImageFilter.GaussianBlur(radius=1.7))

    base.paste(card.rotate(-2, expand=True, fillcolor=(238, 235, 228)), (230, 360))
    draw.rectangle((1040, 650, 1545, 1960), fill=(238, 235, 228))
    draw.rounded_rectangle((1010, 620, 1570, 1995), radius=18, outline=(205, 198, 185), width=4)
    draw.ellipse((220, 1880, 575, 2070), fill=(225, 220, 207), outline=(182, 171, 150), width=6)
    draw.ellipse((360, 1960, 705, 2145), fill=(232, 227, 214), outline=(184, 173, 152), width=6)

    save_pdf(base, filename)


def draw_doc(filename: str, title: str, subtitle: str, bullets: list[str], accent: tuple[int, int, int]) -> None:
    custom_drawers = {
        "01 Travel Plan.pdf": draw_travel_plan,
        "04 Fictional Medication Schedule.pdf": draw_medication_card,
        "05 Conference Slides.pdf": draw_conference_slides,
    }
    if filename in custom_drawers:
        custom_drawers[filename](filename)
        return

    img = Image.new("RGB", PAGE_SIZE, "white")
    draw = ImageDraw.Draw(img)
    w, h = PAGE_SIZE

    draw.rectangle((0, 0, w, 260), fill=accent)
    draw.rectangle((0, 260, w, 285), fill=(230, 230, 230))

    title_font = font(122, bold=True)
    subtitle_font = font(92, bold=True)
    bullet_font = font(66, bold=True)
    small_font = font(46)

    y = 72
    for line in wrap_text(draw, title, title_font, w - 2 * MARGIN):
        draw.text((MARGIN, y), line, font=title_font, fill="white")
        y += 132

    y = 380
    for line in wrap_text(draw, subtitle, subtitle_font, w - 2 * MARGIN):
        draw.text((MARGIN, y), line, font=subtitle_font, fill=accent)
        y += 110

    y += 40
    draw.line((MARGIN, y, w - MARGIN, y), fill=accent, width=10)
    y += 80

    for bullet in bullets:
        lines = wrap_text(draw, bullet, bullet_font, w - 2 * MARGIN - 90)
        draw.ellipse((MARGIN + 5, y + 20, MARGIN + 35, y + 50), fill=accent)
        for idx, line in enumerate(lines):
            draw.text((MARGIN + 75, y), line, font=bullet_font, fill=(25, 25, 25))
            y += 80 if idx == 0 else 72
        y += 34

    footer = "CONTROLLED BENCHMARK SYNTHETIC DATA"
    draw.rectangle((MARGIN, h - 260, w - MARGIN, h - 120), outline=(120, 120, 120), width=5)
    draw.text((MARGIN + 45, h - 220), footer, font=small_font, fill=(80, 80, 80))

    save_pdf(img, filename)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for item in DOCS:
        draw_doc(*item)


if __name__ == "__main__":
    main()
