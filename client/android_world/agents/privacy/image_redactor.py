"""Image redaction (masking) utilities for the v3 privacy system.

图像遮蔽规则
============
遮蔽策略：
  hard_mask → 灰蓝色填充覆盖检测区域 + 眼睛图标（居中）
  expose    → 不处理，原样保留

遮蔽流程（两遍 pass）：
  Pass 1: 对所有 hard_mask 区域填充灰蓝色 + 绘制眼睛图标
  Pass 2: 对所有 expose 区域从原始像素恢复（确保上层控件压过背景遮蔽）

无需 GPU，纯 OpenCV/Pillow 操作。
"""

import logging
import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Candidate font paths (tried in order; first hit wins)
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",          # Linux
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",                                  # macOS
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNSText.ttf",
]
_font_cache: dict = {}  # size -> ImageFont, or None


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load a TTF font at the given size, with caching and fallback to default."""
    if size in _font_cache:
        return _font_cache[size]
    font = None
    for path in _FONT_PATHS:
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font

# Solid fill color for hard_mask regions (muted blue-gray, fully opaque)
_MASK_COLOR = (72, 72, 96)
# Border color — slightly lighter same-hue, avoids harsh white contrast
_BORDER_COLOR = (110, 110, 140)

# Eye icon PNG (visiableEye.png — open eye on circular background, most button-like)
_SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
_EYE_ICON_PATH = os.path.join(_SRC_DIR, "visiableEye.png")
_eye_icon_cache: dict = {}  # size -> numpy RGBA array, or None if load failed


def _load_eye_icon(size: int) -> Optional[np.ndarray]:
    """Load and resize visiableEye.png to (size x size) RGBA array.

    Results are cached per size. Returns None on any load error (triggers
    fallback to OpenCV geometric drawing).
    """
    if size in _eye_icon_cache:
        return _eye_icon_cache[size]
    try:
        from PIL import Image  # pylint: disable=import-outside-toplevel
        img = Image.open(_EYE_ICON_PATH).convert("RGBA").resize((size, size), Image.LANCZOS)
        _eye_icon_cache[size] = np.array(img)
        logger.debug("Eye icon loaded at size %d from %s", size, _EYE_ICON_PATH)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to load eye icon from %s (%s); using fallback.", _EYE_ICON_PATH, exc)
        _eye_icon_cache[size] = None
    return _eye_icon_cache[size]


def _bbox_to_pixels(bbox, h: int, w: int):
    """Convert normalized bbox to pixel coordinates, clamped to image bounds."""
    x1, y1, x2, y2 = bbox
    px1 = max(0, int(x1 * w))
    py1 = max(0, int(y1 * h))
    px2 = min(w, int(x2 * w))
    py2 = min(h, int(y2 * h))
    return px1, py1, px2, py2


def _fill_region(result: np.ndarray, bbox, h: int, w: int) -> None:
    """Fill a bbox region with _MASK_COLOR in-place."""
    px1, py1, px2, py2 = _bbox_to_pixels(bbox, h, w)
    if px2 > px1 and py2 > py1:
        result[py1:py2, px1:px2] = _MASK_COLOR


def _restore_region(result: np.ndarray, original: np.ndarray, bbox, h: int, w: int) -> None:
    """Restore a bbox region from original pixels in-place."""
    px1, py1, px2, py2 = _bbox_to_pixels(bbox, h, w)
    if px2 > px1 and py2 > py1:
        result[py1:py2, px1:px2] = original[py1:py2, px1:px2]


def _draw_border(img: np.ndarray, bbox, h: int, w: int) -> None:
    """Draw a border rectangle around the bbox region (2px, muted blue-gray)."""
    px1, py1, px2, py2 = _bbox_to_pixels(bbox, h, w)
    if px2 <= px1 or py2 <= py1:
        return
    cv2.rectangle(img, (px1, py1), (px2 - 1, py2 - 1), _BORDER_COLOR, 2)


def _alpha_composite_icon(img: np.ndarray, icon_rgba: np.ndarray, cx: int, cy: int) -> None:
    """Place RGBA icon centered at (cx, cy) onto BGR img in-place."""
    ih, iw = icon_rgba.shape[:2]
    x1 = cx - iw // 2
    y1 = cy - ih // 2
    x2 = x1 + iw
    y2 = y1 + ih
    # Clamp to image bounds
    ix1 = max(0, -x1)
    iy1 = max(0, -y1)
    ix2 = iw - max(0, x2 - img.shape[1])
    iy2 = ih - max(0, y2 - img.shape[0])
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = x1 + (ix2 - ix1)
    y2 = y1 + (iy2 - iy1)
    if x2 <= x1 or y2 <= y1:
        return
    alpha = icon_rgba[iy1:iy2, ix1:ix2, 3:4].astype(np.float32) / 255.0
    icon_bgr = icon_rgba[iy1:iy2, ix1:ix2, :3][:, :, ::-1]  # RGB → BGR
    bg = img[y1:y2, x1:x2].astype(np.float32)
    img[y1:y2, x1:x2] = (icon_bgr * alpha + bg * (1 - alpha)).astype(np.uint8)


def _draw_eye_icon(img: np.ndarray, bbox, h: int, w: int) -> None:
    """Draw eye icon centered on the bbox.

    Uses visiableEye.png (open eye on circular background) when available —
    most recognizable as a tappable button by vision-language agents.
    Falls back to OpenCV geometric drawing if PNG load fails.
    """
    px1, py1, px2, py2 = _bbox_to_pixels(bbox, h, w)
    if px2 <= px1 or py2 <= py1:
        return

    # Top-right corner, with inset so icon circle clears the 2px border
    icon_size = max(28, min(56, min(px2 - px1, py2 - py1) // 4))
    inset = 10  # gap between icon edge and mask border
    cx = px2 - icon_size // 2 - inset
    cy = py1 + icon_size // 2 + inset

    icon = _load_eye_icon(icon_size)
    if icon is not None:
        _alpha_composite_icon(img, icon, cx, cy)
        return

    # Fallback: OpenCV geometric closed-eye-with-slash (top-right corner)
    r = max(20, min(40, min(px2 - px1, py2 - py1) // 5))
    cx2 = px2 - r - 8
    cy2 = py1 + r + 8
    cv2.ellipse(img, (cx2, cy2), (r, max(1, r // 2)), 0, 0, 360, (255, 255, 255), 2)
    cv2.circle(img, (cx2, cy2), max(2, r // 3), (255, 255, 255), -1)
    cv2.line(img, (cx2 - r, cy2 + r // 2), (cx2 + r, cy2 - r // 2), (255, 255, 255), 2)


def _render_ner_text(result: np.ndarray, original: np.ndarray,
                     det: dict, h: int, w: int) -> None:
    """Overwrite a NER-replaced OCR region with its anonymized placeholder text.

    Appearance goal: look like normal on-screen text so the agent treats it
    naturally — same background as the surrounding UI, dark text on light bg
    (or light on dark), font size matching the detected text height.

    Steps:
      1. Restore the original background pixels (removes upstream mask artifacts).
      2. Sample background colour from the bbox corners of the original image.
      3. Fill the bbox with that background colour (covers the real name).
      4. Choose text colour by luminance contrast against the background.
      5. Auto-size the font to fill the bbox height, then shrink if too wide.
      6. Draw the placeholder centred vertically in the bbox.
    """
    bbox = det.get("bbox", [])
    if len(bbox) != 4:
        return
    px1, py1, px2, py2 = _bbox_to_pixels(bbox, h, w)
    if px2 <= px1 or py2 <= py1:
        return

    placeholder = det.get("content", "")
    if not placeholder:
        # Nothing to render — just restore background so no mask artefacts show
        result[py1:py2, px1:px2] = original[py1:py2, px1:px2]
        return

    box_w = px2 - px1
    box_h = py2 - py1

    # ── 1. Sample background colour from the four corners of the bbox ──────
    # Corners of a text region are typically background pixels.
    corner_coords = [
        (max(0, py1),      max(0, px1)),
        (max(0, py1),      min(w - 1, px2 - 1)),
        (min(h - 1, py2 - 1), max(0, px1)),
        (min(h - 1, py2 - 1), min(w - 1, px2 - 1)),
    ]
    corner_pixels = np.array([original[cy, cx] for cy, cx in corner_coords], dtype=np.float32)
    bg_bgr = np.median(corner_pixels, axis=0)          # (B, G, R)
    bg_rgb = (int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0]))  # PIL uses RGB

    # ── 2. Choose text colour by luminance contrast ─────────────────────────
    lum = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
    text_rgb = (20, 20, 20) if lum > 128 else (235, 235, 235)

    # ── 3. Compute font size: fit bbox height, then shrink to fit width ──────
    # Initial estimate: 65 % of bbox height (matches typical single-line metrics)
    font_size = max(8, int(box_h * 0.65))
    font = _load_font(font_size)

    # Measure rendered width; shrink until it fits within box_w with 4 px padding
    def _text_width(f: ImageFont.ImageFont, text: str) -> int:
        try:
            return int(f.getlength(text))   # Pillow >= 9.2
        except AttributeError:
            return f.getsize(text)[0]       # Pillow < 9.2

    while font_size > 8 and _text_width(font, placeholder) > box_w - 4:
        font_size -= 1
        font = _load_font(font_size)

    # ── 4. Render onto a fresh PIL surface ──────────────────────────────────
    pil_img = Image.new("RGB", (box_w, box_h), bg_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Vertical centre: use getbbox (Pillow >= 8) or getsize fallback
    try:
        tb = font.getbbox(placeholder)
        text_h = tb[3] - tb[1]
        y_off = max(0, (box_h - text_h) // 2 - tb[1])
    except AttributeError:
        text_h = font.getsize(placeholder)[1]
        y_off = max(0, (box_h - text_h) // 2)

    draw.text((2, y_off), placeholder, fill=text_rgb, font=font)

    # ── 5. Write back (RGB → BGR) ───────────────────────────────────────────
    result[py1:py2, px1:px2] = np.array(pil_img)[:, :, ::-1]


def redact_image(pixels: np.ndarray, detections: list) -> np.ndarray:
    """Apply privacy redaction to an image based on detection results.

    Two-pass rendering:
      Pass 1: Fill all hard_mask regions with solid color + draw eye icon.
      Pass 2: Restore all expose regions from original pixels (ensures that
              overlapping expose elements always show through background masks).

    Args:
        pixels: RGB numpy array (H, W, 3) of the original screenshot.
        detections: List of detection dicts, each containing:
            - "bbox": [x1, y1, x2, y2] in normalized coordinates [0.0, 1.0]
            - "privacy_strategy": "hard_mask" | "expose"

    Returns:
        RGB numpy array of the same shape with protected regions masked.
    """
    if not detections:
        return pixels

    result = pixels.copy()
    h, w = pixels.shape[:2]

    # Pass 1: Mask all hard_mask regions (solid fill + border + eye icon)
    for det in detections:
        if det.get("privacy_strategy") != "hard_mask":
            continue
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            logger.debug("Skipping detection with invalid bbox: %s", bbox)
            continue
        _fill_region(result, bbox, h, w)
        _draw_border(result, bbox, h, w)
        _draw_eye_icon(result, bbox, h, w)

    # Pass 2: Restore expose and ner_replace regions from original pixels.
    # ner_replace regions need their background restored before text is drawn in Pass 3.
    for det in detections:
        if det.get("privacy_strategy") not in ("expose", "ner_replace"):
            continue
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        _restore_region(result, pixels, bbox, h, w)

    # Pass 3: Render anonymized placeholder text for NER-replaced elements.
    # Runs after Pass 2 so text is drawn on the restored background, not on masks.
    for det in detections:
        if det.get("privacy_strategy") != "ner_replace":
            continue
        _render_ner_text(result, pixels, det, h, w)

    return result
