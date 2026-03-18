"""
core/annotationRenderer.py — 标注图生成工具

职责：
  - 在截图上绘制检测框和标签，生成可视化标注图
  - 与业务逻辑完全解耦，只依赖 PIL/io/os/json

对外接口：
  TYPE_COLORS            — 检测类型 → RGB 颜色映射
  loadFont(size)         → ImageFont
  drawBoxes(img, boxes)  → Image    通用框绘制，接受 {bbox, label, color?} 列表
  drawDetections(img, detections) → Image  标准 detection 格式的适配层
  toBase64(img)          → str
  saveImage(img, path)
  saveDebugArtifacts(imagePath, ocrBoxes, compoBoxes, finalDets, debugDir)
"""

import base64
import io
import json
import os

from PIL import Image, ImageDraw, ImageFont


# ── 类型颜色表 ────────────────────────────────────────────────────────────────

TYPE_COLORS: dict = {
    'text':  (100, 149, 237),   # 矢车菊蓝
    'icon':  ( 50, 205,  50),   # 草绿
    'image': (220,  20,  60),   # 深红
    'compo': (180, 180, 180),   # 灰（未分类的原始 UIED compo，debug 用）
}

_DEFAULT_COLOR = (180, 180, 180)


# ── 字体加载（进程内缓存）────────────────────────────────────────────────────

_FONT_CACHE: dict = {}

def loadFont(size: int = 13) -> ImageFont.ImageFont:
    """按 size 加载 CJK 字体，同 size 只加载一次。"""
    if size not in _FONT_CACHE:
        for path in [
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        ]:
            if os.path.exists(path):
                try:
                    _FONT_CACHE[size] = ImageFont.truetype(path, size)
                    break
                except Exception:
                    pass
        else:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


# ── 核心绘制：通用框 + 标签 ──────────────────────────────────────────────────

def drawBoxes(
    img:   Image.Image,
    boxes: list,
    fontSize: int = 13,
) -> Image.Image:
    """通用 bbox 标注，接受任意 {bbox, label, color?} 字典列表。

    Args:
        img:    PIL Image（会在副本上绘制，调用方请先 .copy()）
        boxes:  每项包含：
                  bbox  — [x1,y1,x2,y2] 归一化 [0,1]
                  label — 显示文字（可为空字符串）
                  color — (R,G,B) 可选，默认灰色
        fontSize: 标签字号

    Returns:
        标注后的 PIL Image（原 img 不变）
    """
    draw = ImageDraw.Draw(img)
    w, h = img.size
    font = loadFont(fontSize)

    for box in boxes:
        bbox  = box.get('bbox', [])
        if not bbox or len(bbox) != 4:
            continue
        x1 = int(bbox[0] * w); y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w); y2 = int(bbox[3] * h)
        color = box.get('color', _DEFAULT_COLOR)
        label = str(box.get('label', ''))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if label:
            sz  = draw.textbbox((0, 0), label, font=font)
            lw  = sz[2] - sz[0] + 6
            lh  = sz[3] - sz[1] + 4
            ly  = max(0, y1 - lh - 1)
            draw.rectangle([x1, ly, x1 + lw, ly + lh], fill=color)
            draw.text((x1 + 3, ly + 2), label, fill=(0, 0, 0), font=font)

    return img


# ── 标准 detection 格式适配层 ─────────────────────────────────────────────────

def drawDetections(
    img:        Image.Image,
    detections: list,
    fontSize:   int = 13,
) -> Image.Image:
    """将标准 detection 列表转为 drawBoxes 所需格式并绘制。

    标签格式：'#<image_label_id> <content前缀或type>'
    颜色取自 TYPE_COLORS，未知 type 用默认灰。
    """
    boxes = []
    for det in detections:
        t       = det.get('type', '')
        idx     = det.get('image_label_id', '?')
        # content 取冒号前的 ClassName 部分（如 'ImageView: ...' → 'ImageView'）
        content = (det.get('content') or '').split(':')[0][:14].strip()
        label   = f"#{idx} {content}" if content else f"#{idx} {t}"
        boxes.append({
            'bbox':  det.get('bbox', []),
            'label': label,
            'color': TYPE_COLORS.get(t, _DEFAULT_COLOR),
        })
    return drawBoxes(img, boxes, fontSize=fontSize)


# ── 格式转换工具 ──────────────────────────────────────────────────────────────

def toBase64(img: Image.Image) -> str:
    """PIL Image → base64 PNG 字符串。"""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def saveImage(img: Image.Image, path: str) -> None:
    """保存 PIL Image 到文件，自动创建父目录。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img.save(path)


# ── Debug 输出：三阶段标注图 + raw JSON ──────────────────────────────────────

def saveDebugArtifacts(
    imagePath:  str,
    ocrBoxes:   list,
    compoBoxes: list,
    finalDets:  list,
    debugDir:   str,
) -> None:
    """保存检测流程三阶段的 debug 输出。

    输出文件：
      raw_ocr.json      — OCR 原始结果（含 bbox + content）
      raw_compo.json    — UIED compo 原始结果（含 cnn_class）
      debug_ocr.png     — OCR 检测框标注图（蓝色）
      debug_compo.png   — UIED compo 框标注图（灰色 + cnn_class 标签）
      debug_merged.png  — merge 后最终结果标注图（类型颜色）
    """
    os.makedirs(debugDir, exist_ok=True)
    img = Image.open(imagePath).convert('RGB')
    w, h = img.size

    # ── raw JSON ─────────────────────────────────────────────────────────────
    with open(os.path.join(debugDir, 'raw_ocr.json'), 'w', encoding='utf-8') as f:
        json.dump({'detections': ocrBoxes}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(debugDir, 'raw_compo.json'), 'w', encoding='utf-8') as f:
        json.dump({'detections': compoBoxes}, f, ensure_ascii=False, indent=2)

    # ── OCR 标注图 ────────────────────────────────────────────────────────────
    ocrImg = drawBoxes(img.copy(), [
        {
            'bbox':  o['bbox'],
            'label': f"#{i} {(o.get('content') or '')[:15]}",
            'color': TYPE_COLORS['text'],
        }
        for i, o in enumerate(ocrBoxes)
    ])
    ocrImg.save(os.path.join(debugDir, 'debug_ocr.png'))

    # ── compo 标注图 ──────────────────────────────────────────────────────────
    compoImg = drawBoxes(img.copy(), [
        {
            'bbox':  c['bbox'],
            'label': f"#{i} {c.get('cnn_class', '?')}",
            'color': TYPE_COLORS['compo'],
        }
        for i, c in enumerate(compoBoxes)
    ])
    compoImg.save(os.path.join(debugDir, 'debug_compo.png'))

    # ── 最终 merge 标注图 ─────────────────────────────────────────────────────
    for i, det in enumerate(finalDets):
        det.setdefault('image_label_id', i)
    mergedImg = drawDetections(img.copy(), finalDets)
    mergedImg.save(os.path.join(debugDir, 'debug_merged.png'))

    print(f"[annotationRenderer] debug artifacts → {debugDir}")
