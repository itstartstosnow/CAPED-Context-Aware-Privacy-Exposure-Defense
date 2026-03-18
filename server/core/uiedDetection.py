"""
core/uiedDetection.py — UI 检测流水线

流程（detectUI）：
  1. _runOCR()           — PaddleOCR，获取 text 元素
  2. _runUIEDCompo()     — UIED 组件检测（步骤2 only），中性 compo + cnn_class
  3. _customMerge()      — 自定义合并 + CNN 分类 + Big Image Rule
  4. _generateCaptions() — Florence-2 对 image 元素生成描述
  5. 编号 + 标注图生成（via annotationRenderer）

输出 detection 格式：
    {
        'type':           'text' | 'icon' | 'image',
        'bbox':           [x1, y1, x2, y2],   # 归一化坐标 [0,1]
        'content':        str | None,
                          # text  → OCR 文字内容
                          # icon  → CNN 类名（如 'ImageButton'）
                          # image → '<ClassName>: <Florence caption>'
                          #         或 'BigImage: <Florence caption>'（大图兜底）
        'source':         'ocr' | 'uied',
        'image_label_id': int,
    }

debug 模式（detectUI(debug=True, debugDir=...)）：
  raw_ocr.json / raw_compo.json / debug_ocr.png / debug_compo.png / debug_merged.png
"""

import json
import os
import sys
import shutil
import tempfile
import time

import cv2
import torch
from PIL import Image

# ── 路径初始化 ────────────────────────────────────────────────────────────────
_project_root    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_uied_root       = os.path.join(_project_root, 'UIED')
for _p in [_project_root, _uied_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import detect_compo.ip_region_proposal as _ipModule
import detect_text.text_detection       as _textModule
import detect_merge.merge               as _mergeModule

from uiedTools import getPaddleOCR, getCNNClassifier, cnnClassToType, UIED_PARAMS
from core.annotationRenderer import drawDetections, toBase64, saveDebugArtifacts


# ── Step 1: OCR ──────────────────────────────────────────────────────────────

def _runOCR(imagePath: str) -> list:
    """PaddleOCR 直接调用，返回归一化 text 元素列表。

    绕过 UIED 内部 text_detection.py + refine_elements，避免文本被大组件吞噬。
    """
    paddle = getPaddleOCR()
    if paddle is None:
        print("[_runOCR] PaddleOCR 加载失败，跳过 OCR")
        return []

    img_pil = Image.open(imagePath)
    w, h    = img_pil.size

    try:
        raw = paddle.ocr(imagePath, cls=True)
    except Exception as e:
        print(f"[_runOCR] 调用失败: {e}")
        return []

    results = []
    if not raw or not raw[0]:
        return results

    for line in raw[0]:
        try:
            box_pts, (text, conf) = line
        except Exception:
            continue
        if not text.strip() or conf < 0.5:
            continue
        xs = [p[0] for p in box_pts]
        ys = [p[1] for p in box_pts]
        results.append({
            'type':    'text',
            'bbox':    [min(xs)/w, min(ys)/h, max(xs)/w, max(ys)/h],
            'content': text.strip(),
            'source':  'ocr',
        })

    print(f"[_runOCR] {len(results)} text elements")
    return results


# ── Step 2: UIED 组件检测 ─────────────────────────────────────────────────────

def _runUIEDCompo(imagePath: str) -> list:
    """UIED 步骤2 only（组件检测），跳过内部 OCR 和 merge。

    返回元素 type='compo'（中性），含 cnn_class 字段供 _customMerge 末尾分类使用。
    """
    tmpDir = tempfile.mkdtemp(prefix='uied_compo_')
    try:
        img = cv2.imread(imagePath)
        if img is None:
            return []
        origH, origW = img.shape[:2]
        imgName      = os.path.splitext(os.path.basename(imagePath))[0]

        _ipModule.compo_detection(
            input_img_path=imagePath,
            output_root=tmpDir,
            uied_params=UIED_PARAMS,
            resize_by_height=800,
            classifier={'Elements': getCNNClassifier()},
            show=False,
        )

        ipJson = os.path.join(tmpDir, 'ip', imgName + '.json')
        if not os.path.exists(ipJson):
            print("[_runUIEDCompo] IP JSON not found")
            return []

        with open(ipJson) as f:
            ipData = json.load(f)

        scaleW = origW / int(origW * 800 / origH)
        scaleH = origH / 800

        results = []
        for compo in ipData.get('compos', []):
            x1 = int(compo.get('column_min', 0) * scaleW)
            y1 = int(compo.get('row_min',    0) * scaleH)
            x2 = int(compo.get('column_max', 0) * scaleW)
            y2 = int(compo.get('row_max',    0) * scaleH)
            results.append({
                'type':      'compo',
                'bbox':      [x1/origW, y1/origH, x2/origW, y2/origH],
                'cnn_class': compo.get('class', 'Compo'),
                'content':   None,
                'source':    'uied',
            })

        print(f"[_runUIEDCompo] {len(results)} component elements")
        return results
    finally:
        shutil.rmtree(tmpDir, ignore_errors=True)


# ── Step 3: 自定义合并 ────────────────────────────────────────────────────────

def _customMerge(ocrBoxes: list, compoBoxes: list, imgSize: tuple) -> list:
    """自定义合并。

    输入 compoBoxes type='compo'（中性），本函数末尾完成 CNN 分类和 Big Image Rule。

    规则顺序：
      [Status filter] 状态栏顶部 4% 的 CJK OCR 过滤
      [Rule 2] compo 被单个 OCR 覆盖 ≥60% → 删 compo
      [Rule 3] compo-in-compo 吸收（absorber 面积<0.10）
      [Rule 4] 小 compo（area<0.05）内部 OCR 压制
      [Rule 5] compo 被 OCR 总覆盖 >60% → 删 compo
      [Step 6] CNN 分类：cnn_class → type='icon'|'image'
      [Step 7] Big Image Rule：icon 窄边>0.2 且长边≥窄边/4 → 'image'，content='BigImage'

    输出类型：'text' | 'icon' | 'image'
    """
    def _area(b):
        return max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

    def _inter(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        return max(0.0, x2-x1) * max(0.0, y2-y1)

    def _contained(inner, outer):
        a = _area(inner)
        return _inter(inner, outer) / a if a > 0 else 0.0

    def _hasCJK(text):
        return any('\u4e00' <= c <= '\u9fff' or
                   '\u3000' <= c <= '\u303f' or
                   '\uff00' <= c <= '\uffef' for c in text)

    compos     = [dict(c) for c in compoBoxes]
    ocrs       = [dict(o) for o in ocrBoxes]
    compos_del: set = set()
    ocrs_del:   set = set()

    # Status filter
    ocrs = [o for o in ocrs
            if not ((o['bbox'][1]+o['bbox'][3])/2 < 0.04 and _hasCJK(o['content']))]
    print(f"[_customMerge] after status-bar filter: {len(ocrs)} OCR items")

    # Rule 2
    for ci, c in enumerate(compos):
        if any(_contained(c['bbox'], o['bbox']) >= 0.60 for o in ocrs):
            compos_del.add(ci)
    compos = [c for i,c in enumerate(compos) if i not in compos_del]; compos_del.clear()

    # Rule 3
    areas = [_area(c['bbox']) for c in compos]
    for si in range(len(compos)):
        if si in compos_del: continue
        for li in range(len(compos)):
            if li == si or li in compos_del: continue
            if areas[li] <= areas[si] or areas[li] >= 0.10: continue
            if _contained(compos[si]['bbox'], compos[li]['bbox']) >= 0.80:
                compos_del.add(si); break
    compos = [c for i,c in enumerate(compos) if i not in compos_del]; compos_del.clear()

    # Rule 4
    for c in compos:
        if _area(c['bbox']) >= 0.05: continue
        bot = c['bbox'][3]
        for oi, o in enumerate(ocrs):
            if oi not in ocrs_del and _contained(o['bbox'], c['bbox']) >= 0.80 and o['bbox'][1] < bot - 0.01:
                ocrs_del.add(oi)
    ocrs = [o for i,o in enumerate(ocrs) if i not in ocrs_del]; ocrs_del.clear()

    # Rule 5
    for ci, c in enumerate(compos):
        ca = _area(c['bbox'])
        if ca > 0 and sum(_inter(c['bbox'], o['bbox']) for o in ocrs) / ca > 0.6:
            compos_del.add(ci)
    compos = [c for i,c in enumerate(compos) if i not in compos_del]

    # Step 6: CNN 分类
    for c in compos:
        cls = c.pop('cnn_class', 'Compo')
        t   = cnnClassToType(cls)
        c['type']    = t
        c['content'] = cls  # for both icon and image; Florence will prepend "ClassName: caption"

    # Step 7: Big Image Rule
    for c in compos:
        if c['type'] == 'icon':
            x1, y1, x2, y2 = c['bbox']
            narrow = min(x2-x1, y2-y1)
            wide   = max(x2-x1, y2-y1)
            if narrow > 0.20 and wide >= narrow / 4:
                c['type']    = 'image'
                c['content'] = 'BigImage'

    result = ocrs + compos

    # Step 8: Background Image Rule
    # A background image is a large image (≥40% of screen) with ≥4 other
    # elements' centers inside it and no strictly larger element containing it.
    for elem in result:
        if elem['type'] != 'image':
            continue
        bb = elem.get('bbox', [])
        if len(bb) != 4:
            continue
        x1, y1, x2, y2 = bb
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < 0.40:
            continue
        overlap = sum(
            1 for other in result
            if other is not elem
            and len(other.get('bbox', [])) == 4
            and x1 <= (other['bbox'][0] + other['bbox'][2]) / 2 <= x2
            and y1 <= (other['bbox'][1] + other['bbox'][3]) / 2 <= y2
        )
        if overlap < 4:
            continue
        if any(
            len(o.get('bbox', [])) == 4
            and o['bbox'][0] <= x1 and o['bbox'][1] <= y1
            and o['bbox'][2] >= x2 and o['bbox'][3] >= y2
            and (o['bbox'][2]-o['bbox'][0])*(o['bbox'][3]-o['bbox'][1]) > area
            for o in result if o is not elem
        ):
            continue
        elem['content'] = 'BackgroundImage'
    cnts = {}
    for d in result:
        cnts[d['type']] = cnts.get(d['type'], 0) + 1
    print(f"[_customMerge] total={len(result)}: " +
          ", ".join(f"{t}={n}" for t,n in sorted(cnts.items())))
    return result



# ── Step 3b: paragraph merging ────────────────────────────────────────────

def _merge_text_paragraphs(detections, gap_ratio=0.75):
    """Merge adjacent OCR text lines into paragraph-level detections.

    Merge conditions (both required for adjacent lines):
      1. Vertical gap < previous line height * gap_ratio
      2. X-ranges overlap (prevents merging side-by-side tabs/buttons)
      3. No non-text element overlaps the vertical gap within the group x-span
    """
    text_dets = [
        d for d in detections
        if d.get("source") == "ocr" and d.get("type") == "text"
        and len(d.get("bbox", [])) == 4
    ]
    other_dets = [
        d for d in detections
        if not (d.get("source") == "ocr" and d.get("type") == "text")
    ]

    if not text_dets:
        return detections

    text_sorted = sorted(text_dets, key=lambda d: d["bbox"][1])

    groups = []
    current_group = [text_sorted[0]]

    for det in text_sorted[1:]:
        last = current_group[-1]
        lb = last["bbox"]
        cb = det["bbox"]

        last_height = max(lb[3] - lb[1], 1e-6)
        gap = cb[1] - lb[3]

        # Condition 1: gap threshold
        if gap >= last_height * gap_ratio:
            groups.append(current_group)
            current_group = [det]
            continue

        # Condition 2: x-range overlap (prevents side-by-side merges)
        gx1 = min(d["bbox"][0] for d in current_group)
        gx2 = max(d["bbox"][2] for d in current_group)
        if cb[2] <= gx1 or cb[0] >= gx2:
            groups.append(current_group)
            current_group = [det]
            continue

        # Condition 3: no interleaving non-text element in column
        gap_y1, gap_y2 = min(lb[3], cb[1]), max(lb[3], cb[1])
        has_interleave = any(
            ob["bbox"][1] < gap_y2 and ob["bbox"][3] > gap_y1
            and ob["bbox"][0] < gx2 and ob["bbox"][2] > gx1
            for ob in other_dets
            if len(ob.get("bbox", [])) == 4
        )

        if has_interleave:
            groups.append(current_group)
            current_group = [det]
        else:
            current_group.append(det)

    groups.append(current_group)

    merged_text_dets = []
    for group in groups:
        if len(group) == 1:
            merged_text_dets.append(group[0])
        else:
            bboxes = [d["bbox"] for d in group]
            merged_bbox = [
                min(b[0] for b in bboxes),
                min(b[1] for b in bboxes),
                max(b[2] for b in bboxes),
                max(b[3] for b in bboxes),
            ]
            merged_content = " ".join(d.get("content", "") for d in group)
            merged_det = dict(group[0])
            merged_det["bbox"] = merged_bbox
            merged_det["content"] = merged_content
            # Keep track of original lines for debug (optional)
            merged_det["merged_lines"] = len(group)
            merged_text_dets.append(merged_det)

    all_dets = merged_text_dets + other_dets
    all_dets.sort(key=lambda d: d["bbox"][1] if len(d.get("bbox", [])) == 4 else 0.0)
    return all_dets

# ── Step 4: Florence-2 图像描述 ──────────────────────────────────────────────

_FLORENCE_MODEL: dict | None = None
_FLORENCE_LOAD_FAILED: bool = False  # sentinel: stop retrying after first failure

def _getFlorenceModel(weightsDir: str) -> dict | None:
    """加载并缓存 Florence-2 模型（全局单例）。"""
    global _FLORENCE_MODEL, _FLORENCE_LOAD_FAILED
    if _FLORENCE_LOAD_FAILED:
        return None
    if _FLORENCE_MODEL is None:
        modelPath = os.path.join(weightsDir, 'icon_caption_florence')
        if not os.path.isdir(modelPath):
            print(f"[Florence] 模型目录不存在: {modelPath}，跳过描述生成")
            _FLORENCE_LOAD_FAILED = True
            return None
        try:
            from util.utils import get_caption_model_processor
            print("[Florence] 正在加载模型...")
            _FLORENCE_MODEL = get_caption_model_processor(
                model_name="florence2", model_name_or_path=modelPath)
            print("[Florence] ✓ 模型已就绪")
        except Exception as e:
            print(f"[Florence] 加载失败: {e}")
            _FLORENCE_LOAD_FAILED = True
    return _FLORENCE_MODEL


def _generateCaptions(
    imagePath:   str,
    imageDets:   list,
    weightsDir:  str,
    batchSize:   int   = 5,
    expandRatio: float = 0.1,
) -> None:
    """对 image 类型元素裁剪并用 Florence-2 生成描述，原地更新 content。

    content 格式：'<ClassName>: <caption>'
      ImageView → 'ImageView: a calendar app icon'
      BigImage  → 'BigImage: a large background image'
    """
    if not imageDets:
        return
    modelPkg = _getFlorenceModel(weightsDir)
    if modelPkg is None:
        return

    model     = modelPkg['model']
    processor = modelPkg['processor']

    try:
        img        = Image.open(imagePath).convert('RGB')
        imgW, imgH = img.size
        crops, indices = [], []

        for i, det in enumerate(imageDets):
            b = det.get('bbox')
            if not b or len(b) != 4:
                continue
            x1a = b[0]*imgW; y1a = b[1]*imgH
            x2a = b[2]*imgW; y2a = b[3]*imgH
            pw  = (x2a-x1a)*expandRatio; ph = (y2a-y1a)*expandRatio
            crop = img.crop((max(0,int(x1a-pw)), max(0,int(y1a-ph)),
                             min(imgW,int(x2a+pw)), min(imgH,int(y2a+ph))))
            if crop.width < 5 or crop.height < 5:
                continue
            crops.append(crop); indices.append(i)

        if not crops:
            return
        print(f"[Florence] 批量描述 {len(crops)} 个 image 元素 (batch={batchSize})...")

        for start in range(0, len(crops), batchSize):
            batch = crops[start:start+batchSize]
            bidx  = indices[start:start+batchSize]
            try:
                inputs = processor(text=["<CAPTION>"]*len(batch), images=batch,
                                   return_tensors="pt", padding=True)
                inputs = {k: (v.to(model.device).to(torch.float16)
                              if v.dtype == torch.float32 else v.to(model.device))
                          for k, v in inputs.items()}
                gen_ids = model.generate(input_ids=inputs["input_ids"],
                                         pixel_values=inputs["pixel_values"],
                                         max_new_tokens=64, num_beams=3, do_sample=False)
                for j, caption in enumerate(processor.batch_decode(gen_ids, skip_special_tokens=True)):
                    det       = imageDets[bidx[j]]
                    className = det.get('content') or 'image'
                    caption   = caption.strip()
                    det['content'] = f"{className}: {caption}" if caption else className
            except Exception as e:
                print(f"[Florence] batch {start//batchSize} 失败: {e}")
    except Exception as e:
        print(f"[Florence] 描述生成失败: {e}")


# ── Step 3c: overlapping bbox merge ──────────────────────────────────────────

def _merge_overlapping_text(detections):
    """Merge text elements whose bounding boxes overlap in BOTH x and y.

    Handles the case where OCR produces two adjacent text blocks with
    overlapping y-ranges (e.g., a bot contact name overlapping with its
    date+preview line).  Only merges when x-ranges also overlap — this
    correctly skips the common case of a name (left-aligned) and a
    timestamp (right-aligned) on the same row.
    """
    text_dets = [d for d in detections if d.get("type") == "text"
                 and len(d.get("bbox", [])) == 4]
    other_dets = [d for d in detections if d.get("type") != "text"
                  or len(d.get("bbox", [])) != 4]
    if not text_dets:
        return detections

    text_sorted = sorted(text_dets, key=lambda d: d["bbox"][1])
    groups = []
    current_group = [text_sorted[0]]

    for det in text_sorted[1:]:
        lb = current_group[-1]["bbox"]
        cb = det["bbox"]
        gx1 = min(d["bbox"][0] for d in current_group)
        gx2 = max(d["bbox"][2] for d in current_group)
        y_overlap = cb[1] < lb[3]                        # current starts before prev ends
        x_overlap = cb[0] < gx2 and cb[2] > gx1         # x-ranges intersect
        if y_overlap and x_overlap:
            current_group.append(det)
        else:
            groups.append(current_group)
            current_group = [det]
    groups.append(current_group)

    merged = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
        else:
            bboxes = [d["bbox"] for d in group]
            md = dict(group[0])
            md["bbox"] = [min(b[0] for b in bboxes), min(b[1] for b in bboxes),
                          max(b[2] for b in bboxes), max(b[3] for b in bboxes)]
            md["content"] = " ".join(d.get("content", "") for d in group)
            md["merged_lines"] = len(group)
            merged.append(md)

    all_dets = merged + other_dets
    all_dets.sort(key=lambda d: d["bbox"][1] if len(d.get("bbox", [])) == 4 else 0.0)
    return all_dets


# ── 主入口 ────────────────────────────────────────────────────────────────────

def detectUI(
    imagePath:  str,
    weightsDir: str       = "OmniParser/weights",
    debug:      bool      = False,
    debugDir:   str|None  = None,
) -> dict:
    """UI 检测主入口。

    Args:
        imagePath:  输入截图路径
        weightsDir: Florence-2 权重目录（含 icon_caption_florence/ 子目录）
        debug:      True 时保存中间结果到 debugDir
        debugDir:   debug 输出目录（None → imagePath 同目录下 debug_<imgname>/）

    Returns:
        {'labeled_image_base64': str, 'detections': List[Dict], 'image_size': (w,h)}
    """
    img_pil    = Image.open(imagePath).convert('RGB')
    w, h       = img_pil.size
    ocrBoxes   = _runOCR(imagePath)
    compoBoxes = _runUIEDCompo(imagePath)
    allDets    = _customMerge(ocrBoxes, compoBoxes, (w, h))
    allDets    = _merge_text_paragraphs(allDets)
    allDets    = _merge_overlapping_text(allDets)

    if not allDets:
        return {'labeled_image_base64': None, 'detections': [], 'image_size': (w, h)}

    imageDets = [d for d in allDets if d["type"] == "image"]
    # v3: Caption generation removed; VQA endpoint handles subject matching.

    for i, det in enumerate(allDets):
        det['image_label_id'] = i

    if debug:
        if debugDir is None:
            imgName  = os.path.splitext(os.path.basename(imagePath))[0]
            debugDir = os.path.join(os.path.dirname(imagePath), f'debug_{imgName}')
        saveDebugArtifacts(imagePath, ocrBoxes, compoBoxes, allDets, debugDir)

    labeledImg = drawDetections(img_pil.copy(), allDets)

    # 合并前原始可视化：OCR（蓝）+ compo（灰）叠加，始终生成（cheap PIL 操作）
    from core.annotationRenderer import TYPE_COLORS, drawBoxes as _drawBoxes
    premergeImg = _drawBoxes(img_pil.copy(), [
        {'bbox': o['bbox'], 'label': f"ocr:{(o.get('content') or '')[:10]}", 'color': TYPE_COLORS['text']}
        for o in ocrBoxes
    ] + [
        {'bbox': c['bbox'], 'label': f"cnn:{c.get('cnn_class','?')[:10]}", 'color': TYPE_COLORS['compo']}
        for c in compoBoxes
    ])

    return {
        'labeled_image_base64':       toBase64(labeledImg),
        'labeled_debug_base64':       toBase64(premergeImg),
        'detections':                 allDets,
        'image_size':                 (w, h),
    }


# ── 模型预加载 ────────────────────────────────────────────────────────────────

def preloadModels(weightsDir: str = "OmniParser/weights") -> None:
    """预加载所有模型（服务启动时调用，避免首次请求延迟）。"""
    print("[uiedDetection] 预热模型...")
    getPaddleOCR()
    getCNNClassifier()
    _getFlorenceModel(weightsDir)
    print("[uiedDetection] ✓ 所有模型已就绪")


preload_models = preloadModels   # 向后兼容别名
