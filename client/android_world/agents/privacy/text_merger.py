"""Paragraph merge post-processing for OCR text detections.

OCR 产生逐行文字检测框，本模块将相邻行合并为段落级检测，提升后续
LLM 文字相关性判断的准确性，并减少总检测数量。

合并条件（相邻两行均需满足）：
  1. 纵向间距 gap < 前行高度 × GAP_RATIO
  2. 两行之间（纵向 gap 区间内）在本组横向跨度内无其他元素（不被图片/图标分隔）

只有 source="ocr" 且 type="text" 的检测框参与合并；image/icon 等元素原样保留。
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Default gap threshold: gap < line_height * GAP_RATIO → candidate for merge
GAP_RATIO: float = 0.75


def merge_text_paragraphs(
    detections: List[dict],
    gap_ratio: float = GAP_RATIO,
) -> List[dict]:
    """Post-process detections, merging adjacent OCR text lines into paragraphs.

    Args:
        detections: List of detection dicts from /api/v1/detect.
            Each dict has: type, bbox ([x1,y1,x2,y2] normalized), content, source, ...
        gap_ratio: Fraction of line height used as the vertical gap threshold.

    Returns:
        New list with merged paragraph detections replacing fragmented lines.
        Non-OCR-text elements are returned unchanged.

    Merged detection has:
        bbox:         Union bbox of all member lines.
        content:      Space-joined text of all member lines.
        merged_lines: List of original image_label_ids (added for debug).
        Other fields: Copied from the first line in the group.
    """
    if not detections:
        return detections

    # Split into OCR text candidates and everything else
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

    # Sort by top-y
    text_sorted = sorted(text_dets, key=lambda d: d["bbox"][1])

    # Greedy grouping
    groups: List[List[dict]] = []
    current_group: List[dict] = [text_sorted[0]]

    for det in text_sorted[1:]:
        last = current_group[-1]
        lb = last["bbox"]   # [x1, y1, x2, y2] of last line in group
        cb = det["bbox"]    # [x1, y1, x2, y2] of candidate line

        last_height = max(lb[3] - lb[1], 1e-6)
        gap = cb[1] - lb[3]  # positive = gap; negative = overlap

        # Condition 1: gap threshold
        if gap >= last_height * gap_ratio:
            groups.append(current_group)
            current_group = [det]
            continue

        # Condition 2: x-range overlap — new line must share some horizontal extent
        # with the current group. Prevents merging side-by-side elements (e.g. nav tabs).
        gx1 = min(d["bbox"][0] for d in current_group)
        gx2 = max(d["bbox"][2] for d in current_group)
        if cb[2] <= gx1 or cb[0] >= gx2:
            groups.append(current_group)
            current_group = [det]
            continue

        # Condition 3: no interleaving non-text element in the column
        gap_y1, gap_y2 = min(lb[3], cb[1]), max(lb[3], cb[1])

        has_interleave = any(
            ob["bbox"][1] < gap_y2 and ob["bbox"][3] > gap_y1   # overlaps gap vertically
            and ob["bbox"][0] < gx2 and ob["bbox"][2] > gx1     # overlaps column horizontally
            for ob in other_dets
            if len(ob.get("bbox", [])) == 4
        )

        if has_interleave:
            groups.append(current_group)
            current_group = [det]
            continue

        # Condition 4: candidate merged bbox must not contain other OCR text elements.
        # If merging would create a bbox that swallows an unrelated text element,
        # start a new group instead (prevents cross-column merges in list UIs).
        candidate_gx1 = min(gx1, cb[0])
        candidate_gx2 = max(gx2, cb[2])
        candidate_gy1 = min(d["bbox"][1] for d in current_group)
        candidate_gy2 = max(cb[3], max(d["bbox"][3] for d in current_group))

        contains_foreign_text = any(
            od is not det                            # not the candidate itself
            and od.get("source") == "ocr" and od.get("type") == "text"
            and len(od.get("bbox", [])) == 4
            and od not in current_group              # not already in group
            and od["bbox"][0] >= candidate_gx1 and od["bbox"][2] <= candidate_gx2
            and od["bbox"][1] >= candidate_gy1 and od["bbox"][3] <= candidate_gy2
            for od in text_sorted
        )

        if contains_foreign_text:
            groups.append(current_group)
            current_group = [det]
        else:
            current_group.append(det)

    groups.append(current_group)

    # Convert groups → merged detections
    merged_text_dets: List[dict] = []
    n_merged = 0

    for group in groups:
        if len(group) == 1:
            merged_text_dets.append(group[0])
            continue

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
        merged_det["merged_lines"] = [d.get("image_label_id") for d in group]

        merged_text_dets.append(merged_det)
        n_merged += len(group) - 1
        logger.debug(
            "text_merger: merged %d lines → '%s...'",
            len(group),
            merged_content[:60],
        )

    if n_merged:
        logger.info(
            "text_merger: %d OCR lines → %d paragraphs (%d fewer elements)",
            len(text_dets),
            len(merged_text_dets),
            n_merged,
        )

    # Re-merge into full list, sorted by bbox y1
    all_dets = merged_text_dets + other_dets
    all_dets.sort(key=lambda d: d["bbox"][1] if len(d.get("bbox", [])) == 4 else 0.0)
    return all_dets
