"""launcher_policy.py — Per-task launcher screen exposure policy.

The launcher (home screen) is a special case: it shows app icons with display
names, not package names. Privacy rules here are different from in-app screens:

  - Apps *required* by the current task → EXPOSE (agent needs to tap them)
  - Well-known public/system apps → EXPOSE (safe-list; reveal nothing personal)
  - All other app icons/labels → HARD_MASK (may reveal which private apps the
    user has installed, e.g. dating/health/financial apps)

Architecture
------------
LauncherPolicy is a **per-task singleton** created by ServerPrivacyHooks on
on_task_start() and held for the task's lifetime. It is stable across all
launcher screenshots within one task — the user navigates away and back to the
launcher without retriggering task extraction.

Usage in on_screenshot():
  1. merge_icon_text(detections) → merged list (icon+label pairs collapsed)
  2. launcher_policy.apply(detections) → sets privacy_strategy on every element
  3. Skip the normal should_expose_v2() loop; go directly to NER post-pass.

Helper: merge_icon_text()
  Merges each app icon image with its nearest text label into one text
  detection keyed by the app name.  Migrated from privacy_classifier.py.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from android_world.agents.privacy.utils import _is_searchbar_widget

if TYPE_CHECKING:
    from android_world.agents.privacy.task_extractor import ExpandedRequirements

logger = logging.getLogger(__name__)

_MONTH_TOKEN = (
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|0ct|nov|dec)"
)
_WEEKDAY_TOKEN = r"(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)"
_TIME_TOKEN = r"\d{1,2}:\-?\d{2}(?::\-?\d{2})?"
_DATE_TIME_PATTERNS = (
    # Pure time, including OCR variants like "00:-34".
    re.compile(rf"^\s*{_TIME_TOKEN}\s*$", re.IGNORECASE),
    # Textual date, e.g. "Mon, Apr 13" or OCR "Sun, 0ct 15".
    re.compile(
        rf"^\s*{_WEEKDAY_TOKEN}\s*,?\s+{_MONTH_TOKEN}\s+\d{{1,2}}\s*$",
        re.IGNORECASE,
    ),
    # Time + textual date merged into one detection, sometimes with a stray OCR
    # separator like "15:34S Sun, 0ct 15".
    re.compile(
        rf"^\s*{_TIME_TOKEN}\s*[A-Za-z]?\s+{_WEEKDAY_TOKEN}\s*,?\s+"
        rf"{_MONTH_TOKEN}\s+\d{{1,2}}\s*$",
        re.IGNORECASE,
    ),
    # Numeric dates such as 2026-04-13, 26/04/13, or 04/13.
    re.compile(r"^\s*\d{2,4}[/-]\d{1,2}[/-]\d{1,4}\s*$"),
    re.compile(r"^\s*\d{1,2}[/-]\d{1,2}\s*$"),
)


# ── Shared normalization helpers ───────────────────────────────────────────────

def _norm_app_name(text: str) -> str:
    """Normalize an app display name for fuzzy matching.

    Steps:
      1. Strip leading/trailing whitespace
      2. Strip trailing "..." (OCR truncation artifact)
      3. Remove spaces, dots, hyphens, underscores
      4. Lowercase
    """
    text = text.strip()
    if text.endswith("..."):
        text = text.rstrip(".").rstrip()
    return re.sub(r'[\s.\-_]', '', text).lower()


def _matches_app_name(norm_detected: str, norm_candidate: str) -> bool:
    """True if a normalized detected label matches a normalized candidate name.

    Handles:
      - Exact match
      - OCR prefix-truncation: candidate starts with detected (≥4 chars)
      - Bidirectional substring containment (≥4 chars)
    """
    if not norm_detected:
        return False
    if norm_detected == norm_candidate:
        return True
    if len(norm_detected) >= 4:
        return (norm_candidate.startswith(norm_detected)
                or norm_detected in norm_candidate
                or norm_candidate in norm_detected)
    return False


def _is_public_launcher_datetime(text: str) -> bool:
    """Return True for launcher date/time strings that are safe to expose.

    These are structural launcher hints rather than user-preference signals.
    The matcher is intentionally conservative: it only exposes standalone
    date/time-like strings and the OCR variants observed in launcher widgets.
    """
    text = text.strip()
    if not text:
        return False
    return any(pattern.match(text) for pattern in _DATE_TIME_PATTERNS)


# ── DB-backed safe names (lazy-loaded) ────────────────────────────────────────

def _build_safe_names_from_db() -> frozenset[str]:
    """Return lowercase names of all DB entries with launcher_safe=true."""
    from android_world.agents.privacy.screen_context import _load_app_db  # noqa: PLC0415
    db = _load_app_db()
    names: set[str] = set()
    for key, entry in db.items():
        if key.startswith("__") or not isinstance(entry, dict):
            continue
        if entry.get("launcher_safe"):
            for n in entry.get("names", []):
                names.add(_norm_app_name(n))
    return frozenset(names)


_LAUNCHER_SAFE_FROM_DB: frozenset[str] = frozenset()  # populated on first call


def _get_safe_names_from_db() -> frozenset[str]:
    global _LAUNCHER_SAFE_FROM_DB
    if not _LAUNCHER_SAFE_FROM_DB:
        _LAUNCHER_SAFE_FROM_DB = _build_safe_names_from_db()
    return _LAUNCHER_SAFE_FROM_DB

# ── Constants ─────────────────────────────────────────────────────────────────

EXPOSE = "expose"
HARD_MASK = "hard_mask"

# Element types that represent app icons on launcher screens.
_LAUNCHER_ICON_TYPES: frozenset[str] = frozenset({"image", "photo", "icon"})

# App labels that are safe to expose unconditionally on the launcher.
# These are standard system / Google / AOSP apps present on virtually every
# Android phone and reveal nothing personal about the user.
# Keep lowercase — comparisons use content.strip().lower().
#
# This is the fallback list for AOSP / Google apps not yet in app_context_db.json.
# Preferred: add new safe apps to the DB with "launcher_safe": true instead of
# extending this set (see migrate_launcher_safe_to_db.py for the migration script).
# Apps already in the DB (camera, chrome, contacts, etc.) are covered by
# _get_safe_names_from_db() and checked first in is_safe_app().
LAUNCHER_SAFE_APP_LABELS: frozenset[str] = frozenset({
    # ── Navigation & UI chrome ────────────────────────────────────────────
    "home", "back", "recents",
    # ── Core AOSP / Google apps ───────────────────────────────────────────
    "settings", "设置",
    "phone", "dialer", "电话", "拨号",
    "contacts", "people", "通讯录", "联系人",
    "calendar", "日历",
    "camera", "相机",
    "calculator", "计算器",
    "clock", "时钟", "闹钟",
    "messages", "message", "messaging", "sms", "短信", "信息",
    "files", "file manager", "file manager:", "my files",
    "files by google", "downloads", "下载", "文件", "文件管理",
    # ── Google suite (pre-installed on most Android) ──────────────────────
    "chrome",
    "gmail",
    "maps", "google maps", "地图",
    "play store", "应用商店", "google play",
    "youtube",
    "photos", "google photos", "相册",
    "drive", "google drive",
    "docs", "google docs",
    "sheets", "google sheets",
    "slides", "google slides",
    "meet", "google meet",
    "wallet", "google wallet",
    "pay", "google pay",
    "assistant", "google assistant",
    "duo", "google duo",
    # ── Other system / AOSP ───────────────────────────────────────────────
    "gallery", "图库",
    "music", "google play music", "音乐",
    "email", "邮件", "邮箱",
    "browser", "internet", "浏览器",
    "notes", "记事本", "便签",
    "recorder", "voice recorder", "录音机", "录音",
    "keyboard", "gboard", "键盘",
    "accessibility", "辅助功能",
})

# Precomputed normalized version of LAUNCHER_SAFE_APP_LABELS for O(1) lookup.
# Built once at module load; entries are normalized with _norm_app_name so that
# multi-word labels like "google drive" → "googledrive" match correctly.
_LAUNCHER_SAFE_APP_LABELS_NORM: frozenset[str] = frozenset(
    _norm_app_name(s) for s in LAUNCHER_SAFE_APP_LABELS
)


# ── LauncherPolicy ─────────────────────────────────────────────────────────────

class LauncherPolicy:
    """Per-task singleton: decides which launcher app icons/labels to expose.

    Args:
        expanded_reqs: ExpandedRequirements from task_extractor.to_expanded().
                       All app_names across all requirements are collected
                       at construction time.
    """

    def __init__(self, expanded_reqs: "ExpandedRequirements") -> None:
        # Flatten all registered app names from every requirement, normalized.
        self._required_names: set[str] = set()
        for req in expanded_reqs.requirements:
            for name in req.app_names:
                self._required_names.add(_norm_app_name(name))
        logger.info(
            "[launcher_policy] task requires apps (normalized): %s",
            sorted(self._required_names),
        )

    def is_required_app(self, display_text: str) -> bool:
        """Return True if *display_text* matches any required app name.

        Uses _norm_app_name + _matches_app_name for consistent normalization:
        strips OCR "..." truncation, removes spaces/dots/hyphens, and applies
        bidirectional substring + prefix matching (≥4 normalized chars).
        """
        norm = _norm_app_name(display_text)
        return any(_matches_app_name(norm, r) for r in self._required_names)

    @staticmethod
    def is_safe_app(display_text: str) -> bool:
        """Return True if *display_text* is a well-known safe system/Google app.

        Lookup order:
          1. DB entries with launcher_safe=true (via _get_safe_names_from_db)
          2. Hardcoded LAUNCHER_SAFE_APP_LABELS fallback (AOSP apps not yet in DB)
        Both use _norm_app_name + _matches_app_name for consistent matching.
        """
        norm = _norm_app_name(display_text)
        if not norm:
            return False
        # 1. DB-backed safe names (normalized at load time)
        db_safe = _get_safe_names_from_db()
        if any(_matches_app_name(norm, s) for s in db_safe):
            return True
        # 2. Hardcoded fallback (normalized at module load time)
        return any(_matches_app_name(norm, s) for s in _LAUNCHER_SAFE_APP_LABELS_NORM)

    def apply(self, detections: list) -> None:
        """Set privacy_strategy on every detection for a launcher screen.

        Policy (in order):
          1. Required app (matches task requirement) → EXPOSE
          2. Public launcher date/time text → EXPOSE
          3. Safe-list app (known public system app) → EXPOSE
          4. Search-bar / search-widget (wide + short image strip) → EXPOSE
          5. Everything else → HARD_MASK

        Args:
            detections: Detection list after merge_icon_text() has been applied.
        """
        exposed_required = 0
        exposed_safe = 0
        masked = 0

        for det in detections:
            content = det.get("content", "")

            if self.is_required_app(content):
                det["privacy_strategy"] = EXPOSE
                exposed_required += 1
                logger.info("[launcher] %r → EXPOSE (required app)", content)
            elif _is_public_launcher_datetime(content):
                det["privacy_strategy"] = EXPOSE
                exposed_safe += 1
                logger.debug("[launcher] %r → EXPOSE (public datetime)", content)
            elif self.is_safe_app(content):
                det["privacy_strategy"] = EXPOSE
                exposed_safe += 1
                logger.debug("[launcher] %r → EXPOSE (safe-list)", content)
            elif (det.get("type", "").lower() in _LAUNCHER_ICON_TYPES
                  and _is_searchbar_widget(det.get("bbox", []), None)):
                det["privacy_strategy"] = EXPOSE
                exposed_safe += 1
                logger.info("[launcher] bbox=%s → EXPOSE (search-bar widget geometry)",
                            det.get("bbox", []))
            else:
                det["privacy_strategy"] = HARD_MASK
                masked += 1
                logger.debug("[launcher] %r → HARD_MASK", content)

        logger.info(
            "[launcher] apply: %d required, %d safe, %d masked",
            exposed_required, exposed_safe, masked,
        )


# ── merge_icon_text() ─────────────────────────────────────────────────────────

def merge_icon_text(detections: list) -> list:
    """Merge each app icon with its nearest text label on launcher screens.

    Replaces (icon_det, label_det) pairs with a single text detection whose:
      - type    = "text"
      - content = label_det["content"]
      - bbox    = union of icon_det["bbox"] and label_det["bbox"]
      - source  = "launcher_merged"

    After merging, LauncherPolicy.apply() classifies merged elements by app
    name — no VQA on logos needed.

    Icons without a nearby label are left unchanged (handled as images).

    Pairing geometry — all three constraints must hold:
      1. Icon size: w ≤ 0.25 and h ≤ 0.18 and area ≤ 0.04
         (app icons are small; rejects wallpaper/background images)
      2. Text is below the icon: ty1 ≥ y2 - 0.05
         (label starts at or just below icon bottom)
      3. Horizontal alignment: |txt_cx - icon_cx| ≤ txt_w × 0.5 + icon_w × 0.3
         (label is roughly centred under the icon)
    """
    icon_indices = {i for i, d in enumerate(detections)
                    if d.get("type") in _LAUNCHER_ICON_TYPES
                    and len(d.get("bbox", [])) == 4}
    text_indices = {i for i, d in enumerate(detections)
                    if d.get("type") not in _LAUNCHER_ICON_TYPES
                    and d.get("content", "").strip()
                    and len(d.get("bbox", [])) == 4}

    consumed_icons: set[int] = set()
    consumed_texts: set[int] = set()
    merged: list[dict] = []

    for ii in sorted(icon_indices):
        img = detections[ii]
        x1, y1, x2, y2 = img["bbox"]
        icon_cx = (x1 + x2) / 2
        icon_w  = x2 - x1
        icon_h  = y2 - y1

        # Constraint 1: icon must be small enough to be an app icon
        if icon_w > 0.25 or icon_h > 0.18 or icon_w * icon_h > 0.04:
            continue

        best_ti, best_dist = None, float("inf")
        for ti in text_indices - consumed_texts:
            txt = detections[ti]
            tx1, ty1, tx2, ty2 = txt["bbox"]
            txt_cx = (tx1 + tx2) / 2
            txt_w  = tx2 - tx1

            # Constraint 2: text must start at or just below icon bottom
            if ty1 < y2 - 0.05 or ty1 > y2 + max(0.08, icon_h * 1.5):
                continue

            # Constraint 3: text x-center ≈ icon x-center
            if abs(txt_cx - icon_cx) > txt_w * 0.5 + icon_w * 0.3:
                continue

            dist = abs(ty1 - y2)
            if dist < best_dist:
                best_dist = dist
                best_ti = ti

        if best_ti is not None:
            txt = detections[best_ti]
            tx1, ty1, tx2, ty2 = txt["bbox"]
            merged_bbox = [min(x1, tx1), min(y1, ty1), max(x2, tx2), max(y2, ty2)]
            merged.append({
                "type":    "text",
                "content": txt["content"].strip(),
                "bbox":    merged_bbox,
                "source":  "launcher_merged",
            })
            consumed_icons.add(ii)
            consumed_texts.add(best_ti)

    result = [d for i, d in enumerate(detections)
              if i not in consumed_icons and i not in consumed_texts]
    result.extend(merged)
    result.sort(key=lambda d: d["bbox"][1] if len(d.get("bbox", [])) == 4 else 0.0)
    return result
