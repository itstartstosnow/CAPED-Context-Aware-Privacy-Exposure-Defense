"""privacy/utils.py — Shared geometry and pixel utility helpers.

These pure-utility functions have no dependency on the policy table or
classifier state.  They live here so that both launcher_policy and
privacy_classifier can import them without creating a circular dependency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

# Tiny images (status-bar icons < 0.3% of screen) are always exposed.
# 0.003 × (1264×2800) ≈ 10,618 px² ≈ 103×103 px — excludes profile photos and
# app icons.  Raised from 0.002: UIED sometimes detects only the shadow/edge
# strip of a circular FAB (area ~0.002042, 6.7:1 aspect) which just barely
# exceeded the old threshold.
_ICON_AREA_MAX: float = 0.003

# Pixel colour std-dev threshold for search-bar widget detection.
# Reference: Google Launcher search widget (1080×2400): std=44.4, mean=242.8.
# Threshold 55 leaves ~10 units of headroom; typical banner/photo has std > 70.
_SEARCHBAR_STD_MAX: float = 55.0


# ── Geometry / pixel helpers ───────────────────────────────────────────────────

def _bbox_area(bbox) -> float:
    """Return area of a normalised bbox as a fraction of the screen (0–1)."""
    if len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _is_tiny_icon(bbox: list) -> bool:
    """Return True if *bbox* is a tiny status-bar icon or UI decoration.

    Elements below _ICON_AREA_MAX are physical chrome, not content —
    they are always exposed regardless of screen context or task requirement.
    """
    return _bbox_area(bbox) < _ICON_AREA_MAX


def _is_searchbar_widget(
    bbox: list,
    pixels: Optional[np.ndarray] = None,
) -> bool:
    """Return True if *bbox* matches a search-bar / search-widget.

    Two-stage check:
      Stage 1 — Geometry (always applied):
        - Normalised width  >= 0.70  (spans most of the screen width)
        - Normalised height <= 1/12 ≈ 0.083  (very short horizontal strip)
        Implied minimum aspect ratio: 0.70 / 0.083 ≈ 8.4.

      Stage 2 — Pixel colour uniformity (only when *pixels* is given):
        - Per-pixel std-dev of the crop < _SEARCHBAR_STD_MAX (55.0)
        Critical for non-launcher screens where thin banner images can
        share the same geometry as a search bar.

    Reference measurement — Google Launcher search widget (1080×2400):
      bbox=[0.067, 0.891, 0.928, 0.948], w=0.861, h=0.056, aspect=15.3
      mean=242.8, std=44.4  →  threshold 55 gives ~10 units of headroom.

    Args:
        bbox:   Normalised [x1, y1, x2, y2] bounding box.
        pixels: Full-screen RGB numpy array (H×W×3).  When None, Stage 2 is
                skipped — only safe when the caller guarantees no banner risk
                (e.g. launcher screens, which contain no ad banners).
    """
    if len(bbox) != 4:
        return False
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    if bw < 0.70 or bh <= 0 or bh > (1 / 12):
        return False
    if pixels is None:
        return True  # geometry-only path (launcher — no banner risk)
    # Stage 2: pixel uniformity
    h_img, w_img = pixels.shape[:2]
    px1 = max(0, int(bbox[0] * w_img))
    py1 = max(0, int(bbox[1] * h_img))
    px2 = min(w_img, int(bbox[2] * w_img))
    py2 = min(h_img, int(bbox[3] * h_img))
    if px2 <= px1 or py2 <= py1:
        return False  # degenerate crop: reject conservatively
    crop = pixels[py1:py2, px1:px2].astype(np.float32)
    return float(crop.std()) < _SEARCHBAR_STD_MAX
