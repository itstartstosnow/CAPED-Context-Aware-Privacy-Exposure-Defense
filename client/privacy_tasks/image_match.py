"""Image matching helpers for project-local privacy benchmark tasks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def normalized_image_similarity(
    first_image: np.ndarray, second_image: np.ndarray
) -> float:
  """Returns a correlation-like visual similarity score in [0, 1]."""
  first = first_image.astype(np.float32).reshape(-1)
  second = second_image.astype(np.float32).reshape(-1)
  first -= first.mean()
  second -= second.mean()
  denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
  if denominator == 0:
    return 0.0
  return (float(np.dot(first, second) / denominator) + 1.0) / 2.0


def best_reference_match_score(
    screen_pixels: np.ndarray,
    reference_path: Path,
    match_size: int = 64,
) -> float:
  """Finds the best large square visual match for a reference image."""
  screen = np.asarray(screen_pixels)
  if screen.ndim != 3 or screen.shape[0] < 10 or screen.shape[1] < 10:
    return 0.0

  reference = Image.open(reference_path).convert("RGB").resize(
      (match_size, match_size)
  )
  reference_arr = np.asarray(reference)
  height, width = screen.shape[:2]
  max_square = min(width, height)
  candidate_sizes = [
      int(max_square * ratio) for ratio in (1.0, 0.9, 0.8, 0.7, 0.6)
  ]
  best_score = 0.0
  for size in candidate_sizes:
    if size < 100:
      continue
    x_positions = sorted({0, max(0, (width - size) // 2), max(0, width - size)})
    y_step = max(1, size // 5)
    for x in x_positions:
      for y in range(0, max(1, height - size + 1), y_step):
        crop = screen[y : y + size, x : x + size, :3]
        if crop.shape[0] != size or crop.shape[1] != size:
          continue
        crop_img = Image.fromarray(crop.astype(np.uint8), "RGB").resize(
            (match_size, match_size)
        )
        best_score = max(
            best_score,
            normalized_image_similarity(np.asarray(crop_img), reference_arr),
        )
  return best_score
