"""Privacy hooks backed by the agentPrivacy server — V3 architecture.

V3 架构说明
===========
- 服务器只提供原始模型 API（detect / vqa / llm / ner）
- 本地持有全部业务逻辑（屏幕上下文分类、任务需求提取、隐私分类决策、图像遮蔽）

on_screenshot 流程：
  1. POST /api/v1/detect → 原始 detections JSON
  2. screen_context.classify_screen(activity_name) → ScreenContext
  3. 本地先为每个 detection 做 cheap policy planning（不急着裁图）
  4. 只有需要 image-side matching 的 detection 才做 crop/PNG/base64
     - 图片有 specific subject 时做 inline VQA
     - nav_only_vqa 图片收集后做 batch VQA
  5. NER post-processing：有 original_content 的元素（非图片类）一律改为 NER_REPLACE
  6. image_redactor.redact_image(pixels, detections_with_strategy) → anonymized pixels
  7. 更新 last_detection_json（含 privacy_strategy 字段）

on_task_start 流程：
  1. POST /api/v1/ner/process → anonymized_goal, ner_mapping
  2. task_extractor.extract_v2(anonymized_goal) → TaskRequirements  (V2 Structured CoT)
  3. task_extractor.to_expanded(reqs) → _expanded_reqs (ExpandedRequirements)

服务器端点：
  POST /api/v1/ner/process      — NER 脱敏（接口不变）
  POST /api/v1/detect           — UIED 检测，返回原始 JSON
  POST /api/v1/llm_vqa/single_batch — Florence-2 VQA（由 PrivacyClassifier 内部调用）
  POST /api/v1/llm_2b           — Qwen3.5-2B 文本补全（由 TaskExtractor 调用）
  POST /api/v1/embed_gemma      — EmbeddingGemma-300m（由 PrivacyClassifier 文本相关性后端调用）
  GET  /api/v1/health           — 健康检查

Per-step debug artifacts (instance attributes after on_screenshot):
  last_detection_json        — 完整 detection dict（含 privacy_strategy 字段）
  last_labeled_image_bytes   — 带 bbox overlay 的 PNG bytes（如服务器提供）
  last_server_log            — 服务器日志（仅 debug=True 时）
"""

import base64
import dataclasses
import io
import json
import logging
import logging.handlers
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests
from PIL import Image

from android_world.agents.privacy.hooks import PrivacyHooks
from android_world.agents.privacy import screen_context as sc_module
from android_world.agents.privacy.context_policy import TEXT_PRIVACY, get_level
from android_world.agents.privacy.task_extractor import (
    TaskExtractor, ExpandedRequirements, TaskRequirements,
)
from android_world.agents.privacy.privacy_classifier import (
    PrivacyClassifier, EXPOSE, HARD_MASK, NER_REPLACE,
)
from android_world.agents.privacy.launcher_policy import (
    LauncherPolicy, merge_icon_text,
)
from android_world.agents.privacy import image_redactor

logger = logging.getLogger(__name__)


def deanonymize_requirement_app_names(
    reqs: TaskRequirements,
    ner_mapping: Optional[dict],
) -> TaskRequirements:
    """Restore anonymized placeholders only in extracted requirement app fields."""
    reverse_mapping = (ner_mapping or {}).get("reverse_mapping", {})
    if not isinstance(reverse_mapping, dict) or not reverse_mapping:
        return reqs

    for req in reqs.requirements:
        restored_app = reverse_mapping.get(req.app)
        if restored_app:
            logger.info(
                "TaskExtractor app de-anonymized: %r -> %r",
                req.app,
                restored_app,
            )
            req.app = restored_app
    return reqs


@dataclasses.dataclass
class _TaskState:
    """All state that lives for exactly one task lifetime.

    Created fresh at the start of each task (on_task_start or __init__).
    Reset atomically via ``self._task = _TaskState()`` in on_task_end().
    """
    ner_mapping:     Optional[dict]                 = None
    anonymized_goal: Optional[str]                  = None
    expanded_reqs:   Optional["ExpandedRequirements"] = None
    launcher_policy: Optional["LauncherPolicy"]     = None
    step_counter:    int                            = 0
    task_name:       str                            = ""
    log_handler:     Optional[logging.FileHandler]  = None


# ── Launcher VQA rescue ────────────────────────────────────────────────────────

_LAUNCHER_ICON_TYPES: frozenset = frozenset({"image", "photo", "icon"})
_ICON_AREA_THRESHOLD: float = 0.04   # same as merge_icon_text()
_VQA_ICON_BATCH_SIZE: int = 4        # questions per HTTP request

# Core dock apps to rescue via VQA when they have no text label.
# These 4 always appear in the Android dock without text; keeping the list
# tiny (≤ batch_size=4) means a single HTTP request per icon candidate.
_DOCK_VQA_SAFE_APPS: frozenset[str] = frozenset({"phone", "messages", "camera", "settings"})

# Icon size bounds for dock VQA rescue (normalized coords, 0–1).
# Reference: observed dock icons w≈0.145, h≈0.065 on a typical emulator screen.
# Accept 0.75x – 1.5x of the reference to handle minor layout variation.
_DOCK_ICON_MIN_W: float = 0.11   # 0.145 × 0.75
_DOCK_ICON_MAX_W: float = 0.22   # 0.145 × 1.5
_DOCK_ICON_MIN_H: float = 0.05   # 0.065 × 0.75
_DOCK_ICON_MAX_H: float = 0.10   # 0.065 × 1.5


def _vqa_rescue_launcher_icons(
    detections: list,
    launcher_policy: "LauncherPolicy",
    pixels: "np.ndarray",
    session: "requests.Session",
    base_url: str,
    timeout: float,
) -> None:
    """VQA rescue (two passes): expose icons that have no text label.

    Pass 1 — Required apps:
        For each task-required app not yet exposed via text, ask VQA
        "Does this image contain <app>?" against every HARD_MASK'd icon.

    Pass 2 — Dock safe-list (_DOCK_VQA_SAFE_APPS):
        Same query for the 4 core dock apps (Phone, Messages, Camera,
        Settings), which never show text labels on the launcher shelf.
        Only icon-sized detections (within _DOCK_ICON_MIN/MAX_W/H) are
        considered, filtering out widgets and wallpaper images.
        All 4 concepts fit in one batch → single HTTP request per icon.

    Args:
        detections: Detection list after launcher_policy.apply() (mutated).
        launcher_policy: Per-task LauncherPolicy (provides _required_names).
        pixels: Full screenshot numpy array (H×W×3).
        session: Requests session for HTTP calls.
        base_url: AgentPrivacy server base URL.
        timeout: HTTP timeout in seconds.
    """
    from android_world.agents.privacy.launcher_policy import _norm_app_name, _matches_app_name  # noqa: PLC0415

    h, w = pixels.shape[:2]

    def _already_exposed(name: str) -> bool:
        return any(
            det.get("privacy_strategy") == EXPOSE
            and _matches_app_name(_norm_app_name(det.get("content", "")), name)
            for det in detections
        )

    def _crop_b64(det: dict) -> str | None:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            return None
        x1, y1, x2, y2 = bbox
        px1, py1 = max(0, int(x1 * w)), max(0, int(y1 * h))
        px2, py2 = min(w, int(x2 * w)), min(h, int(y2 * h))
        if px2 <= px1 or py2 <= py1:
            return None
        crop_img = Image.fromarray(pixels[py1:py2, px1:px2].astype(np.uint8))
        buf = io.BytesIO()
        crop_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _vqa_batch(image_b64: str, concepts: list[str]) -> dict[str, bool]:
        """Ask VQA for a list of concepts; return {concept: matched} map."""
        items = [{"image_crop": image_b64, "concept": c} for c in concepts]
        try:
            resp = session.post(
                f"{base_url}/api/v1/llm_vqa/single_batch",
                json={"items": items},
                timeout=timeout,
            )
            resp.raise_for_status()
            return {
                c: r.get("answer") == "yes"
                for c, r in zip(concepts, resp.json().get("results", []))
            }
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[launcher_vqa] VQA batch failed (%s); skipping", exc)
            return {}

    # ── Pass 1: required-app rescue ──────────────────────────────────────────
    unmatched = [r for r in launcher_policy._required_names if not _already_exposed(r)]

    if unmatched:
        logger.info("[launcher_vqa] unmatched required apps: %s — starting VQA rescue", unmatched)
        for det in detections:
            if det.get("privacy_strategy") != HARD_MASK:
                continue
            if det.get("type") not in _LAUNCHER_ICON_TYPES:
                continue
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > _ICON_AREA_THRESHOLD:
                continue
            image_b64 = _crop_b64(det)
            if image_b64 is None:
                continue
            for start in range(0, len(unmatched), _VQA_ICON_BATCH_SIZE):
                chunk = unmatched[start:start + _VQA_ICON_BATCH_SIZE]
                for rname, matched in _vqa_batch(image_b64, chunk).items():
                    if matched:
                        det["privacy_strategy"] = EXPOSE
                        logger.info("[launcher_vqa] %r → EXPOSE via VQA (matched %r)",
                                    det.get("content", "<image>"), rname)
                        unmatched.remove(rname)
                        break
            if not unmatched:
                break

    # ── Pass 2: dock safe-list rescue ────────────────────────────────────────
    dock_unmatched = [name for name in _DOCK_VQA_SAFE_APPS if not _already_exposed(name)]

    if not dock_unmatched:
        return

    logger.info("[launcher_vqa] dock rescue candidates: %s", dock_unmatched)
    for det in detections:
        if det.get("privacy_strategy") != HARD_MASK:
            continue
        if det.get("type") not in _LAUNCHER_ICON_TYPES:
            continue
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        iw, ih = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if not (_DOCK_ICON_MIN_W <= iw <= _DOCK_ICON_MAX_W
                and _DOCK_ICON_MIN_H <= ih <= _DOCK_ICON_MAX_H):
            continue  # not dock-icon sized
        image_b64 = _crop_b64(det)
        if image_b64 is None:
            continue
        # All dock candidates fit in one batch (len ≤ _VQA_ICON_BATCH_SIZE=4)
        for name, matched in _vqa_batch(image_b64, dock_unmatched).items():
            if matched:
                det["privacy_strategy"] = EXPOSE
                logger.info("[launcher_vqa] dock %r → EXPOSE via VQA (matched %r)",
                            det.get("content", "<image>"), name)
                dock_unmatched.remove(name)
                break
        if not dock_unmatched:
            break


class ServerPrivacyHooks(PrivacyHooks):
    """V3 privacy hooks: local decision logic + server model APIs.

    After each on_screenshot() call the following instance attributes are
    updated so the caller can save debug artifacts:

    Attributes:
        last_detection_json: Dict with 'detections' list; each detection
            includes 'privacy_strategy' ("expose" or "hard_mask") added locally.
        last_labeled_image_bytes: Optional PNG bytes with bbox overlays
            (returned by /api/v1/detect if the server provides it).
        last_server_log: Server-side log string (only when debug=True).
        current_ner_mapping: NER entity mapping for the current task.

    Args:
        base_url: Base URL of the agentPrivacy server.
        timeout: HTTP request timeout in seconds.
        output_dir: Per-task output directory for disclosure logging.
        debug: If True, request server-side debug info.
        llm_model: LLM model alias for task extraction ("0.8b" or "2b").
                   Passed through to TaskExtractor. See task_extractor.LLM_ENDPOINTS.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout: float = 30.0,
        output_dir: str = "",
        debug: bool = False,
        debug_dir: str = "",
        llm_model: str = "0.8b",
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        self._output_dir = output_dir
        self._debug = debug
        self._debug_dir = debug_dir

        # Per-task state (reset atomically in on_task_end)
        self._task = _TaskState()

        # Component instances
        self._task_extractor = TaskExtractor(base_url, timeout, llm_model=llm_model)
        self._privacy_classifier = PrivacyClassifier(base_url, timeout)

        # Per-step debug artifacts
        self.last_detection_json: Optional[dict] = None
        self.last_labeled_image_bytes: Optional[bytes] = None
        self.last_server_log: Optional[str] = None

        # Per-step diagnostic info (updated each on_screenshot, reset in on_task_end)
        self.last_activity_name: str = ""
        self.last_screen_ctx: Optional[sc_module.ScreenContext] = None
        self.last_privacy_summary: dict = {}

        # For disclosure mechanism
        self._last_original_pixels: Optional[np.ndarray] = None
        self._last_anon_pixels: Optional[np.ndarray] = None
        self._disclosure_counter: int = 0

        # Per-step timing state (written by env_wrapper and collected in on_screenshot/on_action)
        self._last_observation_fetch_time_s: float = 0.0
        self._last_fgactivity_time_s: float = 0.0
        self._last_screenshot_timing: dict = {}
        self._last_action_check_time_s: float = 0.0
        self._last_action_exec_and_settle_time_s: float = 0.0

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def current_ner_mapping(self) -> Optional[dict]:
        return self._task.ner_mapping

    @property
    def last_raw_pixels(self) -> Optional[np.ndarray]:
        return self._last_original_pixels

    @property
    def task_requirements_list(self) -> list:
        """Return expanded task requirements as a serializable list for debug logging."""
        if not self._task.expanded_reqs:
            return []
        return [
            {
                "app_names":    r.app_names,
                "subject_type": r.subject_type,
                "subject":      r.subject,
            }
            for r in self._task.expanded_reqs.requirements
        ]

    def set_output_dir(self, path: str) -> None:
        self._output_dir = path
        self._disclosure_counter = 0
        # If NER mapping was captured before output_dir was set, save it now.
        self._save_ner_mapping()

    def set_debug_dir(self, path: str) -> None:
        """Set root directory for per-step debug artifact auto-saving."""
        self._debug_dir = path

    def set_task_name(self, name: str) -> None:
        """Set current task name; resets step counter and switches log file."""
        self._task.task_name = name
        self._task.step_counter = 0
        self._rotate_task_log_handler()

    def _save_ner_mapping(self) -> None:
        """Persist current NER mapping to output_dir/ner_mapping.json and debug subdir."""
        if not self._task.ner_mapping:
            return
        paths = []
        if self._output_dir:
            paths.append(os.path.join(self._output_dir, "ner_mapping.json"))
        if self._debug_dir and self._task.task_name:
            debug_subdir = self._task_debug_dir()
            debug_subdir.mkdir(parents=True, exist_ok=True)
            paths.append(str(debug_subdir / "ner_mapping.json"))
        for ner_path in paths:
            try:
                os.makedirs(os.path.dirname(ner_path), exist_ok=True)
                with open(ner_path, "w", encoding="utf-8") as f:
                    json.dump(self._task.ner_mapping, f, ensure_ascii=False, indent=2)
                logger.info("NER mapping saved to %s", ner_path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Failed to save NER mapping to %s (%s).", ner_path, exc)

    def _rotate_task_log_handler(self) -> None:
        """Remove previous per-task FileHandler and open one for the current task."""
        root = logging.getLogger()
        # Remove old handler
        if self._task.log_handler is not None:
            self._task.log_handler.close()
            root.removeHandler(self._task.log_handler)
            self._task.log_handler = None
        # Open new handler only when debug_dir and task name are known
        if not self._debug_dir or not self._task.task_name:
            return
        log_path = self._task_debug_dir() / "privacy_hooks.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(str(log_path), encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        root.addHandler(handler)
        self._task.log_handler = handler
        logger.info("=== Task: %s ===", self._task.task_name)

    def _task_debug_dir(self) -> Path:
        """Return per-task debug subdirectory path."""
        base = Path(self._debug_dir)
        if self._task.task_name:
            return base / self._task.task_name
        return base

    def _save_debug_artifacts(self, anon_pixels: np.ndarray) -> None:
        """Save per-step debug artifacts to <debug_dir>/<task_name>/."""
        from PIL import Image as _Image
        debug_dir = self._task_debug_dir()
        debug_dir.mkdir(parents=True, exist_ok=True)
        idx = self._task.step_counter
        prefix = debug_dir / f"step_{idx:03d}"
        try:
            # raw screenshot
            if self._last_original_pixels is not None:
                _Image.fromarray(self._last_original_pixels.astype("uint8")).save(
                    f"{prefix}_raw.png"
                )
            # labeled PNG (bbox overlays from server)
            if self.last_labeled_image_bytes:
                Path(f"{prefix}_labeled.png").write_bytes(self.last_labeled_image_bytes)
            # detection JSON
            if self.last_detection_json is not None:
                Path(f"{prefix}_detection.json").write_text(
                    json.dumps(self.last_detection_json, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            # context JSON
            ctx_data = {
                "step": idx,
                "activity_name": self.last_activity_name,
                "screen_context": self.last_screen_ctx.value if self.last_screen_ctx else "unknown",
                "task_requirements": self.task_requirements_list,
                "privacy": self.last_privacy_summary,
            }
            Path(f"{prefix}_context.json").write_text(
                json.dumps(ctx_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # server log (only when debug=True was also set)
            if self.last_server_log is not None:
                Path(f"{prefix}_server_log.txt").write_text(
                    self.last_server_log, encoding="utf-8"
                )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("_save_debug_artifacts step %d failed: %s", idx, exc)

    # ------------------------------------------------------------------
    # PrivacyHooks interface
    # ------------------------------------------------------------------

    def on_task_start(self, goal: str) -> str:
        """NER-anonymize the task goal and extract structured task requirements.

        Steps:
          1. POST /api/v1/ner/process → anonymized_goal, ner_mapping
          2. task_extractor.extract_v2(anonymized_goal) → TaskRequirements  (V2 Structured CoT)
          3. task_extractor.to_expanded(reqs) → _expanded_reqs (ExpandedRequirements)
        """
        self._task.step_counter = 0

        # Step 1: NER anonymization
        _t0_ner = time.perf_counter()
        try:
            resp = self._session.post(
                f"{self._base_url}/api/v1/ner/process",
                data={"text": goal},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            self._task.ner_mapping = payload.get("ner_mapping")
            self._task.anonymized_goal = payload.get("anonymized_text", goal)
            logger.info(
                "NER: %d entities found; anonymized goal: %s",
                len(payload.get("entities", [])),
                self._task.anonymized_goal,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("on_task_start: NER failed (%s); using original goal.", exc)
            self._task.ner_mapping = None
            self._task.anonymized_goal = goal
        _t_ner = time.perf_counter() - _t0_ner

        # Step 2: Extract task requirements via LLM + expand to ExpandedRequirements
        _t0_extract = time.perf_counter()
        try:
            raw_reqs = self._task_extractor.extract_v2(self._task.anonymized_goal)
            raw_reqs = deanonymize_requirement_app_names(
                raw_reqs,
                self._task.ner_mapping,
            )
            self._task.expanded_reqs = self._task_extractor.to_expanded(raw_reqs)
            logger.info(
                "on_task_start: expanded → %d req(s): %s",
                len(self._task.expanded_reqs.requirements),
                [(r.app_names, r.subject_type) for r in self._task.expanded_reqs.requirements],
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("on_task_start: task extraction failed (%s); using empty.", exc)
            self._task.expanded_reqs = ExpandedRequirements()
        _t_extract = time.perf_counter() - _t0_extract

        # Step 3: Create per-task LauncherPolicy singleton
        self._task.launcher_policy = LauncherPolicy(self._task.expanded_reqs)

        # Emit task-start timing line
        logger.info("[TIMING] %s", json.dumps({
            "type": "task_start",
            "task": self._task.task_name,
            "privacy": {
                "t_ner_s": round(_t_ner, 4),
                "t_task_extract_s": round(_t_extract, 4),
            },
        }))

        # Persist NER mapping to disk for debugging/audit.
        # output_dir may not be set yet at task start; _save_ner_mapping handles that.
        self._save_ner_mapping()

        return self._task.anonymized_goal

    def on_screenshot(self, pixels: np.ndarray, activity_name: str = "") -> np.ndarray:
        """Process a screenshot through the v3 privacy pipeline.

        Steps:
          1. POST /api/v1/detect → raw detections
          2. classify_screen(activity_name) → screen context
          3. For each detection: classify + decide exposure (+ VQA if needed)
          4. redact_image(pixels, detections) → anonymized pixels

        Side effects:
          - self.last_detection_json updated (with privacy_strategy per detection)
          - self.last_labeled_image_bytes updated if server provides overlay
          - self._last_original_pixels / _last_anon_pixels updated
          - self._task.step_counter incremented
        """
        # Reset per-step artifacts
        self.last_detection_json = None
        self.last_labeled_image_bytes = None
        self.last_server_log = None
        self._last_original_pixels = pixels.copy()
        # Reset classifier per-step accumulators
        self._privacy_classifier.vqa_time_s = 0.0
        self._privacy_classifier.embed_time_s = 0.0

        logger.info("[step %d] activity: %s", self._task.step_counter, activity_name or "(none)")

        expanded_reqs = self._task.expanded_reqs or ExpandedRequirements()

        try:
            # Encode screenshot to PNG bytes
            img = Image.fromarray(pixels.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            png_bytes = buf.getvalue()

            # Build multipart request for /api/v1/detect
            files: dict = {"image": ("screenshot.png", png_bytes, "image/png")}
            data: dict = {}
            if self._debug:
                data["debug"] = "true"
            _t0_detect = time.perf_counter()
            resp = self._session.post(
                f"{self._base_url}/api/v1/detect",
                files=files,
                data=data,
                timeout=self._timeout,
            )
            _t_detect = time.perf_counter() - _t0_detect
            resp.raise_for_status()
            detect_payload = resp.json()

            # Extract optional labeled overlay image
            b64_labeled = detect_payload.get("labeled_image", "")
            if b64_labeled:
                self.last_labeled_image_bytes = base64.b64decode(b64_labeled)

            # Server log (debug mode)
            if self._debug:
                self.last_server_log = detect_payload.get("server_log")

            detections = detect_payload.get("detections", [])

            # NER forward mapping — replace original entity names in OCR
            # content with anonymized placeholders so LLM relevance matching can
            # align detection content with task requirement subjects.
            if self._task.ner_mapping:
                _forward = {
                    orig: ph
                    for ph, orig in self._task.ner_mapping.get("reverse_mapping", {}).items()
                }
                if _forward:
                    for det in detections:
                        if det.get("source") == "ocr":
                            c = det.get("content", "")
                            replaced = False
                            for orig, ph in _forward.items():
                                if orig in c:
                                    if not replaced:
                                        det.setdefault("original_content", c)
                                    c = c.replace(orig, ph)
                                    replaced = True
                            if replaced:
                                det["content"] = c

        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "on_screenshot step %d: /api/v1/detect failed (%s); "
                "returning original pixels.",
                self._task.step_counter, exc,
            )
            self._last_screenshot_timing = {
                "t_detect_s": 0.0, "t_local_policy_and_prep_s": 0.0,
                "t_vqa_s": 0.0, "t_embed_s": 0.0, "t_redact_s": 0.0,
            }
            self._task.step_counter += 1
            self._last_anon_pixels = pixels
            return pixels

        # --- Local classification and exposure decision ---
        screen_ctx = sc_module.classify_screen(activity_name)
        self.last_activity_name = activity_name
        self.last_screen_ctx = screen_ctx

        pkg = activity_name.split("/")[0].strip() if activity_name else ""

        logger.debug(
            "on_screenshot step %d: activity='%s' → ctx=%s pkg=%s, %d detections",
            self._task.step_counter, activity_name, screen_ctx.value, pkg, len(detections),
        )

        h, w = pixels.shape[:2]

        _t0_local_policy = time.perf_counter()
        # ── Launcher: fully handled by LauncherPolicy (bypasses should_expose_v2) ──
        if screen_ctx == sc_module.ScreenContext.launcher:
            detections = merge_icon_text(detections)
            launcher_policy = self._task.launcher_policy or LauncherPolicy(expanded_reqs)
            launcher_policy.apply(detections)
            # VQA rescue: expose required-app icons that had no text label (dock icons).
            _vqa_rescue_launcher_icons(
                detections, launcher_policy, pixels,
                self._session, self._base_url, self._timeout,
            )
            # Skip should_expose_v2 loop; go directly to NER post-pass below.
        else:
            # ── Non-launcher screens: policy plan first, expensive image prep only on demand ──
            req_ctx = self._privacy_classifier.resolve_requirement_context(expanded_reqs, current_pkg=pkg)
            nav_only_candidates: list[dict] = []

            for i, det in enumerate(detections):
                plan = self._privacy_classifier.plan_detection_v2(
                    det, screen_ctx, req_ctx, elem_idx=i, pixels=pixels)

                image_crop_b64 = None
                if plan.needs_image_crop:
                    bbox = det.get("bbox", [])
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        px1 = max(0, int(x1 * w))
                        py1 = max(0, int(y1 * h))
                        px2 = min(w, int(x2 * w))
                        py2 = min(h, int(y2 * h))
                        if px2 > px1 and py2 > py1:
                            crop = pixels[py1:py2, px1:px2]
                            crop_img = Image.fromarray(crop.astype(np.uint8))
                            crop_buf = io.BytesIO()
                            crop_img.save(crop_buf, format="PNG")
                            image_crop_b64 = base64.b64encode(crop_buf.getvalue()).decode("utf-8")

                self._privacy_classifier.should_expose_v2(
                    det,
                    screen_ctx,
                    req_ctx,
                    plan=plan,
                    image_crop_b64=image_crop_b64,
                    elem_idx=i,
                )
                if det.get("privacy_strategy") == HARD_MASK and plan.defer_nav_only and image_crop_b64 is not None:
                    nav_only_candidates.append(
                        {"det": det, "image_crop_b64": image_crop_b64, "elem_idx": i}
                    )

            # Layer 2: batch text relevance check in TEXT_PRIVACY contexts
            if get_level(screen_ctx) == TEXT_PRIVACY:
                text_masked = [
                    d for d in detections
                    if d.get("type") not in {"image", "photo"}
                    and d.get("privacy_strategy") == HARD_MASK
                ]
                if text_masked:
                    logger.info(
                        "[on_screenshot] %s in TEXT_PRIVACY ctx; checking relevance of %d masked text elements",
                        screen_ctx.value, len(text_masked),
                    )
                    self._privacy_classifier.batch_text_relevance_v2(
                        text_masked, expanded_reqs, screen_ctx, current_pkg=pkg
                    )

            if nav_only_candidates:
                self._privacy_classifier.batch_nav_control_vqa_v2(nav_only_candidates)
        _t_local_policy_raw = time.perf_counter() - _t0_local_policy

        # NER post-pass: EXPOSE elements whose OCR text was NER-replaced should have
        # their placeholder rendered in the image rather than showing the real entity.
        # Applies to ALL text elements with NER replacements, regardless of prior strategy
        # (both EXPOSE and HARD_MASK become NER_REPLACE so the agent sees placeholders).
        ner_replaced_count = 0
        for det in detections:
            if ("original_content" in det
                    and det.get("type", "").lower() not in {"image", "photo"}):
                det["privacy_strategy"] = NER_REPLACE
                ner_replaced_count += 1
        if ner_replaced_count:
            logger.info("[on_screenshot] NER post-pass: %d element(s) → ner_replace", ner_replaced_count)

        # Compute per-step privacy summary
        exposed_dets = [d for d in detections if d.get("privacy_strategy") == EXPOSE]
        masked_dets = [d for d in detections if d.get("privacy_strategy") == HARD_MASK]
        self.last_privacy_summary = {
            "total": len(detections),
            "exposed": len(exposed_dets),
            "hard_masked": len(masked_dets),
            "hard_masked_elements": [
                {
                    "type": d.get("type"),
                    "bbox": d.get("bbox"),
                    "content": d.get("content", ""),
                }
                for d in masked_dets
            ],
        }

        # Apply image redaction locally
        _t0_redact = time.perf_counter()
        anon_pixels = image_redactor.redact_image(pixels, detections)
        _t_redact = time.perf_counter() - _t0_redact

        # Collect VQA and embed times from classifier accumulators
        _t_vqa = self._privacy_classifier.vqa_time_s
        _t_embed = self._privacy_classifier.embed_time_s
        # local policy/prep excludes the VQA/embed sub-components so the printed
        # fields remain additive and easier to read in timing reports.
        _t_local_policy = max(0.0, _t_local_policy_raw - _t_vqa - _t_embed)
        self._last_screenshot_timing = {
            # OCR + UIED detection.
            "t_detect_s":   round(_t_detect, 4),
            # Local policy resolution and image prep, excluding VQA/embed.
            "t_local_policy_and_prep_s": round(_t_local_policy, 4),
            # Image VQA time used by subject/nav-only image checks.
            "t_vqa_s":      round(_t_vqa, 4),
            # Text embedding/relevance time used by subject text checks.
            "t_embed_s":    round(_t_embed, 4),
            # Final screenshot redaction/rendering time.
            "t_redact_s":   round(_t_redact, 4),
        }

        # Store results
        self.last_detection_json = {"detections": detections}
        self._last_anon_pixels = anon_pixels

        # Auto-save debug artifacts when debug_dir is configured
        if self._debug_dir:
            self._save_debug_artifacts(anon_pixels)

        self._task.step_counter += 1
        return anon_pixels

    def on_action(
        self,
        action: dict,
        screen_elements=None,
        screen_size=None,
    ) -> Tuple[dict, bool]:
        """NER de-anonymization + Interactive Disclosure.

        Step 1 — NER reverse mapping: replace placeholders back to originals.
        Step 2 — Disclosure detection: tap on hard_mask region → inject revealed
                 screenshot, skip phone action.
        """
        # Step 1: NER de-anonymization
        if self._task.ner_mapping:
            reverse_mapping = self._task.ner_mapping.get("reverse_mapping", {})
            if reverse_mapping:
                _restored: list = []

                def _restore(v: str) -> str:
                    for ph, orig in reverse_mapping.items():
                        if ph in v:
                            _restored.append(f"{ph!r} → {orig!r}")
                            v = v.replace(ph, orig)
                    return v

                def _walk(obj):
                    if isinstance(obj, str):
                        return _restore(obj)
                    if isinstance(obj, dict):
                        return {k: _walk(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_walk(i) for i in obj]
                    return obj

                action = _walk(action)
                if _restored:
                    logger.info("[on_action] NER reverse: %d replacement(s): %s", len(_restored), ", ".join(_restored))

        # Step 2: Interactive Disclosure
        disclosed_det = self._find_disclosed_region(action, screen_elements, screen_size)
        if disclosed_det is not None:
            t0 = time.time()
            revealed_pixels = self._make_revealed_screenshot(disclosed_det)
            reveal_ms = int((time.time() - t0) * 1000)

            if revealed_pixels is not None:
                self.pending_override_pixels = revealed_pixels
                self._log_disclosure(action, disclosed_det, reveal_ms)
                logger.info(
                    "[on_action] disclosure: action=%s bbox=%s content=%r → injecting revealed screenshot",
                    action.get("action_type"),
                    disclosed_det.get("bbox"),
                    disclosed_det.get("content", ""),
                )
                return action, True  # skip execution on phone

        return action, False

    def on_task_end(self) -> None:
        """Clear all per-task state."""
        # Close log handler before resetting task state
        if self._task.log_handler is not None:
            self._task.log_handler.close()
            logging.getLogger().removeHandler(self._task.log_handler)
        self._task = _TaskState()
        self._task_extractor.clear_cache()
        # Clear per-step artifacts (public attrs read by callers between hook calls)
        self.last_detection_json = None
        self.last_labeled_image_bytes = None
        self.last_server_log = None
        self._last_original_pixels = None
        self._last_anon_pixels = None
        self.pending_override_pixels = None
        self.last_activity_name = ""
        self.last_screen_ctx = None
        self.last_privacy_summary = {}

    # ------------------------------------------------------------------
    # Disclosure helpers
    # ------------------------------------------------------------------

    def _find_disclosed_region(self, action: dict, screen_elements, screen_size):
        """Return the hard_mask detection whose eye-icon hotzone the tap action targets.

        Two changes vs. the original:
        - Only the top-right 60×60px hotzone (where the eye icon is drawn) triggers disclosure,
          not the entire masked bbox.
        - EXPOSE priority: if the click point is also covered by an EXPOSE element, skip disclosure.
        """
        if not self.last_detection_json:
            return None
        if action.get("action_type") not in ("click", "double_tap", "long_press"):
            return None

        ax, ay = action.get("x"), action.get("y")
        idx = action.get("index")
        if idx is not None and screen_elements:
            try:
                elem = screen_elements[int(idx)]
                if elem.bbox_pixels is not None:
                    ax, ay = elem.bbox_pixels.center
            except (IndexError, AttributeError):
                pass
        if ax is None or ay is None:
            return None

        if screen_size:
            sw, sh = screen_size
        elif self._last_original_pixels is not None:
            sh, sw = self._last_original_pixels.shape[:2]
        else:
            return None

        nx, ny = float(ax) / sw, float(ay) / sh

        all_detections = self.last_detection_json.get("detections", [])

        # Build set of EXPOSE bboxes covering the click point (for EXPOSE-priority check)
        expose_covers_click = any(
            det.get("privacy_strategy") == "expose"
            and len(det.get("bbox", [])) == 4
            and det["bbox"][0] <= nx <= det["bbox"][2]
            and det["bbox"][1] <= ny <= det["bbox"][3]
            for det in all_detections
        )

        for det in all_detections:
            if det.get("privacy_strategy") != "hard_mask":
                continue
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue

            # Compute eye-icon hotzone in pixel coords (mirrors _draw_eye_icon in image_redactor.py)
            x1, y1, x2, y2 = bbox
            px1 = max(0, int(x1 * sw))
            py1 = max(0, int(y1 * sh))
            px2 = min(sw, int(x2 * sw))
            py2 = min(sh, int(y2 * sh))
            icon_size = max(28, min(56, min(px2 - px1, py2 - py1) // 4))
            inset = 10
            # Icon center (top-right corner, same as _draw_eye_icon)
            icx = px2 - icon_size // 2 - inset
            icy = py1 + icon_size // 2 + inset
            # Hotzone = icon bounding box
            hz_x1 = (icx - icon_size // 2) / sw
            hz_y1 = (icy - icon_size // 2) / sh
            hz_x2 = (icx + icon_size // 2) / sw
            hz_y2 = (icy + icon_size // 2) / sh

            if not (hz_x1 <= nx <= hz_x2 and hz_y1 <= ny <= hz_y2):
                continue

            # EXPOSE priority: if click point also covered by an EXPOSE element, don't disclose
            if expose_covers_click:
                continue

            return det

        return None

    def _make_revealed_screenshot(self, det: dict) -> Optional[np.ndarray]:
        """Return anonymized screenshot with the detection bbox replaced by original."""
        if self._last_original_pixels is None or self._last_anon_pixels is None:
            return None

        h, w = self._last_original_pixels.shape[:2]
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            return None

        x1, y1, x2, y2 = bbox
        px1, py1 = max(0, int(x1 * w)), max(0, int(y1 * h))
        px2, py2 = min(w, int(x2 * w)), min(h, int(y2 * h))

        revealed = self._last_anon_pixels.copy()
        revealed[py1:py2, px1:px2] = self._last_original_pixels[py1:py2, px1:px2]
        return revealed

    def _log_disclosure(self, action: dict, det: dict, reveal_ms: int) -> None:
        """Append disclosure event to audit log and save revealed image."""
        if not self._output_dir:
            return

        log_dir = os.path.join(self._output_dir, "disclosure_logging")
        os.makedirs(log_dir, exist_ok=True)

        self._disclosure_counter += 1
        disclosure_id = str(uuid.uuid4())[:8]
        img_filename = f"disclosure_{self._disclosure_counter:03d}_revealed.png"

        override_px = self.pending_override_pixels
        if override_px is not None:
            img = Image.fromarray(override_px.astype(np.uint8))
            img.save(os.path.join(log_dir, img_filename))

        entry = {
            "disclosure_id": disclosure_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "action_type": action.get("action_type", "unknown"),
            "region_description": det.get("content", ""),
            "region_type": det.get("type", "unknown"),
            "privacy_strategy": det.get("privacy_strategy", ""),
            "bbox": det.get("bbox", []),
            "user_decision": "auto_approve",
            "reveal_duration_ms": reveal_ms,
            "revealed_image": img_filename,
        }

        log_path = os.path.join(log_dir, "disclosure_log.json")
        existing = []
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:  # pylint: disable=broad-except
                existing = []
        existing.append(entry)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        logger.info("Disclosure logged: %s → %s", disclosure_id, img_filename)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def check_health(self) -> bool:
        """Return True if the agentPrivacy server is reachable."""
        try:
            resp = self._session.get(
                f"{self._base_url}/api/v1/health",
                timeout=self._timeout,
            )
            return resp.status_code == 200
        except Exception:  # pylint: disable=broad-except
            return False
