"""Ablation hooks for the privacy benchmark expansion."""

import base64
import io
import json
import logging
import time

import numpy as np
from PIL import Image

from android_world.agents.privacy import image_redactor
from android_world.agents.privacy.context_policy import (
    MODALITY_TEXT,
    POLICY_EXPOSE,
    POLICY_KEEP_MASKED,
    normalize_requirement_family,
    resolve_element_policy,
)
from android_world.agents.privacy.privacy_classifier import (
    EXPOSE,
    HARD_MASK,
    NER_REPLACE,
    PrivacyClassifier,
)
from android_world.agents.privacy.launcher_policy import LauncherPolicy
from android_world.agents.privacy.server_hooks import ServerPrivacyHooks
from android_world.agents.privacy.task_extractor import ExpandedRequirements

logger = logging.getLogger(__name__)


class _NoElemVerifyPrivacyClassifier(PrivacyClassifier):
    """PrivacyClassifier variant with VQA and semantic text rescue disabled."""

    def _vqa_single_batch(self, image_crop_b64: str, concepts: list[str]) -> list[str]:
        logger.info("[no_elem_verify] disabled subject VQA for %d concept(s)", len(concepts))
        return []

    def _vqa_prompt_batch(self, image_crops_b64: list[str], prompt: str) -> list[bool]:
        logger.info("[no_elem_verify] disabled nav-control VQA for %d crop(s)", len(image_crops_b64))
        return [False] * len(image_crops_b64)

    def batch_text_relevance_v2(
        self,
        text_dets: list,
        expanded_reqs,
        screen_ctx,
        current_pkg: str | None = None,
    ) -> None:
        """Keep policy-level text exposure but disable subject-match rescue."""
        if not text_dets:
            return

        req_ctx = self.resolve_requirement_context(expanded_reqs, current_pkg=current_pkg)

        non_text_exposed = 0
        genuine_text_dets = []
        for det in text_dets:
            content = det.get("content", "")
            if det.get("type", "").lower() == "icon" or not content.strip() or len(content.strip()) <= 1:
                det["privacy_strategy"] = EXPOSE
                non_text_exposed += 1
            else:
                genuine_text_dets.append(det)
        if non_text_exposed:
            logger.info("[no_elem_verify] pre-exposed %d icon/trivial text elements", non_text_exposed)

        if req_ctx.req is None:
            logger.info(
                "[no_elem_verify] no requirement for pkg=%s; %d text elements stay masked",
                current_pkg,
                len(genuine_text_dets),
            )
            return

        requirement_family = normalize_requirement_family(req_ctx.req.subject_type)
        text_policy = resolve_element_policy(
            screen_ctx, requirement_family, MODALITY_TEXT, req_ctx.req.subject_type
        )

        if text_policy == POLICY_EXPOSE:
            for det in genuine_text_dets:
                det["privacy_strategy"] = EXPOSE
            logger.info(
                "[no_elem_verify] policy=%s → exposed %d text elements",
                text_policy,
                len(genuine_text_dets),
            )
            return

        if text_policy == POLICY_KEEP_MASKED:
            logger.info(
                "[no_elem_verify] policy=%s → %d text elements stay masked",
                text_policy,
                len(genuine_text_dets),
            )
            return

        subjects = [req_ctx.req.subject]
        pre_exposed = 0
        for det in genuine_text_dets:
            content = det.get("content", "").strip().lower()
            if any(subj.lower() in content for subj in subjects):
                det["privacy_strategy"] = EXPOSE
                pre_exposed += 1
            else:
                det["privacy_strategy"] = HARD_MASK
        if pre_exposed:
            logger.info("[no_elem_verify] pre-exposed %d elements (direct subject match)", pre_exposed)
        logger.info(
            "[no_elem_verify] disabled semantic text rescue; %d text elements stay masked",
            len(genuine_text_dets) - pre_exposed,
        )


class TextOnlyHooks(ServerPrivacyHooks):
    """OCR/NER-only text redaction without visual masking or task matching."""

    def on_task_start(self, goal: str) -> str:
        """NER-anonymize the task goal, skipping task extraction entirely."""
        self._task.step_counter = 0

        t0_ner = time.perf_counter()
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
                "[text_only] NER: %d entities found; anonymized goal: %s",
                len(payload.get("entities", [])),
                self._task.anonymized_goal,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[text_only] on_task_start: NER failed (%s); using original goal.", exc)
            self._task.ner_mapping = None
            self._task.anonymized_goal = goal
        t_ner = time.perf_counter() - t0_ner

        self._task.expanded_reqs = None
        self._task.launcher_policy = None

        logger.info("[TIMING] %s", json.dumps({
            "type": "task_start",
            "task": self._task.task_name,
            "privacy": {
                "mode": "text_only",
                "t_ner_s": round(t_ner, 4),
                "t_task_extract_s": 0.0,
            },
        }))

        self._save_ner_mapping()
        return self._task.anonymized_goal

    def on_screenshot(self, pixels: np.ndarray, activity_name: str = "") -> np.ndarray:
        """Replace OCR-detected PII text with placeholders; expose everything else."""
        self.last_detection_json = None
        self.last_labeled_image_bytes = None
        self.last_server_log = None
        self._last_original_pixels = pixels.copy()
        self.last_activity_name = activity_name
        self.last_screen_ctx = None

        logger.info("[text_only step %d] activity: %s", self._task.step_counter, activity_name or "(none)")

        try:
            img = Image.fromarray(pixels.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            files = {"image": ("screenshot.png", png_bytes, "image/png")}
            data = {"debug": "true"} if self._debug else {}
            t0_detect = time.perf_counter()
            resp = self._session.post(
                f"{self._base_url}/api/v1/detect",
                files=files,
                data=data,
                timeout=self._timeout,
            )
            t_detect = time.perf_counter() - t0_detect
            resp.raise_for_status()
            detect_payload = resp.json()

            b64_labeled = detect_payload.get("labeled_image", "")
            if b64_labeled:
                self.last_labeled_image_bytes = base64.b64decode(b64_labeled)
            if self._debug:
                self.last_server_log = detect_payload.get("server_log")

            detections = detect_payload.get("detections", [])
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "[text_only] on_screenshot step %d: /api/v1/detect failed (%s); returning original pixels.",
                self._task.step_counter, exc,
            )
            self._last_screenshot_timing = {
                "t_detect_s": 0.0,
                "t_local_policy_and_prep_s": 0.0,
                "t_vqa_s": 0.0,
                "t_embed_s": 0.0,
                "t_redact_s": 0.0,
            }
            self._task.step_counter += 1
            self._last_anon_pixels = pixels
            return pixels

        t0_ner = time.perf_counter()
        replaced_count = 0
        for det in detections:
            det["privacy_strategy"] = EXPOSE
            content = det.get("content", "")
            if not content or not isinstance(content, str):
                continue
            if det.get("source") != "ocr":
                continue

            payload = {"text": content}
            if self._task.ner_mapping:
                payload["ner_mapping"] = json.dumps(self._task.ner_mapping, ensure_ascii=False)

            try:
                ner_resp = self._session.post(
                    f"{self._base_url}/api/v1/ner/process",
                    data=payload,
                    timeout=self._timeout,
                )
                ner_resp.raise_for_status()
                ner_payload = ner_resp.json()
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("[text_only] OCR NER failed for %r (%s); exposing original text.", content, exc)
                continue

            anonymized = ner_payload.get("anonymized_text", content)
            self._task.ner_mapping = ner_payload.get("ner_mapping") or self._task.ner_mapping
            if anonymized != content:
                det["original_content"] = content
                det["content"] = anonymized
                det["privacy_strategy"] = NER_REPLACE
                replaced_count += 1
        t_ner = time.perf_counter() - t0_ner

        self.last_privacy_summary = {
            "mode": "text_only",
            "total": len(detections),
            "ner_replaced": replaced_count,
            "hard_masked": 0,
            "hard_masked_elements": [],
        }

        t0_redact = time.perf_counter()
        anon_pixels = image_redactor.redact_image(pixels, detections)
        t_redact = time.perf_counter() - t0_redact

        self._last_screenshot_timing = {
            "t_detect_s": round(t_detect, 4),
            "t_local_policy_and_prep_s": round(t_ner, 4),
            "t_vqa_s": 0.0,
            "t_embed_s": 0.0,
            "t_redact_s": round(t_redact, 4),
        }

        self.last_detection_json = {"detections": detections}
        self._last_anon_pixels = anon_pixels
        self._save_ner_mapping()

        if self._debug_dir:
            self._save_debug_artifacts(anon_pixels)

        logger.info("[text_only] NER post-pass: %d OCR element(s) → ner_replace", replaced_count)
        self._task.step_counter += 1
        return anon_pixels


class NoElemVerifyHooks(ServerPrivacyHooks):
    """CAPED without VQA or embedding-based element verification."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._privacy_classifier = _NoElemVerifyPrivacyClassifier(
            self._base_url,
            self._timeout,
        )


class NoTaskExtractHooks(ServerPrivacyHooks):
    """CAPED without task requirement extraction or requirement matching."""

    def on_task_start(self, goal: str) -> str:
        """NER-anonymize the task goal, then force empty task requirements."""
        self._task.step_counter = 0

        t0_ner = time.perf_counter()
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
                "[no_task_extract] NER: %d entities found; anonymized goal: %s",
                len(payload.get("entities", [])),
                self._task.anonymized_goal,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[no_task_extract] on_task_start: NER failed (%s); using original goal.", exc)
            self._task.ner_mapping = None
            self._task.anonymized_goal = goal
        t_ner = time.perf_counter() - t0_ner

        self._task.expanded_reqs = ExpandedRequirements()
        self._task.launcher_policy = LauncherPolicy(self._task.expanded_reqs)

        logger.info("[TIMING] %s", json.dumps({
            "type": "task_start",
            "task": self._task.task_name,
            "privacy": {
                "mode": "no_task_extract",
                "t_ner_s": round(t_ner, 4),
                "t_task_extract_s": 0.0,
            },
        }))

        self._save_ner_mapping()
        return self._task.anonymized_goal
