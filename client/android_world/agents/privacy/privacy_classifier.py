"""Privacy exposure decision engine driven by the canonical policy table.

The final policy is resolved as:
  (effective_context, requirement_family, modality) -> element_policy

See context_policy.py for the canonical table and refined-context exceptions.

VQA 置信度阈值
--------------
VQA 接口返回 {"answer": "yes"/"no", "confidence": float}。
要求 confidence >= 0.6 才信任 "yes"（过滤边界情况的误报）。
如接口不含 confidence 字段（旧版兼容），默认置信度 1.0。

决策结果
--------
  EXPOSE    → "expose"（让 agent 看到原图）
  HARD_MASK → "hard_mask"（模糊/黑掉）
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import requests

from android_world.agents.privacy.context_policy import (
    MODALITY_IMAGE,
    MODALITY_TEXT,
    POLICY_EXPOSE,
    POLICY_KEEP_MASKED,
    POLICY_NAV_ONLY_VQA,
    POLICY_SUBJECT_MATCH,
    REQ_NONE,
    normalize_requirement_family,
    resolve_element_policy,
)
from android_world.agents.privacy.launcher_policy import _norm_app_name, _matches_app_name
from android_world.agents.privacy.utils import _is_tiny_icon, _is_searchbar_widget
from android_world.agents.privacy.screen_context import ScreenContext
from android_world.agents.privacy.task_extractor import (
    ExpandedRequirements, ExpandedRequirement,
)

logger = logging.getLogger(__name__)

# Decisions
EXPOSE = "expose"
HARD_MASK = "hard_mask"
NER_REPLACE = "ner_replace"  # OCR text NER-anonymized; render placeholder in image

# Element types considered "image content" (everything else is a control)
_IMAGE_TYPES = {"image", "photo"}

# VQA minimum confidence to trust a "yes" answer
_VQA_MIN_CONFIDENCE = 0.6

# Max concepts per server-side batch VQA call.  Must match _VQA_SERVER_BATCH_SIZE in routes.py.
_VQA_SERVER_BATCH_SIZE = 8

_NAV_CONTROL_PROMPT = (
    "a clickable UI navigation or action button, such as a plus button, add button, "
    "menu button, back button, save button, send button, search button, navigation "
    "tab button, or action button"
)

CHUNK_SIZE = 10  # max elements per LLM call in _relevance_llm

_FEW_SHOT_ELEMENTS = (
    '1: "14:30"\n'
    '2: "张三"\n'
    '3: "[合同草稿.pdf]"\n'
    '4: "李四 说：我最近失恋了"\n'
    '5: "王五 说：周末去哪玩"\n'
    '6: "消息"'
)


def _relevance_llm(  # noqa: F841  # Alternate backend; activate via _RELEVANCE_BACKEND = _relevance_llm
    remaining: list,
    subjects: list,
    screen_ctx,
    base_url: str,
    timeout: float,
) -> list[int]:
    """LLM 方案：返回应 EXPOSE 的 0-based 索引列表（相对于 remaining）。

    Internally splits *remaining* into chunks of CHUNK_SIZE and issues one LLM
    request per chunk, then merges the results back into global 0-based indices.

    Error handling:
      - Server unreachable / non-200 response → raises RuntimeError immediately.
        do not fail-open on detection error：服务端出错应直接暴露问题，而非掩盖。
      - LLM output unparseable (no valid JSON array) → exposes entire chunk
        (conservative fallback: treat all elements as relevant rather than
        silently masking content the agent needs).
    """
    session = requests.Session()
    subject_str = ", ".join(f'"{s}"' for s in subjects)

    expose_indices: list[int] = []

    for chunk_start in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[chunk_start: chunk_start + CHUNK_SIZE]

        # Build 1-indexed element list for this chunk
        elements_str = "\n".join(
            f'{i + 1}: {json.dumps(det.get("content", ""), ensure_ascii=False)}'
            for i, det in enumerate(chunk)
        )

        prompt = (
            "You are a privacy filter.\n"
            "Return ONLY a JSON array of 1-based indices of elements whose content "
            "directly contains or mentions the task target. "
            "When uncertain, include the element.\n\n"
            "Example 1:\n"
            f'Task requires: "张三", "合同"\n'
            f"Elements:\n{_FEW_SHOT_ELEMENTS}\n"
            "Answer: [2, 3]\n\n"
            "Example 2:\n"
            f'Task requires: "会议", "项目"\n'
            f"Elements:\n{_FEW_SHOT_ELEMENTS}\n"
            "Answer: []\n\n"
            f"Task requires: {subject_str}\n"
            f"Screen type: {screen_ctx.value}.\n"
            f"Elements:\n{elements_str}\n"
            "Answer:"
        )

        try:
            resp = session.post(
                f"{base_url}/api/v1/llm_2b",
                json={"prompt": prompt, "max_tokens": 128, "stop": ["\n\n", "Task"]},
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("text", "")
            logger.info("[text_rel] LLM raw (chunk %d-%d): %s", chunk_start, chunk_start + len(chunk) - 1, raw)

            m = re.search(r'\[[\d,\s]*\]', raw)
            if m:
                one_based = json.loads(m.group())
                for ob in one_based:
                    zb_in_chunk = int(ob) - 1          # 1-based → 0-based within chunk
                    if 0 <= zb_in_chunk < len(chunk):
                        expose_indices.append(chunk_start + zb_in_chunk)
            else:
                # Parse failed → expose entire chunk (fail open)
                logger.warning(
                    "[text_rel] could not parse LLM output %r; exposing all %d elements in chunk",
                    raw[:100], len(chunk),
                )
                expose_indices.extend(range(chunk_start, chunk_start + len(chunk)))

        except Exception as exc:
            raise RuntimeError(
                f"[text_rel] LLM call failed for chunk {chunk_start}-{chunk_start + len(chunk) - 1}"
            ) from exc

    return expose_indices


# Cosine similarity threshold for text semantic relevance (google/embeddinggemma-300m).
_EMBED_THRESHOLD: float = 0.60

# Stricter cosine similarity threshold for short launcher app-label matching.
_LAUNCHER_EMBED_THRESHOLD: float = 0.65


def _relevance_embedding(
    remaining: list,
    subjects: list,
    screen_ctx,
    base_url: str,
    timeout: float,
) -> list[int]:
    """Embedding 方案：调用 /api/v1/embed_gemma，用余弦相似度筛选相关元素。

    一次 HTTP 请求把 subjects + elements 全部编码，
    对每个 element 取与所有 subjects 的最大余弦相似度，
    超过 _EMBED_THRESHOLD 则 EXPOSE。

    Fail-open：server 不可用时 expose 所有 remaining 元素。
    """
    contents = [det.get("content", "") for det in remaining]
    all_texts = subjects + contents

    try:
        resp = requests.post(
            f"{base_url}/api/v1/embed_gemma",
            json={"texts": all_texts},
            timeout=timeout,
        )
        resp.raise_for_status()
        embs = resp.json().get("embeddings", [])
        if len(embs) != len(all_texts):
            raise ValueError(f"expected {len(all_texts)} embeddings, got {len(embs)}")
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "[text_rel] embed call failed (%s); exposing all %d remaining elements",
            exc, len(remaining),
        )
        return list(range(len(remaining)))

    n_subj = len(subjects)
    sub_embs = embs[:n_subj]
    elem_embs = embs[n_subj:]

    def _dot(a: list, b: list) -> float:
        return sum(x * y for x, y in zip(a, b))

    ctx_threshold = (
        _LAUNCHER_EMBED_THRESHOLD
        if screen_ctx == ScreenContext.launcher
        else _EMBED_THRESHOLD
    )

    expose_indices: list[int] = []
    for i, emb in enumerate(elem_embs):
        best_sim = max(_dot(emb, se) for se in sub_embs)
        if best_sim >= ctx_threshold:
            expose_indices.append(i)
            logger.debug(
                "[text_rel] embed EXPOSE #%d %r sim=%.3f",
                i, contents[i][:40], best_sim,
            )

    logger.info(
        "[text_rel] embed: %d / %d elements exposed (threshold=%.2f, ctx=%s)",
        len(expose_indices), len(remaining), ctx_threshold,
        screen_ctx.value if hasattr(screen_ctx, "value") else str(screen_ctx),
    )
    return expose_indices


# 切换后端：将此变量指向 _relevance_llm 或 _relevance_embedding
_RELEVANCE_BACKEND = _relevance_embedding


def _find_requirement(
    expanded_reqs: ExpandedRequirements,
    current_pkg: Optional[str],
) -> Optional[ExpandedRequirement]:
    """Return the first ExpandedRequirement matching current_pkg via reverse lookup.

    Two-layer matching (either hits → match):
      1. DB reverse: current_pkg → DB entry → registered names → fuzzy match app_name
      2. Token match: ALL significant tokens (≥4 chars) from app_name in pkg string

    This replaces the old expand()-based forward matching that suffered from
    false package resolution (e.g. "Simple SMS Messenger" → com.facebook.orca).
    """
    if not current_pkg:
        return None
    pkg_lower = current_pkg.lower()

    # DB reverse: pkg → DB entry → registered display names
    db_names = _get_db_names_for_pkg(current_pkg)

    for req in expanded_reqs.requirements:
        for app_name in req.app_names:
            norm_app = _norm_app_name(app_name)

            # Check 1: DB registered names match (via _matches_app_name)
            for db_name in db_names:
                if _matches_app_name(_norm_app_name(db_name), norm_app):
                    return req

            # Check 2: ALL significant tokens from app_name appear in pkg
            tokens = [t.lower() for t in app_name.split() if len(t) >= 4]
            if tokens and all(t in pkg_lower for t in tokens):
                return req

    return None


def _get_db_names_for_pkg(pkg: str) -> list:
    """Look up registered display names for a package in app_context_db.json."""
    from android_world.agents.privacy.screen_context import _load_app_db  # noqa: PLC0415
    from android_world.agents.privacy.launcher_policy import (  # noqa: PLC0415
        _norm_app_name,
        _matches_app_name,
    )
    db = _load_app_db()
    entry = db.get(pkg)
    if isinstance(entry, dict):
        return entry.get("names", [])
    # Try prefix match (e.g. com.tencent.mm.plugin → com.tencent.mm)
    pkg_prefix = pkg
    while "." in pkg_prefix:
        pkg_prefix = pkg_prefix.rsplit(".", 1)[0]
        entry = db.get(pkg_prefix)
        if isinstance(entry, dict):
            return entry.get("names", [])
    return []


@dataclass(frozen=True)
class RequirementContext:
    """Resolved requirement info for the current foreground package."""

    req: Optional[ExpandedRequirement]
    req_family: str
    subj_type: str
    subject: str


@dataclass(frozen=True)
class DetectionPlan:
    """Cheap per-detection plan used to avoid eager image crop preparation."""

    policy: str
    is_image: bool
    tag: str
    content: str
    bbox: list
    subj_type: str
    needs_image_crop: bool
    needs_subject_image_vqa: bool
    defer_nav_only: bool


class PrivacyClassifier:
    """Decide exposure/masking for each detected element.

    Args:
        server_base_url: Base URL of the agentPrivacy server (for VQA calls).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(self, server_base_url: str = "http://localhost:8001", timeout: float = 30.0):
        self._base_url = server_base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        # Per-step timing accumulators (reset by caller before each step)
        self.vqa_time_s: float = 0.0
        self.embed_time_s: float = 0.0

    def resolve_requirement_context(
        self,
        expanded_reqs: ExpandedRequirements,
        current_pkg: Optional[str] = None,
    ) -> RequirementContext:
        """Resolve requirement context once per screen for the current foreground package."""
        req = _find_requirement(expanded_reqs, current_pkg)
        if req is None:
            return RequirementContext(
                req=None,
                req_family=REQ_NONE,
                subj_type="none",
                subject="none",
            )
        return RequirementContext(
            req=req,
            req_family=normalize_requirement_family(req.subject_type),
            subj_type=req.subject_type,
            subject=req.subject,
        )

    def plan_detection_v2(
        self,
        det: dict,
        screen_ctx: ScreenContext,
        req_ctx: RequirementContext,
        elem_idx: int = -1,
        pixels: Optional[np.ndarray] = None,
    ) -> DetectionPlan:
        """Cheap local policy plan for one detection, without image crop preparation.

        Args:
            pixels: Full-screen RGB numpy array (H×W×3).  When provided,
                    the search-bar widget check applies pixel colour uniformity
                    (std-dev) in addition to geometry, eliminating false
                    positives from thin banner images.  Pass ``None`` only
                    when pixels are genuinely unavailable.
        """
        elem_type = det.get("type", "").lower()
        content = det.get("content", "")
        bbox = det.get("bbox", [])
        tag = f"#{'?' if elem_idx < 0 else elem_idx}"
        is_image = elem_type in _IMAGE_TYPES

        # Fast-path 1: tiny icon / status-bar decoration — always expose.
        if is_image and _is_tiny_icon(bbox):
            return DetectionPlan(
                policy=POLICY_EXPOSE,
                is_image=True,
                tag=tag,
                content=content,
                bbox=bbox,
                subj_type=req_ctx.subj_type,
                needs_image_crop=False,
                needs_subject_image_vqa=False,
                defer_nav_only=False,
            )

        # Fast-path 2: search-bar widget — wide + short + colour-uniform strip.
        # Pixel std check (requires pixels) prevents false positives from thin banners.
        # type=icon search bars are rescued separately by batch_text_relevance_v2.
        if is_image and _is_searchbar_widget(bbox, pixels):
            logger.info("[plan_v2] %s image bbox=%s → EXPOSE (search-bar widget)",
                        tag, bbox)
            return DetectionPlan(
                policy=POLICY_EXPOSE,
                is_image=True,
                tag=tag,
                content=content,
                bbox=bbox,
                subj_type=req_ctx.subj_type,
                needs_image_crop=False,
                needs_subject_image_vqa=False,
                defer_nav_only=False,
            )

        modality = MODALITY_IMAGE if is_image else MODALITY_TEXT
        policy = resolve_element_policy(screen_ctx, req_ctx.req_family, modality, req_ctx.subj_type)
        return DetectionPlan(
            policy=policy,
            is_image=is_image,
            tag=tag,
            content=content,
            bbox=bbox,
            subj_type=req_ctx.subj_type,
            needs_image_crop=is_image and policy in {POLICY_SUBJECT_MATCH, POLICY_NAV_ONLY_VQA},
            needs_subject_image_vqa=is_image and policy == POLICY_SUBJECT_MATCH,
            defer_nav_only=is_image and policy == POLICY_NAV_ONLY_VQA,
        )

    def should_expose_v2(
        self,
        det: dict,
        screen_ctx: ScreenContext,
        req_ctx: RequirementContext,
        plan: Optional[DetectionPlan] = None,
        image_crop_b64: Optional[str] = None,
        elem_idx: int = -1,
        pixels: Optional[np.ndarray] = None,
    ) -> bool:
        """Exposure decision driven entirely by the canonical policy table.

        Decision logic:
          1. Tiny images and search-bar widgets are always exposed (UI chrome fast-paths
             in plan_detection_v2; not a policy decision).
          2. All other elements: resolve_element_policy(ctx, req_family, modality, subj_type).
             req=None is treated as REQ_NONE (pure-navigation, no content unlocked).
          3. For image SUBJECT_MATCH outcomes: run VQA inline.
          4. For image NAV_ONLY_VQA outcomes: keep masked now; batch_nav_control_vqa_v2
             upgrades matched controls after the full-screen loop.
          5. For text SUBJECT_MATCH/KEEP_MASKED: set HARD_MASK now; batch_text_relevance_v2
             upgrades SUBJECT_MATCH elements after the full-screen loop (TEXT_PRIVACY only).

        Launcher screen is NOT handled here — it bypasses this method entirely
        and is handled by LauncherPolicy.apply() in server_hooks.

        Side effect: sets det["privacy_strategy"] to EXPOSE or HARD_MASK.
        """
        plan = plan or self.plan_detection_v2(det, screen_ctx, req_ctx, elem_idx, pixels=pixels)
        elem_type = det.get("type", "").lower()
        policy = plan.policy

        if policy == POLICY_EXPOSE:
            det["privacy_strategy"] = EXPOSE
            logger.info("[cls_v2] %s %s %r → EXPOSE (policy=%s, ctx=%s, type=%s)",
                        plan.tag, elem_type, plan.content, policy, screen_ctx.value, plan.subj_type)
            return True

        if plan.is_image:
            if plan.needs_subject_image_vqa and image_crop_b64 is not None:
                try:
                    matched = self._vqa_single_batch(image_crop_b64, [req_ctx.subject])
                    if matched:
                        det["privacy_strategy"] = EXPOSE
                        logger.info("[cls_v2] %s image %r bbox=%s → EXPOSE (VQA subject=%r)",
                                    plan.tag, plan.content, plan.bbox, req_ctx.subject)
                        return True
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("[cls_v2] subject VQA failed (%s); treating as no match.", exc)

        # KEEP_MASKED, SUBJECT_MATCH (text, deferred), or failed VQA.
        det["privacy_strategy"] = HARD_MASK
        logger.info("[cls_v2] %s %s %r → HARD_MASK (policy=%s, ctx=%s, type=%s)",
                    plan.tag, elem_type, plan.content, policy, screen_ctx.value, plan.subj_type)
        return False

    def batch_nav_control_vqa_v2(self, candidates: list[dict]) -> None:
        """Upgrade nav-only image candidates to EXPOSE using one shared prompt per crop."""
        if not candidates:
            return

        for start in range(0, len(candidates), _VQA_SERVER_BATCH_SIZE):
            chunk = candidates[start:start + _VQA_SERVER_BATCH_SIZE]
            try:
                matches = self._vqa_prompt_batch(
                    [candidate["image_crop_b64"] for candidate in chunk],
                    _NAV_CONTROL_PROMPT,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("[nav_only_vqa] chunk %d-%d failed (%s); leaving masked",
                               start, start + len(chunk) - 1, exc)
                continue

            for candidate, matched in zip(chunk, matches):
                if not matched:
                    continue
                det = candidate["det"]
                det["privacy_strategy"] = EXPOSE
                tag = f"#{candidate.get('elem_idx', '?')}"
                logger.info("[cls_v2] %s image %r bbox=%s → EXPOSE (nav_only_vqa=%r)",
                            tag, det.get("content", ""), det.get("bbox", []), _NAV_CONTROL_PROMPT)

    def batch_text_relevance_v2(
        self,
        text_dets: list,
        expanded_reqs: ExpandedRequirements,
        screen_ctx: ScreenContext,
        current_pkg: Optional[str] = None,
    ) -> None:
        """v2 version: upgrade relevant HARD_MASK text elements → EXPOSE.

        Uses ExpandedRequirements + current_pkg (package name) for matching.
        Called after should_expose_v2 loop when screen_ctx is TEXT_PRIVACY.
        """
        if not text_dets:
            return

        req = _find_requirement(expanded_reqs, current_pkg)

        # Icons and trivial elements are always exposed (UI chrome, not content)
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
            logger.info("[text_rel_v2] pre-exposed %d icon/trivial elements", non_text_exposed)

        if req is None:
            logger.info("[text_rel_v2] no requirement for pkg=%s; %d text elements stay masked",
                        current_pkg, len(genuine_text_dets))
            return

        requirement_family = normalize_requirement_family(req.subject_type)
        text_policy = resolve_element_policy(
            screen_ctx, requirement_family, MODALITY_TEXT, req.subject_type
        )

        if text_policy == POLICY_EXPOSE:
            for det in genuine_text_dets:
                det["privacy_strategy"] = EXPOSE
            logger.info("[text_rel_v2] policy=%s → exposed %d text elements (ctx=%s type=%s)",
                        text_policy, len(genuine_text_dets), screen_ctx.value, req.subject_type)
            return

        if text_policy == POLICY_KEEP_MASKED:
            logger.info("[text_rel_v2] policy=%s → %d text elements stay masked",
                        text_policy,
                        len(genuine_text_dets))
            return

        # subject_match: run relevance check
        if not genuine_text_dets:
            return

        subjects = [req.subject]

        # Deterministic pre-pass: direct substring match
        pre_exposed = 0
        for det in genuine_text_dets:
            if any(subj.lower() in det.get("content", "").strip().lower() for subj in subjects):
                det["privacy_strategy"] = EXPOSE
                pre_exposed += 1
        if pre_exposed:
            logger.info("[text_rel_v2] pre-exposed %d elements (direct subject match)", pre_exposed)

        remaining = [d for d in genuine_text_dets if d["privacy_strategy"] != EXPOSE]
        if not remaining:
            return

        _t0_embed = time.perf_counter()
        indices = _RELEVANCE_BACKEND(remaining, subjects, screen_ctx, self._base_url, self._timeout)
        self.embed_time_s += time.perf_counter() - _t0_embed
        exposed = 0
        for idx in indices:
            if 0 <= idx < len(remaining):
                remaining[idx]["privacy_strategy"] = EXPOSE
                exposed += 1
        logger.info("[text_rel_v2] exposed %d / %d text elements (backend)", exposed, len(remaining))

    # ──────────────────────────────────────────────────────────────────────────
    # VQA backends — three strategies, production uses _vqa_single_batch
    # ──────────────────────────────────────────────────────────────────────────

    def _vqa_single(self, image_crop_b64: str, concept: str) -> bool:
        """Single-concept VQA: one HTTP call to /api/v1/llm_vqa/single.

        Args:
            image_crop_b64: Base64-encoded PNG crop.
            concept: Concept to verify (e.g. "王者荣耀").

        Returns:
            True if VQA answers "yes".
        """
        resp = self._session.post(
            f"{self._base_url}/api/v1/llm_vqa/single",
            json={"image_crop": image_crop_b64, "concept": concept},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        answer = body.get("answer", "").lower().strip()
        logger.debug("[vqa_single] concept=%r → %s (raw=%r)", concept, answer, body.get("raw", "")[:60])
        return answer == "yes"

    def _vqa_nav_controls(self, image_crop_b64: str) -> list[str]:
        """Backward-compatible helper for tests/debugging of the nav-only prompt."""
        matched = self._vqa_prompt_batch([image_crop_b64], _NAV_CONTROL_PROMPT)
        return [_NAV_CONTROL_PROMPT] if matched and matched[0] else []

    def _vqa_prompt_batch(self, image_crops_b64: list[str], prompt: str) -> list[bool]:
        """Batch one shared prompt over many image crops via /api/v1/llm_vqa/single_batch."""
        if not image_crops_b64:
            return []

        matched: list[bool] = []
        for start in range(0, len(image_crops_b64), _VQA_SERVER_BATCH_SIZE):
            chunk = image_crops_b64[start:start + _VQA_SERVER_BATCH_SIZE]
            items = [{"image_crop": crop_b64, "concept": prompt} for crop_b64 in chunk]
            _t0_vqa = time.perf_counter()
            resp = self._session.post(
                f"{self._base_url}/api/v1/llm_vqa/single_batch",
                json={"items": items},
                timeout=self._timeout,
            )
            self.vqa_time_s += time.perf_counter() - _t0_vqa
            resp.raise_for_status()
            results = resp.json().get("results", [])
            for result in results:
                matched.append(self._result_is_yes(result))
            if len(results) < len(chunk):
                matched.extend([False] * (len(chunk) - len(results)))

        logger.info("[vqa_prompt_batch] %d/%d crops matched prompt=%r",
                    sum(1 for item in matched if item), len(image_crops_b64), prompt)
        return matched

    def _result_is_yes(self, result: dict) -> bool:
        """Treat missing confidence as legacy-compatible yes/no behavior."""
        if result.get("answer") != "yes":
            return False
        confidence = result.get("confidence")
        if confidence is None:
            return True
        try:
            return float(confidence) >= _VQA_MIN_CONFIDENCE
        except (TypeError, ValueError):
            return False

    def _vqa_single_batch(self, image_crop_b64: str, concepts: list[str]) -> list[str]:
        """True server-side batch VQA via /api/v1/llm_vqa/single_batch.

        Sends all (image, concept) pairs in one HTTP request; the server runs a
        single model.generate(batch_size=N) call instead of N sequential calls,
        reducing wall-clock time to roughly that of a single inference.

        If len(concepts) > _VQA_SERVER_BATCH_SIZE, splits into sequential chunks.

        This is the production VQA strategy used by should_expose_v2().

        Args:
            image_crop_b64: Base64-encoded PNG crop (shared across all calls).
            concepts: List of concepts to check.

        Returns:
            Subset of *concepts* that VQA confirmed present (answered "yes").
        """
        if not concepts:
            return []

        matched: list[str] = []
        for start in range(0, len(concepts), _VQA_SERVER_BATCH_SIZE):
            chunk = concepts[start:start + _VQA_SERVER_BATCH_SIZE]
            items = [{"image_crop": image_crop_b64, "concept": c} for c in chunk]
            try:
                _t0_vqa = time.perf_counter()
                resp = self._session.post(
                    f"{self._base_url}/api/v1/llm_vqa/single_batch",
                    json={"items": items},
                    timeout=self._timeout,
                )
                self.vqa_time_s += time.perf_counter() - _t0_vqa
                resp.raise_for_status()
                for concept, result in zip(chunk, resp.json().get("results", [])):
                    if self._result_is_yes(result):
                        matched.append(concept)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("[vqa_single_batch] chunk %d-%d failed (%s); skipping",
                               start, start + len(chunk) - 1, exc)

        logger.info("[vqa_single_batch] %d/%d concepts matched: %s", len(matched), len(concepts), matched)
        return matched

    def _vqa_multi(self, image_crop_b64: str, concepts: list[str]) -> list[str]:
        """Multi-concept VQA: one HTTP call to /api/v1/llm_vqa/multi.

        Sends all concepts in a single request and asks the model to list which
        are present.  Fastest in terms of HTTP round-trips (1-pass inference),
        but requires a model large enough to reliably follow the per-item format.
        Qwen3.5-0.8B achieves ~57% F1 on this task; use _vqa_single_batch for
        production until a larger model is deployed.

        Args:
            image_crop_b64: Base64-encoded PNG crop.
            concepts: List of concepts to check.

        Returns:
            Subset of *concepts* that the model reported as present.
        """
        if not concepts:
            return []
        resp = self._session.post(
            f"{self._base_url}/api/v1/llm_vqa/multi",
            json={"image_crop": image_crop_b64, "concepts": concepts},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        present = body.get("present", [])
        logger.info("[vqa_multi] %d/%d concepts matched: %s (raw=%r)",
                    len(present), len(concepts), present, body.get("raw", "")[:80])
        return present
