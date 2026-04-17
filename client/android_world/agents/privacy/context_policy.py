"""context_policy.py — canonical privacy decision table.

This module owns the final policy mapping:

  (effective_context, requirement_family, modality) -> element_policy

Canonical table
---------------
| requirement | PUBLIC | PRIVACY | TEXT_PRIVACY |
|---|---|---|---|
| `none` text | expose | expose | keep_masked |
| `none` image | expose | keep_masked | keep_masked |
| `subject` text | expose | expose | subject_match |
| `subject` image | expose | subject_match | subject_match |
| `all` text | expose | expose | expose |
| `all` image | expose | context-specific (`show_all_images` / `nav_only_vqa`) | context-specific (`show_all_images` / `nav_only_vqa`) |

Refined-screen exceptions:
  - messaging_list + person + image -> subject_match
  - messaging_conversation + person + text/image -> expose

Implementation intentionally uses merged branches rather than a literal giant
dict, but the result must stay equivalent to the table above.
"""

from android_world.agents.privacy.screen_context import ScreenContext

# ── Privacy level constants ────────────────────────────────────────────────

PUBLIC = "public"
PRIVACY = "privacy"
TEXT_PRIVACY = "text_privacy"

# ── Requirement family constants ───────────────────────────────────────────

REQ_NONE = "none"
REQ_SUBJECT = "subject"
REQ_ALL = "all"

# ── Modality constants ─────────────────────────────────────────────────────

MODALITY_TEXT = "text"
MODALITY_IMAGE = "image"

# ── Final element policy constants ─────────────────────────────────────────

POLICY_EXPOSE = "expose"
POLICY_KEEP_MASKED = "keep_masked"
POLICY_SUBJECT_MATCH = "subject_match"
POLICY_NAV_ONLY_VQA = "nav_only_vqa"

# ── All-image mode constants for requirement_family=all,image ─────────────

SHOW_ALL_IMAGES = "show_all_images"
NAV_ONLY_VQA = POLICY_NAV_ONLY_VQA


CONTEXT_LEVEL: dict[ScreenContext, str] = {
    ScreenContext.tool: PUBLIC,
    ScreenContext.game: PUBLIC,
    ScreenContext.public_content: PUBLIC,
    ScreenContext.settings: PUBLIC,
    ScreenContext.camera: PUBLIC,
    ScreenContext.unknown: PUBLIC,
    ScreenContext.gallery: PRIVACY,
    ScreenContext.financial: PRIVACY,
    ScreenContext.map_view: PRIVACY,
    ScreenContext.file: PRIVACY,
    ScreenContext.messaging: TEXT_PRIVACY,
    ScreenContext.messaging_list: TEXT_PRIVACY,
    ScreenContext.messaging_conversation: TEXT_PRIVACY,
    ScreenContext.contacts: TEXT_PRIVACY,
    ScreenContext.personal_profile: TEXT_PRIVACY,
    ScreenContext.content_feed: TEXT_PRIVACY,
    ScreenContext.health: TEXT_PRIVACY,
    ScreenContext.launcher: TEXT_PRIVACY,
}


ALL_IMAGE_MODE_BY_CONTEXT: dict[ScreenContext, str] = {
    ScreenContext.settings: SHOW_ALL_IMAGES,
    ScreenContext.messaging: SHOW_ALL_IMAGES,
    ScreenContext.messaging_list: SHOW_ALL_IMAGES,
    ScreenContext.messaging_conversation: SHOW_ALL_IMAGES,
    ScreenContext.map_view: SHOW_ALL_IMAGES,
    ScreenContext.file: SHOW_ALL_IMAGES,
    ScreenContext.contacts: SHOW_ALL_IMAGES,
    ScreenContext.financial: NAV_ONLY_VQA,
    ScreenContext.health: NAV_ONLY_VQA,
    ScreenContext.personal_profile: NAV_ONLY_VQA,
    ScreenContext.gallery: NAV_ONLY_VQA,
    ScreenContext.content_feed: NAV_ONLY_VQA,
}


def get_level(ctx: ScreenContext) -> str:
    """Return the privacy level for a screen context.

    Raises KeyError for any ScreenContext not listed in CONTEXT_LEVEL so that a
    newly-added enum value is caught immediately rather than silently defaulting
    to PUBLIC (which would expose everything on that screen).
    """
    return CONTEXT_LEVEL[ctx]


def get_all_image_mode(ctx: ScreenContext) -> str:
    """Return the image mode for requirement_family=all."""
    return ALL_IMAGE_MODE_BY_CONTEXT.get(ctx, NAV_ONLY_VQA)


def normalize_requirement_family(subject_type: str) -> str:
    """Collapse extractor subject types into policy requirement families."""
    if subject_type == "none":
        return REQ_NONE
    if subject_type == "all":
        return REQ_ALL
    return REQ_SUBJECT


def resolve_element_policy(
    effective_ctx: ScreenContext,
    requirement_family: str,
    modality: str,
    subject_type: str,
) -> str:
    """Resolve the final policy for one element."""
    level = get_level(effective_ctx)

    # PUBLIC screens never need task-driven filtering.
    if level == PUBLIC:
        return POLICY_EXPOSE

    # Refined-screen exceptions first.
    if effective_ctx == ScreenContext.messaging_conversation and subject_type == "person":
        return POLICY_EXPOSE
    if effective_ctx == ScreenContext.messaging_list and subject_type == "person":
        return POLICY_SUBJECT_MATCH

    if requirement_family == REQ_NONE:
        if modality == MODALITY_TEXT and level == PRIVACY:
            return POLICY_EXPOSE
        return POLICY_KEEP_MASKED

    if requirement_family == REQ_SUBJECT:
        if modality == MODALITY_TEXT:
            if level == PRIVACY:
                return POLICY_EXPOSE
            return POLICY_SUBJECT_MATCH
        return POLICY_SUBJECT_MATCH

    # requirement_family == REQ_ALL
    if modality == MODALITY_TEXT:
        return POLICY_EXPOSE

    all_image_mode = get_all_image_mode(effective_ctx)
    if all_image_mode == SHOW_ALL_IMAGES:
        return POLICY_EXPOSE
    return POLICY_NAV_ONLY_VQA

