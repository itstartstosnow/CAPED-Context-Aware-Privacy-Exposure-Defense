"""subpage_rules.py — Table B1 only: activity-based context refinement.

This module only refines base app context into effective screen context.
It does not encode any exposure or image policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Optional

from android_world.agents.privacy.screen_context import ScreenContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationRule:
    """One row in activity-based context refinement rules."""
    app: Optional[str]
    base_ctx: Optional[ScreenContext]
    pattern: str
    result_ctx: ScreenContext


CLASSIFICATION_RULES: list[ClassificationRule] = [
    ClassificationRule("com.tencent.mm", None, "*LauncherUI*",         ScreenContext.messaging_list),
    ClassificationRule("com.tencent.mm", None, "*ConvUI*",             ScreenContext.messaging_conversation),
    ClassificationRule("com.tencent.mm", None, "*ChatUI*",             ScreenContext.messaging_conversation),
    ClassificationRule("com.tencent.mm", None, "*SnsTimeLineUI*",      ScreenContext.content_feed),
    ClassificationRule("com.tencent.mm", None, "*WXPayEntryActivity*", ScreenContext.financial),
    ClassificationRule("com.tencent.mm", None, "*WalletUI*",           ScreenContext.financial),
    ClassificationRule("com.tencent.mm", None, "*GalleryActivity*",    ScreenContext.gallery),
    ClassificationRule("com.tencent.mm", None, "*UserInfo*",           ScreenContext.personal_profile),
    ClassificationRule(None, ScreenContext.messaging, "*ThreadList*",       ScreenContext.messaging_list),
    ClassificationRule(None, ScreenContext.messaging, "*ConversationList*", ScreenContext.messaging_list),
    ClassificationRule(None, ScreenContext.messaging, "*MessageList*",      ScreenContext.messaging_list),
    ClassificationRule(None, ScreenContext.messaging, "*ContactList*",      ScreenContext.messaging_list),
    ClassificationRule(None, ScreenContext.messaging, "*Inbox*",            ScreenContext.messaging_list),
    ClassificationRule(None, ScreenContext.messaging, "*Main*",             ScreenContext.messaging_list),
    ClassificationRule(None, ScreenContext.messaging, "*Conversation*",     ScreenContext.messaging_conversation),
    ClassificationRule(None, ScreenContext.messaging, "*ChatWindow*",       ScreenContext.messaging_conversation),
    ClassificationRule(None, ScreenContext.messaging, "*ChatDetail*",       ScreenContext.messaging_conversation),
    ClassificationRule(None, ScreenContext.messaging, "*MessageDetail*",    ScreenContext.messaging_conversation),
    ClassificationRule(None, ScreenContext.messaging, "*SingleConv*",       ScreenContext.messaging_conversation),
    ClassificationRule(None, ScreenContext.messaging, "*TalkActivity*",     ScreenContext.messaging_conversation),
    ClassificationRule(None, ScreenContext.messaging, "*Chat*",             ScreenContext.messaging_conversation),
    ClassificationRule(None, ScreenContext.messaging, "*Thread*",           ScreenContext.messaging_conversation),
    ClassificationRule(None, ScreenContext.messaging, "*Message*",          ScreenContext.messaging_conversation),
    ClassificationRule(None, ScreenContext.messaging, "*Gallery*",          ScreenContext.gallery),
    ClassificationRule(None, ScreenContext.messaging, "*Photo*",            ScreenContext.gallery),
    ClassificationRule(None, ScreenContext.messaging, "*Album*",            ScreenContext.gallery),
    ClassificationRule(None, ScreenContext.messaging, "*Profile*",          ScreenContext.personal_profile),
    ClassificationRule(None, ScreenContext.messaging, "*UserInfo*",         ScreenContext.personal_profile),
    ClassificationRule(None, ScreenContext.messaging, "*Payment*",          ScreenContext.financial),
    ClassificationRule(None, ScreenContext.messaging, "*Wallet*",           ScreenContext.financial),
    ClassificationRule(None, None, "*Setting*",    ScreenContext.settings),
    ClassificationRule(None, None, "*Preference*", ScreenContext.settings),
    ClassificationRule(None, None, "*Prefs*",      ScreenContext.settings),
]


def classify_subpage(
    package_name: str,
    base_ctx: ScreenContext,
    activity_class: str,
) -> ScreenContext | None:
    """Return refined effective context for this activity when a rule matches."""
    if not activity_class:
        return None

    for rule in CLASSIFICATION_RULES:
        if rule.app is not None:
            if package_name != rule.app and not package_name.startswith(rule.app + "."):
                continue
        if rule.base_ctx is not None and base_ctx != rule.base_ctx:
            continue
        if fnmatch(activity_class, rule.pattern):
            logger.info(
                "[B1] pkg=%s base=%s act=%r → pattern=%r → %s",
                package_name, base_ctx.value, activity_class,
                rule.pattern, rule.result_ctx.value,
            )
            return rule.result_ctx

    return None
