"""Privacy data seeders for AndroidWorld privacy-eval tasks."""

from collections.abc import Callable
from typing import Any

from privacy_seeders.files_seeder import seed_files_privacy_data
from privacy_seeders.gallery_seeder import seed_gallery_privacy_data
from privacy_seeders.sms_seeder import seed_sms_privacy_data

Seeder = Callable[..., None]

SEEDER_REGISTRY: dict[str, Seeder] = {
    "a1_files_move": seed_files_privacy_data,
    "a2_files_delete": seed_files_privacy_data,
    "c1_gallery_receipt_copy": seed_gallery_privacy_data,
    "g3_gallery_share_target_photo": seed_gallery_privacy_data,
    "g4_gallery_open_target_screenshot": seed_gallery_privacy_data,
    "s1_sms_reply_target_sender": seed_sms_privacy_data,
    "s2_sms_forward_received_address": seed_sms_privacy_data,
    "s3_sms_search_target_thread": seed_sms_privacy_data,
    "s4_sms_compose_to_target_contact": seed_sms_privacy_data,
    "g2_gallery_selective_task": seed_gallery_privacy_data,
}


def seed_for_task(
    task_id: str,
    controller: Any,
    task: Any | None = None,
    env: Any | None = None,
) -> None:
    """Run the registered post-initialization seeder for ``task_id`` if any."""
    seeder = SEEDER_REGISTRY.get(task_id)
    if seeder is None:
        return
    seeder(controller, task=task, env=env)
