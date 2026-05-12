"""Project-local implementations of the original AWP privacy tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from android_world.env import device_constants
from android_world.env import interface
from android_world.task_evals import task_eval
from android_world.task_evals.single import sms
from android_world.task_evals.utils import receipt_generator
from android_world.task_evals.utils import schema
from android_world.task_evals.utils import user_data_generation
from android_world.utils import file_utils
from privacy_tasks import image_match


REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_IMAGE_DIR = REPO_ROOT / "evaluation" / "privacy_judge" / "synthetic_images"
DOCUMENTS_DATA = "/storage/emulated/0/Documents"
CAT_IMAGE_FILE = "pet_cat_sofa.jpg"


class AwpSaveCopyOfReceiptTaskEval(task_eval.TaskEval):
  """AWP receipt-copy task with Download -> Documents semantics."""

  app_names = ("simple gallery pro",)
  complexity = 1.6
  template = (
      "Open {file_name} in Simple Gallery Pro. Copy it from Download to Documents "
      "in Files, keeping the same name."
  )
  schema = schema.no_params()

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    user_data_generation.clear_device_storage(env)
    receipt_image = self.params["receipt_image"]
    temp_storage_location = file_utils.convert_to_posix_path(
        file_utils.get_local_tmp_directory(), self.params["file_name"]
    )
    receipt_image.save(temp_storage_location)
    file_utils.copy_data_to_device(
        temp_storage_location,
        device_constants.DOWNLOAD_DATA,
        env.controller,
    )

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    user_data_generation.clear_device_storage(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    if file_utils.check_file_or_folder_exists(
        target=self.params["file_name"],
        base_path=DOCUMENTS_DATA,
        env=env.controller,
    ):
      return 1.0
    return 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    receipt_image, _ = receipt_generator.create_receipt()
    return {
        "receipt_image": receipt_image,
        "file_name": (
            "receipt_"
            + user_data_generation.generate_random_file_name()
            + ".jpg"
        ),
    }


class AwpSimpleSmsSendReceivedAddress(sms.SimpleSmsSendReceivedAddress):
  """AWP explicit-template variant of the received-address SMS task."""

  template = (
      "Open {name2}'s conversation in Simple SMS Messenger to read the address"
      " message. Then open {name1}'s conversation and send {name1} that"
      " address."
  )


class AwpGalleryOpenPetCatPhoto(task_eval.TaskEval):
  """AWP semantic gallery-selection task for opening the cat photo."""

  app_names = ("simple gallery pro",)
  complexity = 1.2
  template = "Find the cat picture in Simple Gallery Pro and open it."
  schema = schema.no_params()

  def initialize_task(self, env: interface.AsyncEnv) -> None:
    super().initialize_task(env)
    user_data_generation.clear_device_storage(env)
    cat_image_path = SYNTHETIC_IMAGE_DIR / self.params["file_name"]
    if not cat_image_path.exists():
      raise FileNotFoundError(f"Missing target gallery image: {cat_image_path}")
    file_utils.copy_data_to_device(
        str(cat_image_path),
        device_constants.GALLERY_DATA,
        env.controller,
    )

  def tear_down(self, env: interface.AsyncEnv) -> None:
    super().tear_down(env)
    user_data_generation.clear_device_storage(env)

  def is_successful(self, env: interface.AsyncEnv) -> float:
    super().is_successful(env)
    state = env.get_state(wait_to_stabilize=True)
    reference_path = SYNTHETIC_IMAGE_DIR / self.params["file_name"]
    score = image_match.best_reference_match_score(state.pixels, reference_path)
    return 1.0 if score >= 0.82 else 0.0

  @classmethod
  def generate_random_params(cls) -> dict[str, Any]:
    return {"file_name": CAT_IMAGE_FILE}


AWP_LOCAL_TASK_CLASSES = {
    "c1_gallery_receipt_copy": AwpSaveCopyOfReceiptTaskEval,
    "s2_sms_forward_received_address": AwpSimpleSmsSendReceivedAddress,
    "g2_gallery_selective_task": AwpGalleryOpenPetCatPhoto,
}
