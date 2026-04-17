"""Abstract privacy hook interfaces for PrivacyAwareGUIOwl."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class PrivacyHooks(ABC):
    """Abstract interface defining three hook points for privacy processing.

    Subclasses may also set ``pending_override_pixels`` (np.ndarray or None)
    to inject a synthetic next-observation screenshot — used by the Interactive
    Disclosure mechanism when an eye-icon region is tapped.
    """

    # Injected by disclosure logic: if set, PrivacyEnvWrapper returns this
    # array as the next observation instead of capturing a new phone screenshot.
    pending_override_pixels: Optional[np.ndarray] = None

    # Per-step timing state written by PrivacyEnvWrapper and read by PrivacyAwareGUIOwl.
    # Defaults to 0.0 so PassthroughHooks works without AttributeError.
    # Observation fetch includes the env.get_state() chain, i.e. screenshot
    # retrieval plus accessibility/UI-tree collection and element extraction.
    _last_observation_fetch_time_s: float = 0.0
    _last_fgactivity_time_s: float = 0.0
    _last_screenshot_timing: dict = {}
    _last_action_check_time_s: float = 0.0
    # Action timing includes the adb actuation itself plus any built-in settle
    # waits inside actuation.execute_adb_action(), including the common
    # post-action sleep used before the next observation is fetched.
    _last_action_exec_and_settle_time_s: float = 0.0

    @abstractmethod
    def on_task_start(self, goal: str) -> str:
        """Called once per task before the first step.

        Performs NER de-identification of the task goal and stores the mapping
        for subsequent calls within the same task.

        Args:
            goal: The original task goal string.

        Returns:
            Anonymized goal string.
        """

    @abstractmethod
    def on_screenshot(self, pixels: np.ndarray, activity_name: str = "") -> np.ndarray:
        """Called on each screenshot before it is sent to the model.

        Args:
            pixels: RGB numpy array of the original screenshot.
            activity_name: Full Android activity name of the foreground app
                (e.g. "com.google.android.apps.photos/.PhotosActivity").
                Used for screen context classification.  Defaults to "" if
                the activity could not be determined.

        Returns:
            Privacy-processed RGB numpy array (same shape as input).
        """

    @abstractmethod
    def on_action(self, action: dict,
                  screen_elements=None,
                  screen_size=None) -> Tuple[dict, bool]:
        """Called after the agent produces an action, before execution.

        Performs two duties:
          1. De-anonymize text fields (NER reverse mapping).
          2. Detect disclosure: if the action targets a protected (eye-icon)
             region, set ``pending_override_pixels`` with the revealed screenshot
             and return ``skip=True`` so the action is NOT sent to the phone.

        Args:
            action: Dict representation of the agent action.
            screen_elements: List of UI elements from the env (may be None).
            screen_size: (width, height) tuple in pixels (may be None).

        Returns:
            Tuple (modified_action, skip_execution).  If skip_execution is True
            the caller must NOT execute the action on the real device.
        """

    def on_task_end(self) -> None:
        """Called when a task ends (agent reset). Override to add cleanup."""
        pass

    def set_output_dir(self, path: str) -> None:
        """Optionally set the per-task output directory for disclosure logging."""
        pass


class PassthroughHooks(PrivacyHooks):
    """No-op hooks — passes all data through unchanged.

    Use for control-group evaluation or unit testing to verify that
    PrivacyAwareGUIOwl behaves identically to GUIOwl when hooks are inactive.
    """

    def on_task_start(self, goal: str) -> str:
        return goal

    def on_screenshot(self, pixels: np.ndarray, activity_name: str = "") -> np.ndarray:
        return pixels

    def on_action(self, action: dict,
                  screen_elements=None,
                  screen_size=None) -> Tuple[dict, bool]:
        return action, False
