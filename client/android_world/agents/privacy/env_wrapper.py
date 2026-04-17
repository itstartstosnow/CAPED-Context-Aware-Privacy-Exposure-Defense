"""Environment wrapper that intercepts screenshots for privacy processing."""

import dataclasses
import logging
import re as _re
import time

from android_world.env import interface
from android_world.agents.privacy.adb_helper import run_adb_text
from android_world.agents.privacy.hooks import PrivacyHooks

logger = logging.getLogger(__name__)


def _get_foreground_activity(env) -> str:
    """Get foreground activity via 'adb shell dumpsys activity activities'.

    Reads adb_path and device_serial from env attributes.  SimpleAdbEnv stores
    them inside _ctrl._adb / _ctrl._serial (private controller), so we check
    there as a fallback before giving up and using bare 'adb'.
    Does NOT touch android_world's gRPC interface.
    """
    # Try public attributes first (future-proof / other env types)
    adb = getattr(env, 'adb_path', None) or 'adb'
    serial = getattr(env, 'device_serial', None) or getattr(env, 'serial', None)

    # SimpleAdbEnv stores these inside self._ctrl
    ctrl = getattr(env, '_ctrl', None)
    if ctrl:
        adb = getattr(ctrl, '_adb', None) or adb
        serial = serial or getattr(ctrl, '_serial', None)

    try:
        out = run_adb_text(
            adb,
            serial,
            "shell",
            "dumpsys activity activities | grep -m1 topResumedActivity",
            timeout=4,
        )
        m = _re.search(
            r'topResumedActivity=ActivityRecord\{[^}]*\s+([\w.]+/[\w.$]+)',
            out,
        )
        if m:
            return m.group(1)
        logger.debug("_get_foreground_activity: topResumedActivity not found in dump")
    except Exception as exc:  # pylint: disable=broad-except
        logger.info("_get_foreground_activity failed: %s", exc)
    return ""


class PrivacyEnvWrapper:
    """Wraps an AsyncEnv to intercept screenshots via an on_screenshot hook.

    All attribute access is transparently delegated to the underlying env
    except for ``get_state()``, which replaces the pixels field with the
    privacy-processed version returned by ``hooks.on_screenshot()``.

    The ``interaction_cache`` attribute is proxied explicitly so that
    assignment by the agent (``self.env.interaction_cache = value``) is
    forwarded to the real env and remains visible to the episode runner.

    Args:
        env: The real AsyncEnv instance to wrap.
        hooks: PrivacyHooks instance providing the screenshot hook.
    """

    def __init__(self, env: interface.AsyncEnv, hooks: PrivacyHooks):
        # Use object.__setattr__ to avoid triggering our own __setattr__
        object.__setattr__(self, '_env', env)
        object.__setattr__(self, '_hooks', hooks)

    # ------------------------------------------------------------------
    # Core override: screenshot interception
    # ------------------------------------------------------------------

    def get_state(self, wait_to_stabilize: bool = False) -> interface.State:
        """Get state and apply the screenshot privacy hook to pixels.

        If a disclosure was triggered on the previous action, the hooks will
        have set ``pending_override_pixels`` to the revealed screenshot.  In
        that case we return it directly (without calling on_screenshot again)
        so the agent sees the disclosed content as the next observation.
        """
        override = self._hooks.pending_override_pixels
        if override is not None:
            self._hooks.pending_override_pixels = None  # consume once
            _t0 = time.perf_counter()
            state = self._env.get_state(wait_to_stabilize=wait_to_stabilize)
            self._hooks._last_observation_fetch_time_s = time.perf_counter() - _t0
            self._hooks._last_fgactivity_time_s = 0.0
            return dataclasses.replace(state, pixels=override)

        # Time original observation fetch (not just raw screenshot capture).
        # This includes env.get_state(), which in the main environment also
        # gathers accessibility/UI-tree data and builds ui_elements.
        _t0 = time.perf_counter()
        state = self._env.get_state(wait_to_stabilize=wait_to_stabilize)
        self._hooks._last_observation_fetch_time_s = time.perf_counter() - _t0

        # Time foreground activity query (privacy system cost)
        _t1 = time.perf_counter()
        activity_name = _get_foreground_activity(self._env)
        self._hooks._last_fgactivity_time_s = time.perf_counter() - _t1
        if not activity_name:
            logger.info("foreground activity: (none)")

        protected_pixels = self._hooks.on_screenshot(state.pixels, activity_name=activity_name)
        return dataclasses.replace(state, pixels=protected_pixels)

    # ------------------------------------------------------------------
    # interaction_cache proxy (get + set forwarded to real env)
    # ------------------------------------------------------------------

    @property
    def interaction_cache(self):
        return self._env.interaction_cache

    @interaction_cache.setter
    def interaction_cache(self, value):
        self._env.interaction_cache = value

    # ------------------------------------------------------------------
    # Transparent delegation for everything else
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def __setattr__(self, name: str, value):
        # Private attributes stay on the wrapper; everything else goes to env.
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._env, name, value)
