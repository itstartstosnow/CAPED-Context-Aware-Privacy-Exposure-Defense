"""PrivacyAwareGUIOwl — GUIOwl agent with three privacy hook points.

Inherits from GUIOwl without modifying the original source.  Three hooks are
injected transparently:

  Hook 1 (task start) — NER de-identification of the task goal.
    Fires once per task on the first call to step().  Subsequent steps reuse
    the anonymized goal returned by on_task_start().

  Hook 2 (screenshot) — Privacy processing of each screenshot.
    Handled inside PrivacyEnvWrapper.get_state(), which is called by the
    inherited GUIOwl.step() via self.get_post_transition_state().  No
    changes to GUIOwl code are required.

  Hook 3 (action) — Interception of each action before execution.
    Implemented via a temporary monkey-patch of
    android_world.env.actuation.execute_adb_action within each step() call.
    The patch is installed before super().step() and restored in a finally
    block to prevent leakage.

Usage example::

    from android_world.agents.privacy_guiowl import PrivacyAwareGUIOwl
    from android_world.agents.privacy.server_hooks import ServerPrivacyHooks

    hooks = ServerPrivacyHooks(base_url="http://localhost:8001", llm_model="0.8b")
    agent = PrivacyAwareGUIOwl(
        env=env,
        vllm=vllm,
        src_format=src_format,
        api_key=api_key,
        url=model_url,
        hooks=hooks,
        name="PrivacyGUIOwl",
        output_path=output_path,
    )
"""

import json
import logging
import time

import android_world.env.actuation as _actuation_module
from android_world.agents.gui_owl import GUIOwl
from android_world.agents.privacy.env_wrapper import PrivacyEnvWrapper
from android_world.agents.privacy.hooks import PassthroughHooks, PrivacyHooks
from android_world.env import interface

logger = logging.getLogger(__name__)


class PrivacyAwareGUIOwl(GUIOwl):
    """GUIOwl extended with pluggable privacy hooks.

    Args:
        env: The AsyncEnv instance (will be wrapped with PrivacyEnvWrapper).
        vllm: VLM inference backend passed through to GUIOwl.
        src_format: Source coordinate format passed through to GUIOwl.
        api_key: API key passed through to GUIOwl.
        url: Model server URL passed through to GUIOwl.
        hooks: PrivacyHooks implementation.  Defaults to PassthroughHooks
            (no-op) so the agent behaves identically to GUIOwl when no hooks
            are configured.
        name: Agent name string.
        output_path: Directory for trajectory output, passed through to GUIOwl.
    """

    def __init__(
        self,
        env: interface.AsyncEnv,
        vllm,
        src_format,
        api_key: str,
        url: str,
        hooks: PrivacyHooks | None = None,
        name: str = "PrivacyGUIOwl",
        output_path: str = "",
    ):
        self._hooks = hooks if hooks is not None else PassthroughHooks()
        privacy_env = PrivacyEnvWrapper(env, self._hooks)
        super().__init__(
            env=privacy_env,
            vllm=vllm,
            src_format=src_format,
            api_key=api_key,
            url=url,
            name=name,
            output_path=output_path,
        )
        self._task_started = False
        self._output_path = output_path
        self._last_goal: str = ""  # track goal changes to detect new tasks
        self._anonymized_goal: str = ""

    # ------------------------------------------------------------------
    # reset: clear per-task state and notify hooks
    # ------------------------------------------------------------------

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self._task_started = False
        self._last_goal = ""  # ensure goal-change detection works across task instances
        self._anonymized_goal = ""
        self._hooks.on_task_end()

    # ------------------------------------------------------------------
    # step: inject Hook 1 and Hook 3; Hook 2 is transparent via env wrapper
    # ------------------------------------------------------------------

    def _task_output_dir(self, goal: str) -> str:
        """Return per-task output directory (mirrors GUIOwl's logic)."""
        if not self._output_path:
            return ""
        if goal in self.task_name:
            task_name = self.task_name[goal]
        else:
            task_name = goal.replace(" ", "_")[:50]
        import os as _os
        return _os.path.join(self._output_path, task_name)

    def step(self, goal: str):
        # Detect task change: update per-task output_dir and task_name in hooks.
        if goal != self._last_goal:
            self._last_goal = goal
            task_dir = self._task_output_dir(goal)
            if task_dir:
                self._hooks.set_output_dir(task_dir)
            task_label = self.task_name.get(goal, goal.replace(" ", "_")[:50])
            self._hooks.set_task_name(task_label)

        # Hook 1: NER de-identification of goal (once per task)
        if not self._task_started:
            self._anonymized_goal = self._hooks.on_task_start(goal)
            if goal in self.task_name:
                self.task_name[self._anonymized_goal] = self.task_name[goal]
            self._task_started = True
        goal = self._anonymized_goal or goal

        # Hook 3: monkey-patch actuation to intercept each action
        _orig_execute = _actuation_module.execute_adb_action
        _hooks_ref = self._hooks

        def _hooked_execute(action, screen_elements, screen_size, env):
            action_dict = {
                k: v
                for k, v in vars(action).items()
                if v is not None
            }
            # Time on_action (privacy overhead)
            _t0_ac = time.perf_counter()
            _, skip = _hooks_ref.on_action(
                action_dict,
                screen_elements=screen_elements,
                screen_size=screen_size,
            )
            _hooks_ref._last_action_check_time_s = time.perf_counter() - _t0_ac
            if skip:
                # Disclosure intercepted: revealed screenshot will be injected
                # as next observation via hooks.pending_override_pixels.
                _hooks_ref._last_action_exec_and_settle_time_s = 0.0
                return None
            # Time original actuation cost, including built-in settle waits in
            # actuation.execute_adb_action() after the adb command is issued.
            _t0_exec = time.perf_counter()
            result = _orig_execute(action, screen_elements, screen_size, env)
            _hooks_ref._last_action_exec_and_settle_time_s = time.perf_counter() - _t0_exec
            return result

        _actuation_module.execute_adb_action = _hooked_execute
        try:
            # Hook 2 fires transparently inside super().step() via PrivacyEnvWrapper
            _t0_step = time.perf_counter()
            result = super().step(goal)
            _t_step_total = time.perf_counter() - _t0_step
        finally:
            # Always restore original function to prevent leakage
            _actuation_module.execute_adb_action = _orig_execute

        # Emit per-step [TIMING] log line
        st = self._hooks._last_screenshot_timing or {}
        _task_obj = getattr(self._hooks, "_task", None)
        task_name = _task_obj.task_name if _task_obj is not None else ""
        step_idx  = (_task_obj.step_counter - 1) if _task_obj is not None else -1
        logger.info("[TIMING] %s", json.dumps({
            "type": "step",
            "task": task_name,
            "step": step_idx,
            "original": {
                # This is the full observation fetch path, not just a raw screenshot.
                "t_observation_fetch_s":  round(self._hooks._last_observation_fetch_time_s, 4),
                "t_vlm_inference_s":      round(self._last_step_vlm_time_s, 4),
                # This includes adb actuation plus built-in post-action settle waits.
                "t_action_exec_and_settle_s": round(self._hooks._last_action_exec_and_settle_time_s, 4),
            },
            "privacy": {
                # Foreground-activity query used to determine app/package context.
                "t_foreground_activity_s": round(self._hooks._last_fgactivity_time_s, 4),
                # Detection stage: OCR + UIED-based privacy detections.
                "t_detect_s":              round(st.get("t_detect_s", 0.0), 4),
                # Local policy resolution and image prep, excluding VQA/embed sub-time.
                "t_local_policy_and_prep_s": round(st.get("t_local_policy_and_prep_s", 0.0), 4),
                # Image VQA time spent by privacy policy.
                "t_vqa_s":                 round(st.get("t_vqa_s", 0.0), 4),
                # Text embedding/relevance time spent by privacy policy.
                "t_embed_s":               round(st.get("t_embed_s", 0.0), 4),
                # Time to render the final redacted screenshot.
                "t_redact_s":              round(st.get("t_redact_s", 0.0), 4),
                # Time to inspect / rewrite / block the action before execution.
                "t_action_check_s":        round(self._hooks._last_action_check_time_s, 4),
            },
            "t_step_total_s": round(_t_step_total, 4),
        }))

        return result
