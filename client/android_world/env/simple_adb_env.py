"""Lightweight ADB-only environment for real (non-emulator) Android devices.


Implements AsyncEnv using subprocess adb calls instead of the gRPC-based
android_env framework. No emulator console port or gRPC accessibility server
is required — any device visible to `adb devices` works.

Usage:
    env = SimpleAdbEnv(adb_path="~/Library/Android/sdk/platform-tools/adb",
                       device_serial="R3CT90XXXXX")  # from `adb devices`
    # or omit device_serial if only one device is connected

Limitations vs AsyncAndroidEnv:
    - No accessibility tree (ui_elements is always empty; agent uses coordinates)
    - No a11y-based element lookup; all actions are coordinate-based (fine for
      GUI-Owl which uses abs_resized coordinates)
    - `ask_question` raises NotImplementedError
"""

import io
import subprocess
import time
from typing import Any, Optional

import numpy as np
from PIL import Image

from android_world.env import interface

try:
    from android_env.proto import adb_pb2
    _HAS_ADB_PB2 = True
except ImportError:
    _HAS_ADB_PB2 = False


class SimpleAdbController:
    """Minimal controller that implements execute_adb_call via subprocess adb.

    Only the AdbRequest types actually used by adb_utils / actuation are
    implemented.  Unrecognised request types return OK silently.

    Args:
        adb_path: Path to the adb binary.
        device_serial: Device serial from `adb devices`.  If None, adb's
            default device selection is used (works when exactly one device
            is connected).
        default_timeout: Subprocess timeout in seconds for adb calls.
    """

    def __init__(
        self,
        adb_path: str,
        device_serial: Optional[str] = None,
        default_timeout: float = 30.0,
    ):
        self._adb = adb_path
        self._serial = device_serial
        self._timeout = default_timeout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adb_cmd(self, *args: str) -> list[str]:
        cmd = [self._adb]
        if self._serial:
            cmd += ["-s", self._serial]
        cmd += list(args)
        return cmd

    def _run(self, *args: str, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            self._adb_cmd(*args),
            capture_output=True,
            timeout=timeout or self._timeout,
        )

    def _ok(self) -> "adb_pb2.AdbResponse":
        resp = adb_pb2.AdbResponse()
        resp.status = adb_pb2.AdbResponse.Status.OK
        return resp

    def _fail(self) -> "adb_pb2.AdbResponse":
        resp = adb_pb2.AdbResponse()
        resp.status = adb_pb2.AdbResponse.Status.ADB_ERROR
        return resp

    # ------------------------------------------------------------------
    # Public interface expected by adb_utils / actuation
    # ------------------------------------------------------------------

    def execute_adb_call(self, request: "adb_pb2.AdbRequest") -> "adb_pb2.AdbResponse":
        """Translate an AdbRequest protobuf to a subprocess adb call."""
        which = request.WhichOneof("command")

        if which == "tap":
            r = self._run("shell", "input", "tap",
                          str(int(request.tap.x)), str(int(request.tap.y)))
            return self._ok() if r.returncode == 0 else self._fail()

        elif which == "press_button":
            _keymap = {
                adb_pb2.AdbRequest.PressButton.HOME:  "KEYCODE_HOME",
                adb_pb2.AdbRequest.PressButton.BACK:  "KEYCODE_BACK",
                adb_pb2.AdbRequest.PressButton.ENTER: "KEYCODE_ENTER",
            }
            keycode = _keymap.get(request.press_button.button, "KEYCODE_HOME")
            r = self._run("shell", "input", "keyevent", keycode)
            return self._ok() if r.returncode == 0 else self._fail()

        elif which == "input_text":
            text = request.input_text.text
            r = self._run("shell", "input", "text", text)
            return self._ok() if r.returncode == 0 else self._fail()

        elif which == "generic":
            args = list(request.generic.args)
            timeout = request.timeout_sec if request.timeout_sec > 0 else self._timeout
            r = self._run(*args, timeout=timeout)
            resp = adb_pb2.AdbResponse()
            resp.status = (adb_pb2.AdbResponse.Status.OK
                           if r.returncode == 0
                           else adb_pb2.AdbResponse.Status.ADB_ERROR)
            resp.generic.output = r.stdout + r.stderr
            return resp

        elif which == "send_broadcast":
            action = request.send_broadcast.action
            component = request.send_broadcast.component
            cmd_args = ["shell", "am", "broadcast", "-a", action]
            if component:
                cmd_args += ["-n", component]
            r = self._run(*cmd_args)
            return self._ok() if r.returncode == 0 else self._fail()

        elif which == "start_activity":
            act = request.start_activity
            cmd_args = ["shell", "am", "start"]
            if act.app_name:
                cmd_args += ["-n", act.app_name]
            if act.intent_data:
                cmd_args += ["-d", act.intent_data]
            r = self._run(*cmd_args)
            return self._ok() if r.returncode == 0 else self._fail()

        elif which == "force_stop":
            r = self._run("shell", "am", "force-stop", request.force_stop.package_name)
            return self._ok() if r.returncode == 0 else self._fail()

        else:
            # Unknown / unimplemented — return OK to avoid crashing the agent
            return self._ok()

    # ------------------------------------------------------------------
    # Screen size helpers (used by SimpleAdbEnv and actuation indirectly)
    # ------------------------------------------------------------------

    def get_screen_size(self) -> tuple[int, int]:
        """Returns (width, height) via `adb shell wm size`."""
        r = self._run("shell", "wm", "size")
        output = r.stdout.decode("utf-8", errors="replace")
        for line in output.strip().splitlines():
            if "size:" in line.lower():
                parts = line.split()[-1].split("x")
                if len(parts) == 2:
                    return (int(parts[0]), int(parts[1]))
        return (1080, 1920)  # safe fallback


class SimpleAdbEnv(interface.AsyncEnv):
    """AsyncEnv backed by pure adb subprocess calls.

    Suitable for any device visible to `adb devices` without requiring the
    gRPC accessibility service or emulator console port used by AsyncAndroidEnv.

    Args:
        adb_path: Path to the adb binary.
        device_serial: Device serial from `adb devices`.
        screenshot_timeout: adb screencap timeout in seconds.
    """

    interaction_cache: str = ""

    def __init__(
        self,
        adb_path: str,
        device_serial: Optional[str] = None,
        screenshot_timeout: float = 15.0,
    ):
        self._ctrl = SimpleAdbController(adb_path, device_serial)
        self._screenshot_timeout = screenshot_timeout
        self.interaction_cache = ""
        self._cached_screen_size: Optional[tuple[int, int]] = None

    # ------------------------------------------------------------------
    # AsyncEnv interface
    # ------------------------------------------------------------------

    @property
    def controller(self) -> SimpleAdbController:
        return self._ctrl

    def reset(self, go_home: bool = False) -> interface.State:
        self.interaction_cache = ""
        if go_home:
            self._ctrl._run("shell", "input", "keyevent", "KEYCODE_HOME")
            time.sleep(1.0)
        return self.get_state()

    def get_state(self, wait_to_stabilize: bool = False) -> interface.State:
        if wait_to_stabilize:
            time.sleep(1.5)
        pixels = self._take_screenshot()
        return interface.State(pixels=pixels, forest=None, ui_elements=[])

    def execute_action(self, action: Any) -> None:
        pass  # Not called by GUIOwl directly; actuation uses controller

    def ask_question(self, question: str, timeout_seconds: float = -1.0) -> Optional[str]:
        raise NotImplementedError("ask_question not supported by SimpleAdbEnv")

    @property
    def foreground_activity_name(self) -> str:
        r = self._ctrl._run(
            "shell", "dumpsys", "activity", "activities"
        )
        output = r.stdout.decode("utf-8", errors="replace")
        for line in output.splitlines():
            if "mCurrentFocus" in line or "mFocusedActivity" in line:
                parts = line.strip().split()
                if parts:
                    return parts[-1]
        return ""

    @property
    def device_screen_size(self) -> tuple[int, int]:
        return self.logical_screen_size

    @property
    def logical_screen_size(self) -> tuple[int, int]:
        if self._cached_screen_size is None:
            self._cached_screen_size = self._ctrl.get_screen_size()
        return self._cached_screen_size

    def close(self) -> None:
        pass

    def hide_automation_ui(self) -> None:
        """No-op — no AndroidWorld overlay on real devices."""
        pass

    @property
    def orientation(self) -> int:
        r = self._ctrl._run("shell", "settings", "get", "system", "user_rotation")
        try:
            return int(r.stdout.strip())
        except (ValueError, AttributeError):
            return 0

    @property
    def physical_frame_boundary(self) -> tuple[int, int, int, int]:
        w, h = self.logical_screen_size
        return (0, 0, w, h)

    def display_message(self, message: str, header: str = "") -> None:
        pass  # No overlay on real devices

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _take_screenshot(self) -> np.ndarray:
        """Capture screenshot via `adb exec-out screencap -p`."""
        r = subprocess.run(
            self._ctrl._adb_cmd("exec-out", "screencap", "-p"),
            capture_output=True,
            timeout=self._screenshot_timeout,
        )
        img = Image.open(io.BytesIO(r.stdout)).convert("RGB")
        # Invalidate cached size so it matches the actual screenshot
        self._cached_screen_size = img.size  # (width, height)
        return np.array(img, dtype=np.uint8)
