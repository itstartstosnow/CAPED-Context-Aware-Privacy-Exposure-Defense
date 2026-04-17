"""Small adb subprocess helpers shared inside privacy code."""

from __future__ import annotations

import subprocess
from typing import Optional


def run_adb_text(
    adb_path: str,
    device_serial: Optional[str],
    *args: str,
    timeout: float,
) -> str:
    """Run an adb command and return decoded stdout text."""
    cmd = [adb_path]
    if device_serial:
        cmd += ["-s", device_serial]
    cmd += list(args)
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return res.stdout
