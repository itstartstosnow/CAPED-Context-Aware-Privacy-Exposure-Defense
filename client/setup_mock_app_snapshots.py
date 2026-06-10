#!/usr/bin/env python3
"""Install mock privacy apps and save AndroidWorld-style app snapshots.

This script is intentionally independent of AVD snapshots. Use it after booting
the known-good AndroidWorld AVD snapshot: it installs the three mock app APKs,
clears each app to a clean baseline, launches it once so app data is created,
and stores /data/data/<package>/ under /data/data/android_world/snapshots/.
If all packages and snapshots already exist, it exits without changing state.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
DEFAULT_ADB = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")
SNAPSHOT_ROOT = "/data/data/android_world/snapshots"
APK_CACHE_DIR = SCRIPT_DIR / "mock_app_apks_260511"


MOCK_APPS = [
    {
        "name": "sunshop",
        "package": "com.anon.sunshop",
        "source_apk": os.environ.get("MOCK_APK_SOURCE_DIR", "") and os.path.join(os.environ["MOCK_APK_SOURCE_DIR"], "sunshop-debug.apk") or "",
        "cached_apk": "sunshop-debug.apk",
    },
    {
        "name": "messengerplus",
        "package": "com.anon.messengerplus",
        "source_apk": os.environ.get("MOCK_APK_SOURCE_DIR", "") and os.path.join(os.environ["MOCK_APK_SOURCE_DIR"], "messengerplus-debug.apk") or "",
        "cached_apk": "messengerplus-debug.apk",
    },
    {
        "name": "healthtrack",
        "package": "com.anon.healthtrack",
        "source_apk": os.environ.get("MOCK_APK_SOURCE_DIR", "") and os.path.join(os.environ["MOCK_APK_SOURCE_DIR"], "healthtrack-debug.apk") or "",
        "cached_apk": "healthtrack-debug.apk",
    },
]


def _adb_cmd(adb_path: str, serial: str, *args: str) -> list[str]:
    cmd = [adb_path]
    if serial:
        cmd += ["-s", serial]
    return cmd + list(args)


def _run(
    adb_path: str,
    serial: str,
    *args: str,
    check: bool = True,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    cmd = _adb_cmd(adb_path, serial, *args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        joined = " ".join(cmd)
        raise RuntimeError(
            f"Command failed ({result.returncode}): {joined}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _auto_serial(adb_path: str) -> str:
    result = subprocess.run(
        [adb_path, "devices"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    serials = [
        line.split()[0]
        for line in result.stdout.splitlines()[1:]
        if len(line.split()) >= 2 and line.split()[1] == "device"
    ]
    if not serials:
        raise RuntimeError("No ADB device found.")
    if len(serials) > 1:
        raise RuntimeError(f"Multiple ADB devices found: {serials}. Pass --device_serial.")
    return serials[0]


def _copy_apk_to_cache(app: dict[str, str], refresh: bool = False) -> Path:
    APK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = APK_CACHE_DIR / app["cached_apk"]
    raw_source = (app.get("source_apk") or "").strip()
    if target.is_file() and (not refresh or not raw_source):
        return target
    if not raw_source:
        raise FileNotFoundError(f"Cached APK not found at {target}; place {app['cached_apk']} in {APK_CACHE_DIR}")
    source = Path(raw_source).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Source APK not found: {source}")

    needs_copy = refresh or not target.is_file()
    if target.is_file() and not refresh:
        needs_copy = source.stat().st_size != target.stat().st_size
    if needs_copy:
        shutil.copy2(source, target)
        print(f"[apk] cached {app['name']}: {target}")
    return target


def _is_installed(adb_path: str, serial: str, package: str) -> bool:
    result = _run(adb_path, serial, "shell", "pm", "path", package, check=False, timeout=10)
    return result.returncode == 0 and "package:" in result.stdout


def _snapshot_exists(adb_path: str, serial: str, package: str, snapshot_root: str) -> bool:
    snapshot = f"{snapshot_root}/{package}"
    script = f'test -d "{snapshot}" && test -n "$(ls -A "{snapshot}" 2>/dev/null)"'
    result = _run(adb_path, serial, "shell", script, check=False, timeout=10)
    return result.returncode == 0


def _all_ready(adb_path: str, serial: str, snapshot_root: str) -> bool:
    ready = True
    for app in MOCK_APPS:
        package = app["package"]
        installed = _is_installed(adb_path, serial, package)
        snap = _snapshot_exists(adb_path, serial, package, snapshot_root)
        print(f"[check] {package}: installed={installed} snapshot={snap}")
        ready = ready and installed and snap
    return ready


def _install_app(adb_path: str, serial: str, apk_path: Path, package: str) -> None:
    print(f"[install] {package} <- {apk_path}")
    result = _run(
        adb_path,
        serial,
        "install",
        "-r",
        "-t",
        "-g",
        str(apk_path),
        check=False,
        timeout=180,
    )
    if result.returncode != 0 and "INSTALL_FAILED_VERSION_DOWNGRADE" in result.stderr:
        result = _run(
            adb_path,
            serial,
            "install",
            "-r",
            "-d",
            "-t",
            "-g",
            str(apk_path),
            check=False,
            timeout=180,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to install {package} from {apk_path}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _launch_once(adb_path: str, serial: str, package: str) -> None:
    print(f"[init] launch {package}")
    _run(
        adb_path,
        serial,
        "shell",
        "monkey",
        "--pct-syskeys",
        "0",
        "-p",
        package,
        "-c",
        "android.intent.category.LAUNCHER",
        "1",
        timeout=30,
    )
    time.sleep(2.0)
    _run(adb_path, serial, "shell", "am", "force-stop", package, timeout=20)


def _save_snapshot(adb_path: str, serial: str, package: str, snapshot_root: str) -> None:
    snapshot = f"{snapshot_root}/{package}"
    app_data = f"/data/data/{package}"
    print(f"[snapshot] save {package} -> {snapshot}")
    commands = [
        f"am force-stop {package}",
        f"rm -rf {snapshot}",
        f"mkdir -p {snapshot}",
        f"cp -a {app_data}/. {snapshot}/",
        f"chmod 777 -R {snapshot}",
    ]
    for command in commands:
        _run(adb_path, serial, "shell", command, timeout=60)
    if not _snapshot_exists(adb_path, serial, package, snapshot_root):
        raise RuntimeError(f"Snapshot verification failed for {package}: {snapshot}")


def setup_mock_apps(args: argparse.Namespace) -> None:
    serial = args.device_serial or _auto_serial(args.adb_path)
    print(f"Device: {serial}")
    print(f"Snapshot root: {args.snapshot_root}")

    if not args.force and _all_ready(args.adb_path, serial, args.snapshot_root):
        print("All mock app packages and snapshots already exist. Nothing to do.")
        return

    _run(args.adb_path, serial, "shell", "getprop", "sys.boot_completed", timeout=10)
    _run(args.adb_path, serial, "shell", "mkdir", "-p", args.snapshot_root, timeout=20)

    for app in MOCK_APPS:
        package = app["package"]
        apk_path = _copy_apk_to_cache(app, refresh=args.refresh_apks)
        if args.force or not _is_installed(args.adb_path, serial, package):
            _install_app(args.adb_path, serial, apk_path, package)
        else:
            print(f"[install] {package} already installed")

        if args.force or not _snapshot_exists(args.adb_path, serial, package, args.snapshot_root):
            print(f"[clean] pm clear {package}")
            _run(args.adb_path, serial, "shell", "pm", "clear", package, timeout=30)
            _launch_once(args.adb_path, serial, package)
            _save_snapshot(args.adb_path, serial, package, args.snapshot_root)
        else:
            print(f"[snapshot] {package} already exists")

    print("Mock app setup complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install SunShop/MessengerPlus/HealthTrack and save app snapshots."
    )
    parser.add_argument("--adb_path", default=DEFAULT_ADB)
    parser.add_argument("--device_serial", default="")
    parser.add_argument("--snapshot_root", default=SNAPSHOT_ROOT)
    parser.add_argument("--force", action="store_true", help="Reinstall/recreate snapshots.")
    parser.add_argument("--refresh_apks", action="store_true", help="Refresh cached APK copies.")
    return parser.parse_args()


def main() -> None:
    try:
        setup_mock_apps(parse_args())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
