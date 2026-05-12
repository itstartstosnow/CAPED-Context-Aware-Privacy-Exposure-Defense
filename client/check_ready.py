#!/usr/bin/env python3
"""Pre-flight checks before running privacy eval scripts."""

import json
import os
import subprocess
import sys
import urllib.request
import urllib.error

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

ADB = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")
PYTHON = os.path.expanduser("~/.virtualenvs/android_world/bin/python")
SNAPSHOT_ROOT = "/data/data/android_world/snapshots"
SNAPSHOT_PKGS = ["com.anon.sunshop", "com.anon.messengerplus", "com.anon.healthtrack"]

passed = 0
failed = 0
hints = []


def ok(msg):
    global passed
    passed += 1
    print(f"  [PASS] {msg}")


def fail(msg):
    global failed
    failed += 1
    print(f"  [FAIL] {msg}")


def adb(*args, timeout=5):
    return subprocess.run(
        [ADB, *args], capture_output=True, text=True, timeout=timeout
    )


# ── Android Emulator ────────────────────────────────────────────
print("=== Android Emulator ===")

if os.access(ADB, os.X_OK):
    ok(f"ADB binary found")
else:
    fail(f"ADB binary not found at {ADB}")
    hints.append("Install Android SDK platform-tools")

boot = ""
if os.access(ADB, os.X_OK):
    try:
        r = adb("devices")
        device_lines = [
            l for l in r.stdout.strip().splitlines()[1:]
            if l.strip().endswith("\tdevice")
        ]
        n = len(device_lines)
        if n == 1:
            ok("ADB device connected")
        elif n == 0:
            fail("No ADB device found — is the emulator running?")
            hints.append(
                "Start emulator:\n"
                "  ~/Library/Android/sdk/emulator/emulator -avd AndroidWorldAvd "
                "-no-snapshot -grpc 8554 -skip-adb-auth -no-boot-anim -no-audio"
            )
        else:
            fail(f"Multiple ADB devices ({n}) — specify one with --device_serial")

        if n >= 1:
            try:
                r = adb("shell", "getprop", "sys.boot_completed")
                boot = r.stdout.strip()
                if boot == "1":
                    ok("Device boot completed")
                else:
                    fail(f"Device not fully booted (sys.boot_completed={boot or 'empty'})")
                    hints.append("Wait for emulator to finish booting")
            except subprocess.TimeoutExpired:
                fail("adb shell getprop timed out — device may be hanging")
                hints.append("Restart the emulator")
    except subprocess.TimeoutExpired:
        fail("adb devices timed out")

if boot == "1":
    missing = []
    for pkg in SNAPSHOT_PKGS:
        try:
            r = adb("shell", "ls", "-d", f"{SNAPSHOT_ROOT}/{pkg}")
            if r.returncode != 0:
                missing.append(pkg)
        except subprocess.TimeoutExpired:
            missing.append(pkg)
    if not missing:
        ok(f"All {len(SNAPSHOT_PKGS)} app snapshots present")
    else:
        fail(f"Missing snapshots: {', '.join(missing)}")
        hints.append(
            "Prepare mock app snapshots:\n"
            "  python setup_mock_app_snapshots.py"
        )

# ── Server Connectivity ─────────────────────────────────────────
print("\n=== Server Connectivity ===")

# Port 8000 — GUI-Owl vLLM
try:
    req = urllib.request.Request("http://localhost:8000/v1/models")
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read())
        model_id = data.get("data", [{}])[0].get("id", "?")
        ok(f"GUI-Owl vLLM server reachable (model: {model_id})")
except Exception:
    fail("GUI-Owl vLLM server NOT reachable on localhost:8000")
    hints.append(
        "Check SSH port forwarding for port 8000, or start vLLM on the GPU inference server"
    )

# Port 8001 — agentPrivacy
try:
    req = urllib.request.Request("http://localhost:8001/api/v1/health")
    with urllib.request.urlopen(req, timeout=5) as resp:
        if resp.status == 200:
            ok("agentPrivacy server reachable on localhost:8001")
        else:
            fail(f"agentPrivacy returned status {resp.status}")
except Exception:
    fail("agentPrivacy server NOT reachable on localhost:8001")
    hints.append(
        "On the GPU inference server:\n"
        "  cd <CAPED_SERVER_DIR>\n"
        "  source <VENV_PATH>/bin/activate\n"
        "  AGENT_PRIVACY_GPU=0 python -m uvicorn api.main:app --host 0.0.0.0 --port 8001"
    )

# ── Python Environment ──────────────────────────────────────────
print("\n=== Python Environment ===")

if os.access(PYTHON, os.X_OK):
    ok(f"Python virtualenv found")
else:
    fail(f"Python virtualenv not found at {PYTHON}")
    hints.append("Create virtualenv: mkvirtualenv android_world")

if os.access(PYTHON, os.X_OK):
    import_ok = True
    for mod in ["requests", "cv2", "grpc"]:
        r = subprocess.run(
            [PYTHON, "-c", f"import {mod}"],
            capture_output=True, timeout=10,
        )
        if r.returncode != 0:
            fail(f"Cannot import '{mod}' in virtualenv")
            import_ok = False
    if import_ok:
        ok("Core Python packages importable (requests, cv2, grpc)")

# ── Summary ─────────────────────────────────────────────────────
total = passed + failed
print(f"\n=== Summary ===")
print(f"  Passed: {passed} / {total}")

if failed == 0:
    print("  All checks passed. Ready to run.")
else:
    print(f"  {failed} check(s) failed.")
    if hints:
        print("\n  Hints:")
        for h in hints:
            for line in h.splitlines():
                print(f"    {line}")
    sys.exit(1)
