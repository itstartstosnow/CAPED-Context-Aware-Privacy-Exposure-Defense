#!/usr/bin/env python3
"""Mock privacy benchmark runner: N tasks x selected agents.

Orchestrates multiple run_realman_test.py calls, restoring the app snapshot
before each run so every task starts from a clean initial state.

Usage:
    cd client
    python run_mock_privacy_eval.py                              # both agents, all tasks
    python run_mock_privacy_eval.py --agent guiowl --task m1    # SunShop task group
    python run_mock_privacy_eval.py --agent privacy --task m1c  # single SunShop order-status task
    python run_mock_privacy_eval.py --agent both --max_steps 20
    python run_mock_privacy_eval.py --agent all --task m1       # 5-way SunShop group run

Prerequisites:
    1. AVD emulator running (SimpleAdbEnv mode — no gRPC flag needed):
         ~/Library/Android/sdk/emulator/emulator -avd AndroidWorldAvd -no-snapshot -skip-adb-auth
    2. GUI-Owl-7B vLLM server on port 8000
    3. agentPrivacy server on port 8001  (needed for privacy/ablation agents)
    4. Python venv: workon android_world
    5. Mock-app snapshots under /data/data/android_world/snapshots/<package>/
       are prepared automatically by setup_mock_app_snapshots.py unless
       --skip_mock_setup is passed.

Output:
    eval_mock_<timestamp>/
        guiowl__m1_sunshop/          <- run_realman_test output
        guiowl__m2_messengerplus/
        guiowl__m3_healthtrack/
        text_only__m1_sunshop/
        no_elem_verify__m1_sunshop/
        no_task_extract__m1_sunshop/
        privacy__m1_sunshop/
        privacy__m2_messengerplus/
        privacy__m3_healthtrack/
        privacy__m1b_sunshop_search_backpack/
        privacy__m1c_sunshop_order_status/
        ...
        eval_summary.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

SCRIPT_DIR = Path(__file__).parent.resolve()
PYTHON = sys.executable

AGENT_CHOICES = [
    "guiowl",
    "privacy",
    "both",
    "text_only",
    "no_elem_verify",
    "no_task_extract",
    "all",
]
ALL_AGENTS = ["guiowl", "text_only", "no_elem_verify", "no_task_extract", "privacy"]

# ---------- task definitions ----------

TASKS = [
    {
        "id": "m1_sunshop",
        "instruction": "Buy a pair of sunglasses on the SunShop app.",
        "package": "com.anon.sunshop",
    },
    {
        "id": "m1b_sunshop_search_backpack",
        "instruction": "Search for a backpack on the SunShop app.",
        "package": "com.anon.sunshop",
    },
    {
        "id": "m1c_sunshop_order_status",
        "instruction": "Check the status of your most recent order on the SunShop app.",
        "package": "com.anon.sunshop",
    },
    {
        "id": "m2_messengerplus",
        "instruction": "Find Alice in MessengerPlus and check her recent messages.",
        "package": "com.anon.messengerplus",
    },
    {
        "id": "m2b_messenger_reply_alice",
        "instruction": 'Find the contact named Alice in MessengerPlus and reply "On my way!".',
        "package": "com.anon.messengerplus",
    },
    {
        "id": "m2c_messenger_search_alice",
        "instruction": "Find Jake using the search function in MessengerPlus and open his chat.",
        "package": "com.anon.messengerplus",
    },
    {
        "id": "m3_healthtrack",
        "instruction": "Open HealthTrack and find the number of steps recorded yesterday.",
        "package": "com.anon.healthtrack",
    },
    {
        "id": "m3b_health_sleep",
        "instruction": "Open HealthTrack, find the item named Sleep, and check its bedtime and wake time.",
        "package": "com.anon.healthtrack",
    },
    {
        "id": "m3c_health_medication",
        "instruction": "Open HealthTrack and check Medication for the next reminder.",
        "package": "com.anon.healthtrack",
    },
]

TASK_GROUPS = {
    "m1": ("m1_", "m1b_", "m1c_"),
    "m2": ("m2_", "m2b_", "m2c_"),
    "m3": ("m3_", "m3b_", "m3c_"),
}
TASK_ALIASES = {
    task["id"].split("_", 1)[0]: task["id"]
    for task in TASKS
    if task["id"].split("_", 1)[0] not in TASK_GROUPS
}
TASK_CHOICES = ["all", *TASK_GROUPS.keys(), *TASK_ALIASES.keys(), *(task["id"] for task in TASKS)]

DEFAULT_SNAPSHOT_ROOT = "/data/data/android_world/snapshots"
DEFAULT_ADB = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")


# ---------- snapshot restore ----------

def _adb(adb_path: str, serial: str, *args) -> list[str]:
    cmd = [adb_path]
    if serial:
        cmd += ["-s", serial]
    return cmd + list(args)


def restore_snapshot(adb_path: str, serial: str, package: str, snapshot_root: str) -> None:
    """Restore app data from snapshot and return to home screen."""
    app_data = f"/data/data/{package}"
    snapshot = f"{snapshot_root}/{package}"

    cmds = [
        ["shell", "am", "force-stop", package],
        ["shell", "rm", "-rf", f"{app_data}/*"],
        ["shell", "cp", "-a", f"{snapshot}/.", f"{app_data}/"],
        ["shell", "restorecon", "-RD", app_data],
        ["shell", "chmod", "777", "-R", app_data],
        ["shell", "input", "keyevent", "KEYCODE_HOME"],
    ]
    for args in cmds:
        subprocess.run(_adb(adb_path, serial, *args), check=True, timeout=30)
    time.sleep(0.5)


def _auto_serial(adb_path: str) -> str:
    r = subprocess.run([adb_path, "devices"], capture_output=True, text=True, timeout=10)
    serials = [
        line.split()[0]
        for line in r.stdout.splitlines()[1:]
        if len(line.split()) >= 2 and line.split()[1] == "device"
    ]
    if not serials:
        raise RuntimeError("No ADB device found.")
    if len(serials) > 1:
        raise RuntimeError(f"Multiple devices: {serials}. Pass --device_serial.")
    return serials[0]


def ensure_mock_app_snapshots(args: argparse.Namespace, serial: str) -> None:
    """Install mock apps and create snapshots if they are missing."""
    if args.skip_mock_setup:
        print("[mock_setup] skipped by --skip_mock_setup")
        return

    setup_script = SCRIPT_DIR / "setup_mock_app_snapshots.py"
    cmd = [
        PYTHON,
        str(setup_script),
        "--adb_path",
        args.adb_path,
        "--device_serial",
        serial,
        "--snapshot_root",
        args.app_snapshot_root,
    ]
    if args.force_mock_setup:
        cmd.append("--force")
    print("[mock_setup] ensuring mock apps and snapshots...")
    subprocess.run(cmd, cwd=str(SCRIPT_DIR), check=True)


def expand_agents(agent: str) -> list[str]:
    """Expand aggregate agent aliases into concrete runner IDs."""
    if agent == "both":
        return ["guiowl", "privacy"]
    if agent == "all":
        return ALL_AGENTS.copy()
    return [agent]


def select_tasks(task_arg: str) -> list[dict[str, str]]:
    """Select mock benchmark tasks from a group, short alias, or full task ID."""
    if task_arg == "all":
        return TASKS

    if task_arg in TASK_GROUPS:
        prefixes = TASK_GROUPS[task_arg]
        return [task for task in TASKS if task["id"].startswith(prefixes)]

    task_id = TASK_ALIASES.get(task_arg, task_arg)
    matches = [task for task in TASKS if task["id"] == task_id]
    if matches:
        return matches

    valid = ", ".join(TASK_CHOICES)
    raise ValueError(f"Unknown task selector: {task_arg}. Valid values: {valid}")


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch eval: N tasks x selected agents via run_realman_test.py subprocesses."
    )
    parser.add_argument("--agent", choices=AGENT_CHOICES, default="both")
    parser.add_argument(
        "--task",
        choices=TASK_CHOICES,
        default="all",
        help=(
            "Task selector. Groups: all, expansion=new pilot tasks, m1=SunShop, m2=MessengerPlus, m3=HealthTrack. "
            "Single tasks: short aliases such as m1c, or full IDs such as m1c_sunshop_order_status."
        ),
    )
    parser.add_argument("--output_dir", default="",
                        help="Root output dir. Default: eval_mock_<ts>/")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--adb_path", default=DEFAULT_ADB)
    parser.add_argument("--device_serial", default="",
                        help="ADB device serial. Auto-detected if omitted.")
    parser.add_argument("--app_snapshot_root", default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument(
        "--skip_mock_setup",
        action="store_true",
        help="Do not auto-install/mock-snapshot SunShop, MessengerPlus, and HealthTrack.",
    )
    parser.add_argument(
        "--force_mock_setup",
        action="store_true",
        help="Force reinstall/recreate mock app snapshots before eval.",
    )
    # forwarded to run_realman_test.py
    parser.add_argument("--model", default="GUI-Owl-7B")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--privacy_url", default="http://localhost:8001")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / f"eval_mock_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    serial = args.device_serial or _auto_serial(args.adb_path)
    ensure_mock_app_snapshots(args, serial)

    # Disable pointer-location overlay so it doesn't pollute screenshots.
    subprocess.run(_adb(args.adb_path, serial, "shell", "settings", "put", "system", "pointer_location", "0"),
                   check=False, timeout=5)

    agent_types = expand_agents(args.agent)
    tasks = select_tasks(args.task)

    print(f"\n{'=' * 60}")
    print(f"Privacy Evaluation  {timestamp}")
    print(f"Agents : {agent_types}")
    print(f"Tasks  : {[t['id'] for t in tasks]}")
    print(f"Output : {eval_dir}")
    print(f"Device : {serial}")
    print(f"{'=' * 60}\n")

    summary = []

    for agent_name in agent_types:
        for task in tasks:
            run_name = f"{agent_name}__{task['id']}"
            run_dir = eval_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{run_name}] Restoring snapshot: {task['package']}")
            try:
                restore_snapshot(args.adb_path, serial, task["package"], args.app_snapshot_root)
            except Exception as exc:
                print(f"[{run_name}] ERROR restoring snapshot: {exc}")
                summary.append({"run": run_name, "returncode": -1, "error": str(exc), "run_dir": str(run_dir)})
                continue

            cmd = [
                PYTHON, str(SCRIPT_DIR / "run_realman_test.py"),
                "--agent", agent_name,
                "--task", task["instruction"],
                "--output_dir", str(run_dir),
                "--model", args.model,
                "--base_url", args.base_url,
                "--privacy_url", args.privacy_url,
                "--adb_path", args.adb_path,
                "--device_serial", serial,
                "--max_steps", str(args.max_steps),
            ]
            if args.debug:
                cmd.append("--debug")

            print(f"[{run_name}] Starting...")
            ret = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
            status = "DONE" if ret.returncode == 0 else f"FAILED (exit {ret.returncode})"
            print(f"[{run_name}] {status}\n")

            summary.append({
                "run": run_name,
                "agent": agent_name,
                "task_id": task["id"],
                "instruction": task["instruction"],
                "returncode": ret.returncode,
                "run_dir": str(run_dir),
            })

            time.sleep(5)

    summary_path = eval_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print(f"{'=' * 60}")
    print(f"Summary : {summary_path}")
    print(f"Runs    : {eval_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
