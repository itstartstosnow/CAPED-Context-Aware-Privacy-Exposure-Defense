#!/usr/bin/env python3
"""AndroidWorld privacy benchmark runner.

Runs Phase 2 benchmark task IDs through the real AndroidWorld lifecycle:
task.initialize_task(env) -> privacy seeding -> agent loop -> task.is_successful(env).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "none")

from android_world import constants
from android_world import registry
from android_world.agents import base_agent
from android_world.agents import gui_owl
from android_world.agents import infer_ma3
from android_world.agents import privacy_guiowl
from android_world.agents.privacy.ablation_hooks import (
    NoElemVerifyHooks,
    NoTaskExtractHooks,
    TextOnlyHooks,
)
from android_world.agents.privacy.server_hooks import ServerPrivacyHooks
from android_world.env import adb_utils
from android_world.env import env_launcher
from android_world.env import interface
from privacy_seeders import seed_for_task
from privacy_tasks.awp_tasks import AWP_LOCAL_TASK_CLASSES
from privacy_tasks.awp_expand import EXPAND_TASK_CLASSES

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_ADB = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")

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

AWP_OFFICIAL_TASK_TO_AW_CLASS = {
    "a1_files_move": "FilesMoveFile",
    "a2_files_delete": "FilesDeleteFile",
    "s1_sms_reply_target_sender": "SimpleSmsReply",
}

TASK_TO_CLASS_NAME = {
    **AWP_OFFICIAL_TASK_TO_AW_CLASS,
    **{
        task_id: task_class.__name__
        for task_id, task_class in AWP_LOCAL_TASK_CLASSES.items()
    },
    **{
        task_id: task_class.__name__
        for task_id, task_class in EXPAND_TASK_CLASSES.items()
    },
}

TASK_TO_SUITE = {
    **{task_id: "awp" for task_id in AWP_OFFICIAL_TASK_TO_AW_CLASS},
    **{task_id: "awp" for task_id in AWP_LOCAL_TASK_CLASSES},
    **{task_id: "expand" for task_id in EXPAND_TASK_CLASSES},
}

AWP_TASK_IDS = [
    "a1_files_move",
    "a2_files_delete",
    "c1_gallery_receipt_copy",
    "s1_sms_reply_target_sender",
    "s2_sms_forward_received_address",
    "g2_gallery_selective_task",
]

TASK_GROUPS = {
    "awp": AWP_TASK_IDS,
    "expand": list(EXPAND_TASK_CLASSES),
    "expandOther": [
        "g3_gallery_share_target_photo",
        "g4_gallery_open_target_screenshot",
        "s3_sms_search_target_thread",
        "s4_sms_compose_to_target_contact",
    ],
    "expandGoogle": [
        "gm1_open_target_sender",
        "gm2_search_target_subject",
        "gm3_read_target_email_detail",
        "gd1_open_target_doc",
        "gd2_search_target_doc",
        "gd3_share_target_file",
        "gp1_open_target_album_photo",
        "gp2_open_target_receipt_album_photo",
        "gp3_share_target_album_photo",
    ],
    "controlled_google": [
        "gm1_open_target_sender",
        "gm2_search_target_subject",
        "gm3_read_target_email_detail",
        "gd1_open_target_doc",
        "gd2_search_target_doc",
        "gd3_share_target_file",
        "gp1_open_target_album_photo",
        "gp2_open_target_receipt_album_photo",
        "gp3_share_target_album_photo",
    ],
    "files": [
        "a1_files_move",
        "a2_files_delete",
        # Deprecated expand candidates; removed from active runs.
    ],
    "gallery": [
        "c1_gallery_receipt_copy",
        "g2_gallery_selective_task",
        "g3_gallery_share_target_photo",
        "g4_gallery_open_target_screenshot",
    ],
    "sms": [
        "s1_sms_reply_target_sender",
        "s2_sms_forward_received_address",
        "s3_sms_search_target_thread",
        "s4_sms_compose_to_target_contact",
    ],
    "all": AWP_TASK_IDS + list(EXPAND_TASK_CLASSES),
}

TASK_CHOICES = list(TASK_TO_CLASS_NAME) + list(TASK_GROUPS)

A2_FIXED_PARAMS = {
    "file_name": "quarterly_budget_archive.pdf",
    "subfolder": "Documents",
}


def expand_agents(agent: str) -> list[str]:
    """Expand aggregate agent aliases into concrete runner IDs."""
    if agent == "both":
        return ["guiowl", "privacy"]
    if agent == "all":
        return ALL_AGENTS.copy()
    return [agent]


def expand_tasks(task: str) -> list[str]:
    """Expand a benchmark task or app-group alias into benchmark task IDs."""
    if task in TASK_TO_CLASS_NAME:
        return [task]
    if task in TASK_GROUPS:
        return TASK_GROUPS[task].copy()
    raise ValueError(f"Unknown task alias: {task}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AndroidWorld privacy eval runner for Phase 2 benchmark tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agent", choices=AGENT_CHOICES, default="both")
    parser.add_argument("--task", choices=TASK_CHOICES, default="all")
    parser.add_argument(
        "--output_dir",
        default="",
        help="Root output dir. Default: eval_awp_<timestamp>/",
    )
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--adb_path", default=DEFAULT_ADB)
    parser.add_argument(
        "--device_serial",
        default="",
        help="Accepted for CLI symmetry; AndroidWorld env uses --console_port instead.",
    )
    parser.add_argument("--console_port", type=int, default=5554)
    parser.add_argument("--grpc_port", type=int, default=8554)
    parser.add_argument("--perform_emulator_setup", action="store_true")
    parser.add_argument("--task_random_seed", type=int, default=30)
    parser.add_argument("--model", default="GUI-Owl-7B")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--privacy_url", default="http://localhost:8001")
    parser.add_argument("--privacy_timeout", type=float, default=30.0)
    parser.add_argument("--llm_model", default="0.8b", choices=["0.8b", "2b"])
    return parser.parse_args()


def _resolve_task_class(benchmark_task_id: str) -> type:
    if benchmark_task_id in AWP_LOCAL_TASK_CLASSES:
        return AWP_LOCAL_TASK_CLASSES[benchmark_task_id]
    if benchmark_task_id in EXPAND_TASK_CLASSES:
        return EXPAND_TASK_CLASSES[benchmark_task_id]

    aw_class_name = AWP_OFFICIAL_TASK_TO_AW_CLASS[benchmark_task_id]
    aw_registry = registry.TaskRegistry().get_registry(
        registry.TaskRegistry.ANDROID_WORLD_FAMILY
    )
    if aw_class_name not in aw_registry:
        raise KeyError(f"AndroidWorld task class {aw_class_name!r} is not registered.")
    return aw_registry[aw_class_name]


def _instantiate_task(benchmark_task_id: str, env: interface.AsyncEnv, seed: int):
    task_type = _resolve_task_class(benchmark_task_id)
    task_type.set_device_time(env)
    random.seed(seed)
    params = task_type.generate_random_params()
    if benchmark_task_id == "a2_files_delete":
        # Keep the benchmark in the Files app instead of letting a random
        # image filename steer the model toward a gallery app.
        params.update(A2_FIXED_PARAMS)
    params[constants.EpisodeConstants.SEED] = seed
    return task_type(params)


def _build_hooks(agent_name: str, args: argparse.Namespace, run_dir: Path):
    debug_dir = str(run_dir / "debug") if args.debug else ""
    kwargs = {
        "base_url": args.privacy_url,
        "timeout": args.privacy_timeout,
        "debug": args.debug,
        "debug_dir": debug_dir,
        "llm_model": args.llm_model,
    }
    if agent_name == "privacy":
        return ServerPrivacyHooks(**kwargs)
    if agent_name == "text_only":
        return TextOnlyHooks(**kwargs)
    if agent_name == "no_elem_verify":
        return NoElemVerifyHooks(**kwargs)
    if agent_name == "no_task_extract":
        return NoTaskExtractHooks(**kwargs)
    raise ValueError(f"Agent {agent_name!r} does not use privacy hooks.")


def _build_agent(
    agent_name: str,
    env: interface.AsyncEnv,
    args: argparse.Namespace,
    run_dir: Path,
) -> base_agent.EnvironmentInteractingAgent:
    vllm = infer_ma3.GUIOwlWrapper(args.api_key, args.base_url, args.model)
    if agent_name == "guiowl":
        agent = gui_owl.GUIOwl(
            env,
            vllm,
            "abs_resized",
            api_key=None,
            url=None,
            name="guiowl",
            output_path=str(run_dir),
        )
    else:
        hooks = _build_hooks(agent_name, args, run_dir)
        agent = privacy_guiowl.PrivacyAwareGUIOwl(
            env=env,
            vllm=vllm,
            src_format="abs_resized",
            api_key=None,
            url=None,
            hooks=hooks,
            name=agent_name,
            output_path=str(run_dir),
        )
    agent.name = agent_name
    agent.transition_pause = None
    return agent


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return repr(value)


def _resolve_eval_dir(output_dir: str, timestamp: str) -> Path:
    """Resolve eval output under this script dir unless an absolute path is used."""
    if not output_dir:
        return SCRIPT_DIR / f"eval_awp_{timestamp}"
    path = Path(output_dir).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2) + "\n")


class _TeeStream:
    """Mirror text writes to the live console and the per-run log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)


@contextlib.contextmanager
def _capture_run_output(log_path: Path):
    """Capture prints, tracebacks, and Python logger output for one run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s")
    )
    root_logger.addHandler(file_handler)

    with log_path.open("a", encoding="utf-8") as log_file:
        stdout_tee = _TeeStream(sys.stdout, log_file)
        stderr_tee = _TeeStream(sys.stderr, log_file)
        try:
            with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
                yield
        finally:
            root_logger.removeHandler(file_handler)
            file_handler.close()


def _preflight_android_world_env(env: interface.AsyncEnv, eval_dir: Path) -> None:
    """Fail fast if AndroidWorld gRPC/a11y observation is not usable."""
    try:
        state = env.get_state()
        if state.forest is None:
            raise RuntimeError("AndroidWorld env returned state.forest=None.")
    except Exception as exc:
        diagnostic = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "hint": (
                "AndroidWorld Phase 2 needs the emulator started with -grpc 8554 "
                "and a working AccessibilityForwarder a11y tree. Restart the AVD "
                "with the command from CLAUDE.md, then rerun this eval."
            ),
        }
        _write_json(eval_dir / "preflight_error.json", diagnostic)
        raise RuntimeError(
            f"AndroidWorld preflight failed before running tasks: {exc}. "
            f"Details written to {eval_dir / 'preflight_error.json'}"
        ) from exc


def _run_agent_loop(
    agent: base_agent.EnvironmentInteractingAgent,
    task_goal: str,
    max_steps: int,
) -> bool:
    agent.reset(go_home=True)
    agent.set_max_steps(max_steps)
    is_done = False
    for step_idx in range(max_steps):
        result = agent.step(task_goal)
        print(f"Completed step {step_idx + 1}.")
        if result.done:
            print("Agent indicates task is done.")
            is_done = True
            break
    if not is_done:
        print("Agent reached max steps without indicating done.")
    return is_done


def _adb_shell(
    env: interface.AsyncEnv,
    args: list[str],
    timeout_sec: float = 10.0,
) -> str:
    response = adb_utils.issue_generic_request(
        ["shell", *args], env.controller, timeout_sec=timeout_sec
    )
    output = response.generic.output
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace").strip()
    return str(output).strip()


def _log_adb_probe(
    env: interface.AsyncEnv,
    log: Any,
    label: str,
    args: list[str],
    timeout_sec: float = 10.0,
) -> str:
    try:
        output = _adb_shell(env, args, timeout_sec=timeout_sec)
    except Exception as exc:  # pragma: no cover - diagnostic path.
        output = f"<error: {exc}>"
    log(f"[network] {label}: {output}")
    return output


def _sync_device_time_from_host(env: interface.AsyncEnv, log: Any) -> None:
    host_epoch = int(time.time())
    _log_adb_probe(env, log, "date-before", ["date", "-u"])
    _log_adb_probe(
        env,
        log,
        "set-date",
        ["su", "0", "date", "-u", f"@{host_epoch}"],
    )
    _log_adb_probe(env, log, "date-after", ["date", "-u"])


def _network_probe_ok(env: interface.AsyncEnv, log: Any) -> bool:
    ip_ping = _log_adb_probe(
        env,
        log,
        "ping-8.8.8.8",
        ["ping", "-c", "1", "-W", "3", "8.8.8.8"],
        timeout_sec=6.0,
    )
    dns_ping = _log_adb_probe(
        env,
        log,
        "ping-google.com",
        ["ping", "-c", "1", "-W", "3", "google.com"],
        timeout_sec=6.0,
    )
    return "1 received" in ip_ping and "1 received" in dns_ping


def _warmup_gmail_sync_if_needed(task: Any, env: interface.AsyncEnv, log: Any) -> None:
    if not str(getattr(task, "name", "")).startswith("gm"):
        return

    log(f"[{task.name}] warming up Gmail sync/indexing.")
    _log_adb_probe(
        env, log, "gmail-force-stop", ["am", "force-stop", "com.google.android.gm"]
    )
    _log_adb_probe(
        env, log, "gmail-launch", ["monkey", "-p", "com.google.android.gm", "1"]
    )
    _log_adb_probe(
        env,
        log,
        "gmail-sync-request",
        [
            "content",
            "call",
            "--uri",
            "content://com.android.contacts",
            "--method",
            "requestSync",
            "--arg",
            "gmail-ls",
        ],
    )
    time.sleep(15.0)


def _warmup_drive_if_needed(task: Any, env: interface.AsyncEnv, log: Any) -> None:
    if not str(getattr(task, "name", "")).startswith("gd"):
        return

    log(f"[{task.name}] warming up Drive launch.")
    _log_adb_probe(
        env,
        log,
        "drive-force-stop",
        ["am", "force-stop", "com.google.android.apps.docs"],
    )
    _log_adb_probe(
        env,
        log,
        "drive-launch",
        ["monkey", "-p", "com.google.android.apps.docs", "1"],
    )
    time.sleep(2.0)
    _log_adb_probe(env, log, "drive-home", ["input", "keyevent", "KEYCODE_HOME"])


def _ensure_task_network_if_required(
    task: Any,
    env: interface.AsyncEnv,
    log: Any,
) -> None:
    """Enable AVD networking only for tasks that explicitly require it."""

    if not getattr(task, "requires_network", False):
        return

    log(f"[{task.name}] requires_network=True; enabling AVD networking.")
    adb_utils.toggle_airplane_mode("off", env.controller)
    adb_utils.toggle_wifi(env.controller, "on")
    _log_adb_probe(env, log, "data-enable", ["svc", "data", "enable"])
    _sync_device_time_from_host(env, log)
    _log_adb_probe(
        env,
        log,
        "airplane_mode_on",
        ["settings", "get", "global", "airplane_mode_on"],
    )
    _log_adb_probe(
        env, log, "http_proxy", ["settings", "get", "global", "http_proxy"]
    )
    _log_adb_probe(env, log, "no_proxy", ["settings", "get", "global", "no_proxy"])
    _log_adb_probe(
        env,
        log,
        "private_dns_mode",
        ["settings", "get", "global", "private_dns_mode"],
    )

    ready = False
    for attempt in range(1, 7):
        log(f"[network] readiness probe {attempt}/6")
        if _network_probe_ok(env, log):
            ready = True
            break
        time.sleep(5.0)
    if not ready:
        log(
            f"[{task.name}] WARNING: network readiness probe failed; "
            "continuing so the run log preserves the task failure."
        )

    _warmup_gmail_sync_if_needed(task, env, log)
    _warmup_drive_if_needed(task, env, log)


def _run_one(
    agent_name: str,
    benchmark_task_id: str,
    env: interface.AsyncEnv,
    args: argparse.Namespace,
    eval_dir: Path,
) -> dict[str, Any]:
    run_name = f"{agent_name}__{benchmark_task_id}"
    run_dir = eval_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    log_path.write_text("", encoding="utf-8")

    summary: dict[str, Any] = {
        "run": run_name,
        "agent": agent_name,
        "task_id": benchmark_task_id,
        "task_suite": TASK_TO_SUITE[benchmark_task_id],
        "task_class": TASK_TO_CLASS_NAME[benchmark_task_id],
        "aw_task_class": AWP_OFFICIAL_TASK_TO_AW_CLASS.get(benchmark_task_id),
        "local_task_class": (
            TASK_TO_CLASS_NAME[benchmark_task_id]
            if benchmark_task_id in AWP_LOCAL_TASK_CLASSES
            or benchmark_task_id in EXPAND_TASK_CLASSES
            else None
        ),
        "run_dir": str(run_dir),
        "returncode": 0,
        "task_successful": 0.0,
        "raw_task_successful": 0.0,
        "manual_review_required": False,
        "agent_done": False,
        "error": None,
    }

    task = None
    def log(message: str) -> None:
        print(message)

    with _capture_run_output(log_path):
        try:
            task = _instantiate_task(benchmark_task_id, env, args.task_random_seed)
            summary["params"] = _json_safe(task.params)
            summary["goal"] = task.goal
            summary["manual_review_required"] = bool(getattr(task, "manual_review_required", False))
            summary["requires_network"] = bool(getattr(task, "requires_network", False))

            log(f"[{run_name}] Initializing {summary['task_suite']} task: {task.name}")
            task.initialize_task(env)

            log(f"[{run_name}] Running privacy seeder.")
            seed_for_task(benchmark_task_id, env.controller, task=task, env=env)

            _ensure_task_network_if_required(task, env, log)

            agent = _build_agent(agent_name, env, args, run_dir)
            agent.task_name[task.goal] = "screenshots"

            log(f"[{run_name}] Starting agent loop.")
            agent_done = _run_agent_loop(agent, task.goal, args.max_steps)
            task_successful = float(task.is_successful(env))

            summary["agent_done"] = agent_done
            summary["raw_task_successful"] = task_successful
            summary["task_successful"] = task_successful if agent_done else 0.0
            status = "DONE" if summary["task_successful"] > 0.5 else "FAILED"
            log(f"[{run_name}] {status}; successful={summary['task_successful']}")
        except KeyboardInterrupt as exc:
            summary["returncode"] = -130
            summary["error"] = "Interrupted by user"
            summary["traceback"] = traceback.format_exc()
            log(f"[{run_name}] INTERRUPTED: {exc}")
            raise
        except Exception as exc:  # pylint: disable=broad-except
            summary["returncode"] = -1
            summary["error"] = str(exc)
            summary["traceback"] = traceback.format_exc()
            log(f"[{run_name}] ERROR: {exc}")
            traceback.print_exc()
        finally:
            if task is not None and getattr(task, "initialized", False):
                try:
                    task.tear_down(env)
                except Exception as exc:  # pylint: disable=broad-except
                    summary["tear_down_error"] = str(exc)
                    log(f"[{run_name}] WARNING: tear_down failed: {exc}")
            try:
                env.reset(go_home=True)
            except Exception as exc:  # pylint: disable=broad-except
                summary["env_reset_error"] = str(exc)
            try:
                adb_utils.press_home_button(env.controller)
            except Exception:
                pass
            _write_json(run_dir / "run_summary.json", summary)

    return summary


def _validate_task_registry(task_ids: list[str]) -> None:
    for task_id in task_ids:
        _resolve_task_class(task_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = _resolve_eval_dir(args.output_dir, timestamp)
    eval_dir.mkdir(parents=True, exist_ok=True)

    agent_types = expand_agents(args.agent)
    task_ids = expand_tasks(args.task)
    _validate_task_registry(task_ids)

    print(f"\n{'=' * 60}")
    print(f"AndroidWorld Privacy Evaluation  {timestamp}")
    print(f"Agents : {agent_types}")
    print(f"Tasks  : {task_ids}")
    print(f"Output : {eval_dir}")
    print(f"Console: {args.console_port}")
    print(f"gRPC   : {args.grpc_port}")
    print(f"{'=' * 60}\n")

    summary = []
    env = None
    try:
        env = env_launcher.load_and_setup_env(
            console_port=args.console_port,
            emulator_setup=args.perform_emulator_setup,
            adb_path=args.adb_path,
            grpc_port=args.grpc_port,
        )
        _preflight_android_world_env(env, eval_dir)
        for agent_name in agent_types:
            for benchmark_task_id in task_ids:
                run_summary = _run_one(agent_name, benchmark_task_id, env, args, eval_dir)
                summary.append(run_summary)
                _write_json(eval_dir / "eval_summary.json", summary)
    except Exception as exc:  # pylint: disable=broad-except
        if not summary:
            summary.append({
                "run": "__preflight__",
                "returncode": -1,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
        raise
    finally:
        summary_path = eval_dir / "eval_summary.json"
        _write_json(summary_path, summary)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    print(f"{'=' * 60}")
    print(f"Summary : {summary_path}")
    print(f"Runs    : {eval_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
