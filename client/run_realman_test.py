"""Basic running test for PrivacyAwareGUIOwl on any connected ADB device.

Connects to a real Android device (or emulator) via adb, runs the agent on a
single free-form task, and saves all artifacts to a single output directory.

Output layout:
  realman_test_YYYY-MM-DD_HH-MM-SS/
  ├── run.log                        # full log: timing + action_response per step
  ├── ner_mapping_final.json         # NER entity table written once at the end
  ├── debug/                         # debug/visualisation artifacts
  │   ├── step_000_tap.png           # anonymized + crosshair at tap coords (abs_origin)
  │   ├── step_000_raw.png           # original screenshot before any processing
  │   ├── step_000_labeled.png       # original + detection bounding-box overlays
  │   ├── step_000_detection.json    # raw detection results
  │   ├── step_000_context.json      # activity, screen_ctx, requirements, privacy
  │   ├── step_000_server_log.txt    # server-side detect log (when available)
  │   ├── privacy_hooks.log          # detailed privacy pipeline log
  │   └── ...
  └── <GOAL_SUBDIR>/                 # GUIOwl's internal structure — actual model I/O
      ├── action.jsonl               # raw model actions (abs_resized coords)
      └── screenshot_N.png           # resized image sent to model

Device selection:
  --device_serial  explicit serial (from `adb devices`)
  (omit)           auto-detect: picks the only connected device, or errors

Connection modes:
  Real phone / default  : uses SimpleAdbEnv (pure adb, no gRPC required)
  Emulator / --use_avd  : uses env_launcher (needs -grpc emulator flag)

Usage:
  bash run_realman_test.sh --task "Open the Weather app and check today's forecast."
  python run_realman_test.py --task "..." --device_serial <DEVICE_SERIAL>
  python run_realman_test.py --task "..." --use_avd --grpc_port 8554
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "none")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from android_world.agents.infer_ma3 import GUIOwlWrapper
from android_world.agents.privacy.ablation_hooks import (
    NoElemVerifyHooks,
    NoTaskExtractHooks,
    TextOnlyHooks,
)
from android_world.agents.privacy.server_hooks import ServerPrivacyHooks
from android_world.agents.privacy_guiowl import PrivacyAwareGUIOwl


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _list_adb_devices(adb_path: str) -> list[str]:
    r = subprocess.run([adb_path, "devices"], capture_output=True, text=True, timeout=10)
    return [
        line.split()[0]
        for line in r.stdout.splitlines()[1:]
        if len(line.split()) >= 2 and line.split()[1] == "device"
    ]


def _auto_detect_serial(adb_path: str) -> str:
    serials = _list_adb_devices(adb_path)
    if not serials:
        raise RuntimeError(
            "No ADB device found. Connect a device or start an emulator, "
            "then retry. Run `adb devices` to verify."
        )
    if len(serials) > 1:
        raise RuntimeError(
            f"Multiple ADB devices found: {serials}. "
            "Pass --device_serial to specify which one to use."
        )
    return serials[0]


def _build_env(args: argparse.Namespace):
    if args.use_avd:
        from android_world.env import env_launcher
        print(f"Connecting to AVD emulator "
              f"(console_port={args.console_port}, grpc_port={args.grpc_port})...")
        return env_launcher.load_and_setup_env(
            console_port=args.console_port,
            emulator_setup=False,
            freeze_datetime=False,
            adb_path=args.adb_path,
            grpc_port=args.grpc_port,
        )
    else:
        from android_world.env.simple_adb_env import SimpleAdbEnv
        serial = args.device_serial or _auto_detect_serial(args.adb_path)
        print(f"Connecting to device: {serial}")
        return SimpleAdbEnv(adb_path=args.adb_path, device_serial=serial)


# ---------------------------------------------------------------------------
# Logging — writes to run.log inside the output dir
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Path) -> logging.Logger:
    fmt = "%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.WARNING,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    logger = logging.getLogger("realman_test")
    logger.setLevel(logging.INFO)
    logging.getLogger("android_world.agents.privacy").setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Per-step artifact saving
# ---------------------------------------------------------------------------

def _draw_tap_crosshair(
    pixels: np.ndarray, x: int, y: int, label: str = ""
) -> np.ndarray:
    """Return a copy of pixels with a red crosshair drawn at (x, y).

    The crosshair consists of full-width/height lines plus a circle, so the
    tap position is visible even in busy UIs.  A coordinate label is drawn
    near the intersection.
    """
    img = Image.fromarray(pixels.astype("uint8")).copy()
    draw = ImageDraw.Draw(img)
    h, w = pixels.shape[:2]
    red = (255, 50, 50)
    lw = 3
    draw.line([(x, 0), (x, h - 1)], fill=red, width=lw)
    draw.line([(0, y), (w - 1, y)], fill=red, width=lw)
    r = 22
    draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=red, width=lw + 1)
    if label:
        tx = min(x + r + 4, w - 80)
        ty = max(y - r - 20, 2)
        # white shadow for readability
        draw.text((tx + 1, ty + 1), label, fill=(255, 255, 255))
        draw.text((tx, ty), label, fill=red)
    return np.array(img)


def _save_tap_artifact(
    debug_dir: Path,
    step_idx: int,
    anon_pixels: np.ndarray,
    tap_xy: tuple[int, int] | None = None,
) -> list[str]:
    """Save realman-specific tap visualization into debug_dir."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # tap crosshair — anonymized image with click position marked
    if tap_xy is not None:
        x, y = tap_xy
        tap_pixels = _draw_tap_crosshair(
            anon_pixels, x, y, label=f"({x},{y})"
        )
        p = debug_dir / f"step_{step_idx:03d}_tap.png"
        Image.fromarray(tap_pixels).save(p)
        saved.append(p.name)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run GUIOwl or PrivacyAwareGUIOwl on a real/emulator ADB device.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--task", "--instruction",
        dest="task",
        required=True,
        help="Task instruction for the agent.",
    )
    p.add_argument("--output_dir", default="",
                   help="Output directory. Auto-generated as realman_test_<datetime> if empty.")
    p.add_argument("--model", default="GUI-Owl-7B")
    p.add_argument("--api_key", default="EMPTY")
    p.add_argument("--base_url", default="http://localhost:8000/v1")
    p.add_argument("--privacy_url", default="http://localhost:8001")
    p.add_argument("--adb_path",
                   default=os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"))
    p.add_argument("--device_serial", default="")
    p.add_argument("--use_avd", action="store_true",
                   help="Use AndroidWorld AVD emulator instead of SimpleAdbEnv.")
    p.add_argument("--console_port", type=int, default=5554)
    p.add_argument("--grpc_port", type=int, default=8554)
    p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--add_info", default="")
    p.add_argument("--src_format", default="abs_resized")
    p.add_argument("--privacy_timeout", type=float, default=30.0)
    p.add_argument(
        "--agent",
        choices=["privacy", "text_only", "no_elem_verify", "no_task_extract", "guiowl"],
        default="privacy",
        help="Agent to run: privacy, text_only, no_elem_verify, no_task_extract, or guiowl.",
    )
    p.add_argument("--debug", action="store_true",
                   help="Enable debug mode: save step artifacts under output_dir/debug/.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Output directory — created by shell script or auto-generated here
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = SCRIPT_DIR / f"realman_test_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "run.log"
    logger = _setup_logging(log_path)

    print(f"\n{'='*60}")
    print(f"{args.agent} — realman test")
    print(f"{'='*60}")
    print(f"Task      : {args.task}")
    print(f"Model     : {args.model} @ {args.base_url}")
    if args.agent in ("privacy", "text_only", "no_elem_verify", "no_task_extract"):
        print(f"Privacy   : {args.privacy_url}")
    print(f"Artifacts : {out_dir}")
    print(f"{'='*60}\n")

    logger.info("task=%s", args.task)
    logger.info("model=%s base_url=%s", args.model, args.base_url)

    if args.add_info:
        os.environ["ADD_INFO"] = args.add_info

    # ------------------------------------------------------------------
    # Connect to device
    # ------------------------------------------------------------------
    try:
        env = _build_env(args)
        print(f"Device connected. Screen size: {env.logical_screen_size}\n")
        logger.info("screen_size=%s", env.logical_screen_size)
    except Exception as exc:
        print(f"\nERROR: Could not connect to device: {exc}", file=sys.stderr)
        logger.error("device connection failed: %s", exc, exc_info=True)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build agent
    # ------------------------------------------------------------------
    debug_dir = out_dir / "debug"
    vllm = GUIOwlWrapper(api_key=args.api_key, base_url=args.base_url, model_name=args.model)

    if args.agent in ("privacy", "text_only", "no_elem_verify", "no_task_extract"):
        hook_classes = {
            "privacy": ServerPrivacyHooks,
            "text_only": TextOnlyHooks,
            "no_elem_verify": NoElemVerifyHooks,
            "no_task_extract": NoTaskExtractHooks,
        }
        hook_cls = hook_classes[args.agent]
        hooks = hook_cls(
            base_url=args.privacy_url,
            timeout=args.privacy_timeout,
            debug=args.debug,
            debug_dir=str(debug_dir) if args.debug else "",
            llm_model="0.8b",
        )
        if hooks.check_health():
            print("Privacy server: OK\n")
            logger.info("privacy server: reachable")
        else:
            print("Privacy server: NOT REACHABLE — running without privacy processing.\n",
                  file=sys.stderr)
            logger.warning("privacy server not reachable; screenshots unprocessed")
        agent = PrivacyAwareGUIOwl(
            env=env,
            vllm=vllm,
            src_format=args.src_format,
            api_key=args.api_key,
            url=args.base_url,
            hooks=hooks,
            name={
                "privacy": "PrivacyGUIOwl",
                "text_only": "TextOnlyGUIOwl",
                "no_elem_verify": "NoElemVerifyGUIOwl",
                "no_task_extract": "NoTaskExtractGUIOwl",
            }[args.agent],
            output_path=str(out_dir),
        )
    else:  # guiowl
        from android_world.agents import gui_owl
        hooks = None
        agent = gui_owl.GUIOwl(
            env=env,
            vllm=vllm,
            src_format=args.src_format,
            api_key=args.api_key,
            url=args.base_url,
            output_path=str(out_dir),
        )
    agent.transition_pause = None

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    print(f"Starting task (max {args.max_steps} steps)...\n")
    agent.reset(go_home=True)

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"Task: {args.task}\n")
        log_file.write(f"Model: {args.model}  url: {args.base_url}\n")
        log_file.write(f"Started: {out_dir.name}\n\n")

        done = False
        step_idx = 0

        while not done and step_idx < args.max_steps:
            print(f"[Step {step_idx}] Capturing state & calling model...")
            logger.info("--- step %d start ---", step_idx)
            t0 = time.time()

            try:
                result = agent.step(args.task)
            except Exception as exc:
                err = f"Step {step_idx} ERROR: {exc}"
                print(err, file=sys.stderr)
                logger.error(err, exc_info=True)
                log_file.write(f"\n{err}\n")
                break

            elapsed = time.time() - t0

            # --- action_response ---
            action_response = result.data.get("action_response", "")
            if isinstance(action_response, tuple):
                action_response = str(action_response[0])

            # --- extract tap coordinates (abs_origin) from JSONAction ---
            json_action = result.data.get("action")
            tap_xy = None
            if json_action is not None:
                ax = getattr(json_action, "x", None)
                ay = getattr(json_action, "y", None)
                if ax is not None and ay is not None:
                    tap_xy = (int(ax), int(ay))

            tap_info = f" tap=({tap_xy[0]},{tap_xy[1]})" if tap_xy else ""
            print(f"\n[Step {step_idx}] action_response ({elapsed:.1f}s){tap_info}:")
            print("-" * 50)
            print(action_response)
            print("-" * 50)
            print(f"[Step {step_idx}] Status: {'DONE ✓' if result.done else 'continuing...'}\n")

            if args.debug and hooks is not None:
                # Enhanced log format in debug mode (privacy agent only)
                act_name = hooks.last_activity_name or ""
                ctx_name = hooks.last_screen_ctx.value if hooks.last_screen_ctx else "unknown"
                ps = hooks.last_privacy_summary
                n_exp = ps.get("exposed", 0)
                n_mask = ps.get("hard_masked", 0)
                masked_elems = ps.get("hard_masked_elements", [])
                masked_str = " | masked: " + ", ".join(
                    f"{e.get('type','?')}@{[round(v, 2) for v in (e.get('bbox') or [])]}"
                    for e in masked_elems
                ) if masked_elems else ""
                log_file.write(
                    f"\n==== STEP {step_idx}  ({elapsed:.1f}s){tap_info} ====\n"
                )
                log_file.write(f"[ctx]     {act_name} → {ctx_name}\n")
                log_file.write(f"[privacy] {n_exp} expose / {n_mask} hard_mask{masked_str}\n")
                log_file.write(f"[model]\n")
                log_file.write(str(action_response))
                log_file.write(f"\ndone={result.done}\n")
            else:
                log_file.write(f"\n{'='*60}\nSTEP {step_idx}  ({elapsed:.1f}s){tap_info}\n{'='*60}\n")
                log_file.write(str(action_response))
                log_file.write(f"\ndone={result.done}\n")
            log_file.flush()

            # --- debug artifacts: only when --debug flag is set ---
            if args.debug:
                anon_pixels = result.data.get("screenshot")
                if anon_pixels is not None:
                    saved = _save_tap_artifact(
                        debug_dir=debug_dir,
                        step_idx=step_idx,
                        anon_pixels=anon_pixels,
                        tap_xy=tap_xy,
                    )
                    auto_candidates = [
                        f"step_{step_idx:03d}_raw.png",
                        f"step_{step_idx:03d}_labeled.png",
                        f"step_{step_idx:03d}_detection.json",
                        f"step_{step_idx:03d}_context.json",
                        f"step_{step_idx:03d}_server_log.txt",
                        "privacy_hooks.log",
                    ]
                    auto_saved = [
                        name for name in auto_candidates if (debug_dir / name).exists()
                    ]
                    all_saved = auto_saved + saved
                    print(f"[Step {step_idx}] Debug saved: {', '.join(all_saved)}")

            logger.info("step %d done in %.1fs | done=%s", step_idx, elapsed, result.done)
            done = result.done
            step_idx += 1

        # --- final NER mapping (written once at the end, privacy agent only) ---
        if hooks is not None and hooks.current_ner_mapping is not None:
            ner_path = out_dir / "ner_mapping_final.json"
            ner_path.write_text(
                json.dumps(hooks.current_ner_mapping, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"NER mapping saved → ner_mapping_final.json")
            log_file.write(f"\nNER mapping → {ner_path.name}\n")

        summary = f"\nFinished after {step_idx} step(s). "
        summary += "Status: DONE" if done else f"Status: STOPPED (max_steps={args.max_steps})"
        print(f"\n{'='*60}")
        print(summary)
        print(f"Artifacts saved to: {out_dir}")
        print(f"{'='*60}\n")
        log_file.write(f"\n{summary}\n")

    env.close()


if __name__ == "__main__":
    main()
