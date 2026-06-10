#!/usr/bin/env python3
"""Measure TaskExtractor latency with cached app labels on the Apr 17 task set.

This does not start AndroidWorld tasks or query the emulator for installed apps.
It reuses the anonymized goals recorded in the Apr 17 logs, builds the same V2
TaskExtractor prompt with a fixed cached app list, calls the configured LLM
endpoint, parses the output, applies the existing fallback retry when needed,
and runs to_expanded().
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
EVAL_DIR = THIS_DIR.parent
ANDROID_WORLD_DIR = EVAL_DIR.parent
PROJECT_ROOT = ANDROID_WORLD_DIR
sys.path.insert(0, str(PROJECT_ROOT))

from android_world.agents.privacy import task_extractor as te  # noqa: E402


RUNNING_RE = re.compile(r"Running task (?P<task>.+?) with goal \"(?P<goal>.*)\"$")
TASK_RE = re.compile(r"=== Task: (?P<task>.+?) ===")
NER_RE = re.compile(r"NER: (?P<n>\d+) entities found; anonymized goal: (?P<goal>.*)$")


def load_cached_app_labels(cache_path: Path) -> list[str]:
    with cache_path.open(encoding="utf-8") as f:
        data = json.load(f)
    labels = data.get("labels")
    if not isinstance(labels, list):
        raise ValueError(f"Malformed app label cache: {cache_path}")
    cleaned = sorted({str(label).strip() for label in labels if str(label).strip()}, key=str.casefold)
    if not cleaned:
        raise ValueError(f"Empty app label cache: {cache_path}")
    return cleaned


def parse_goals_from_logs(log_paths: list[Path]) -> tuple[dict[str, str], dict[str, str]]:
    raw_goals: dict[str, str] = {}
    anonymized_goals: dict[str, str] = {}

    for log_path in log_paths:
        current_task: str | None = None
        awaiting_ner = False
        with log_path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                running = RUNNING_RE.search(line)
                if running:
                    task = running.group("task")
                    raw_goals.setdefault(task, running.group("goal"))
                    current_task = task
                    awaiting_ner = True
                    continue

                task_match = TASK_RE.search(line)
                if task_match and not awaiting_ner:
                    current_task = task_match.group("task")
                    continue

                ner_match = NER_RE.search(line)
                if ner_match and current_task:
                    anonymized_goals.setdefault(current_task, ner_match.group("goal"))
                    awaiting_ner = False

    return raw_goals, anonymized_goals


def parse_anonymized_goals_from_debug_logs(eval_dir: Path) -> dict[str, str]:
    anonymized_goals: dict[str, str] = {}
    pattern = re.compile(
        r"NER: \d+ entities found; anonymized goal: (?P<goal>.*?)(?=\n\d{4}-\d{2}-\d{2} |\Z)",
        re.DOTALL,
    )
    for hook_log in sorted(eval_dir.glob("eval_privacy_2026-04-17_*/checkpoints/debug/*/privacy_hooks.log")):
        task = hook_log.parent.name
        text = hook_log.read_text(encoding="utf-8", errors="replace")
        match = pattern.search(text)
        if match:
            anonymized_goals.setdefault(task, match.group("goal").strip())
    return anonymized_goals


def load_task_order(timing_csv: Path) -> list[dict[str, str]]:
    with timing_csv.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_prompt(goal: str, app_labels: list[str]) -> str:
    installed_app_list = "\n".join(app_labels)
    return te._SYSTEM_PROMPT_V2.replace("{installed_app_list}", installed_app_list) + f"\nTask: {te._clean_task(goal)}\n"


def run_one(
    extractor: te.TaskExtractor,
    task_name: str,
    goal: str,
    app_labels: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    timings: dict[str, float] = {}
    t_total0 = time.perf_counter()

    t0 = time.perf_counter()
    prompt = build_prompt(goal, app_labels)
    timings["t_prompt_build_s"] = time.perf_counter() - t0

    status = "ok"
    error = ""
    raw_v2 = ""
    raw_v1 = ""
    used_fallback = False

    try:
        t0 = time.perf_counter()
        raw_v2 = extractor._call_llm_v2(prompt)
        timings["t_llm_v2_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        reqs = extractor._parse(raw_v2)
        timings["t_parse_v2_s"] = time.perf_counter() - t0

        if not reqs.requirements:
            used_fallback = True
            prompt_v1 = te._SYSTEM_PROMPT_V1_RETRY + f"\nTask: {te._clean_task(goal)}\n"

            t0 = time.perf_counter()
            raw_v1 = extractor._call_llm_v1_retry(prompt_v1)
            timings["t_llm_v1_fallback_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            reqs = extractor._parse(raw_v1)
            timings["t_parse_v1_fallback_s"] = time.perf_counter() - t0
        else:
            timings["t_llm_v1_fallback_s"] = 0.0
            timings["t_parse_v1_fallback_s"] = 0.0

        t0 = time.perf_counter()
        expanded = extractor.to_expanded(reqs)
        timings["t_to_expanded_s"] = time.perf_counter() - t0
        requirements = [(r.app, r.subject, r.type) for r in reqs.requirements]
        expanded_requirements = [
            {
                "app_names": r.app_names,
                "subject_type": r.subject_type,
                "subject": r.subject,
            }
            for r in expanded.requirements
        ]
    except Exception as exc:  # pylint: disable=broad-except
        status = "error"
        error = repr(exc)
        reqs = te.TaskRequirements()
        requirements = []
        expanded_requirements = []
        for key in (
            "t_llm_v2_s",
            "t_parse_v2_s",
            "t_llm_v1_fallback_s",
            "t_parse_v1_fallback_s",
            "t_to_expanded_s",
        ):
            timings.setdefault(key, 0.0)

    total = time.perf_counter() - t_total0
    timings["cached_app_list_task_extract_s"] = total

    row: dict[str, Any] = {
        "task": task_name,
        "status": status,
        "error": error,
        "used_fallback": used_fallback,
        "n_requirements": len(reqs.requirements),
        **{k: round(v, 6) for k, v in timings.items()},
    }
    raw: dict[str, Any] = {
        "task": task_name,
        "status": status,
        "error": error,
        "goal_used": goal,
        "raw_v2": raw_v2,
        "raw_v1_fallback": raw_v1,
        "requirements": requirements,
        "expanded_requirements": expanded_requirements,
        "timing": row,
    }
    return row, raw


def summarize(rows: list[dict[str, Any]], out_path: Path, args: argparse.Namespace) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    fields = [
        "cached_app_list_task_extract_s",
        "t_prompt_build_s",
        "t_llm_v2_s",
        "t_parse_v2_s",
        "t_llm_v1_fallback_s",
        "t_parse_v1_fallback_s",
        "t_to_expanded_s",
    ]

    lines = [
        "# Cached App List TaskExtractor Timing",
        "",
        "## Method",
        "",
        "- Source task set: Apr 17 AndroidWorld 116-task privacy run.",
        "- Input text: anonymized goals parsed from the Apr 17 logs.",
        "- Tasks without an anonymized goal are skipped by default because they did not reach the old privacy hook extraction boundary.",
        "- App list: fixed cached labels from `android_world/agents/privacy/installed_app_labels_cache.json`.",
        "- Excluded: emulator reset, task execution, screenshots, privacy per-step processing, and live ADB/Pixel Launcher app discovery.",
        "- Included: prompt construction, V2 LLM HTTP request, V2 parse, optional V1 fallback LLM+parse, and `to_expanded()`.",
        "",
        "## Configuration",
        "",
        f"- base_url: `{args.base_url}`",
        f"- model: `{args.model}`",
        f"- repeats: `{args.repeats}`",
        f"- tasks requested: `{len(rows)}`",
        f"- successful rows: `{len(ok_rows)}`",
        f"- fallback rows: `{sum(1 for r in ok_rows if r['used_fallback'])}`",
        "",
        "## Summary",
        "",
        "| metric | mean_s | median_s | min_s | max_s |",
        "|---|---:|---:|---:|---:|",
    ]

    for field in fields:
        vals = [float(r[field]) for r in ok_rows]
        if vals:
            lines.append(
                f"| {field} | {statistics.mean(vals):.3f} | "
                f"{statistics.median(vals):.3f} | {min(vals):.3f} | {max(vals):.3f} |"
            )
        else:
            lines.append(f"| {field} | n/a | n/a | n/a | n/a |")

    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "`cached_app_list_task_extract_s` is comparable to the old Apr 17 "
            "`t_task_extract_s` except that live installed-app discovery is replaced "
            "by the cached app list. It is not a pure LLM-only metric.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=THIS_DIR)
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--model", choices=sorted(te.LLM_ENDPOINTS), default=te.DEFAULT_LLM_MODEL)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Debug only: run first N rows.")
    parser.add_argument("--task", action="append", default=[], help="Debug only: run selected task name(s).")
    parser.add_argument(
        "--include-raw-fallback",
        action="store_true",
        help="Also run tasks that have only a raw logged goal. Default skips them.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timing_csv = args.eval_dir / "timing_per_task.csv"
    log_paths = sorted(args.eval_dir.glob("eval_privacy_2026-04-17_*/log_ma3_*.log"))
    cache_path = PROJECT_ROOT / "android_world/agents/privacy/installed_app_labels_cache.json"

    task_rows = load_task_order(timing_csv)
    raw_goals, anonymized_goals = parse_goals_from_logs(log_paths)
    anonymized_goals = {**anonymized_goals, **parse_anonymized_goals_from_debug_logs(args.eval_dir)}
    app_labels = load_cached_app_labels(cache_path)
    extractor = te.TaskExtractor(server_base_url=args.base_url, llm_model=args.model)

    selected: list[dict[str, str]] = []
    task_filter = set(args.task)
    for task_row in task_rows:
        task = task_row["task"]
        if task_filter and task not in task_filter:
            continue
        selected.append(task_row)
    if args.limit:
        selected = selected[: args.limit]

    all_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []
    csv_path = args.out_dir / "per_task_timing.csv"
    raw_path = args.out_dir / "raw_outputs.jsonl"
    skipped_path = args.out_dir / "skipped_tasks.csv"

    csv_fields = [
        "repeat",
        "task",
        "timing_status_apr17",
        "goal_source",
        "status",
        "error",
        "used_fallback",
        "n_requirements",
        "cached_app_list_task_extract_s",
        "t_prompt_build_s",
        "t_llm_v2_s",
        "t_parse_v2_s",
        "t_llm_v1_fallback_s",
        "t_parse_v1_fallback_s",
        "t_to_expanded_s",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as csv_f, raw_path.open("w", encoding="utf-8") as raw_f:
        writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
        writer.writeheader()
        for repeat in range(1, args.repeats + 1):
            for task_row in selected:
                task = task_row["task"]
                if task in anonymized_goals:
                    goal = anonymized_goals[task]
                    goal_source = "anonymized_log"
                elif args.include_raw_fallback and task in raw_goals:
                    goal = raw_goals[task]
                    goal_source = "raw_log_fallback"
                else:
                    goal = ""
                    goal_source = "missing"

                if not goal:
                    skipped_rows.append(
                        {
                            "repeat": str(repeat),
                            "task": task,
                            "timing_status_apr17": task_row.get("timing_status", ""),
                            "reason": "missing anonymized goal; task did not reach old privacy hook extraction boundary",
                            "raw_goal_available": str(task in raw_goals),
                        }
                    )
                    raw = {
                        "repeat": repeat,
                        "task": task,
                        "status": "skipped",
                        "reason": skipped_rows[-1]["reason"],
                        "raw_goal_available": task in raw_goals,
                    }
                    raw_f.write(json.dumps(raw, ensure_ascii=False) + "\n")
                    continue

                row, raw = run_one(extractor, task, goal, app_labels)
                row = {
                    "repeat": repeat,
                    "timing_status_apr17": task_row.get("timing_status", ""),
                    "goal_source": goal_source,
                    **row,
                }

                raw["repeat"] = repeat
                raw["goal_source"] = goal_source
                raw["timing_status_apr17"] = task_row.get("timing_status", "")
                writer.writerow({field: row.get(field, "") for field in csv_fields})
                raw_f.write(json.dumps(raw, ensure_ascii=False) + "\n")
                all_rows.append(row)

    with skipped_path.open("w", encoding="utf-8", newline="") as skipped_f:
        writer = csv.DictWriter(
            skipped_f,
            fieldnames=["repeat", "task", "timing_status_apr17", "reason", "raw_goal_available"],
        )
        writer.writeheader()
        writer.writerows(skipped_rows)

    summarize(all_rows, args.out_dir / "summary.md", args)
    print(f"Wrote {csv_path}")
    print(f"Wrote {raw_path}")
    print(f"Wrote {skipped_path}")
    print(f"Wrote {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
