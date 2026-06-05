#!/usr/bin/env python3
"""Create method-by-task metric tables from a privacy judge_results.json file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


AGENT_ORDER = ["guiowl", "text_only", "no_elem_verify", "no_task_extract", "privacy"]
AGENT_LABELS = {
    "guiowl": "No Protection",
    "text_only": "Text-Only",
    "no_elem_verify": "CAPED w/o Elem Verify",
    "no_task_extract": "CAPED w/o Task Extract",
    "privacy": "Full CAPED",
}


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_REGISTRY_PATH = SCRIPT_DIR / "privacy_task_registry.json"


def _load_task_registry(path: Path = TASK_REGISTRY_PATH) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError(f"{path} has no valid tasks list")
    return [task for task in tasks if task.get("status", "current") != "deprecated"]


TASK_REGISTRY = _load_task_registry()
TASK_ORDER = [str(task["task_id"]) for task in TASK_REGISTRY]
TASK_LABELS = {
    str(task["task_id"]): str(task.get("short_label") or task["task_id"])
    for task in TASK_REGISTRY
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric_text(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def _ordered_values(discovered: set[str], preferred: list[str]) -> list[str]:
    ordered = [item for item in preferred if item in discovered]
    ordered.extend(sorted(discovered - set(ordered)))
    return ordered


def _index_results(runs: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for run in runs:
        agent = run.get("agent")
        task_id = run.get("task_id")
        if not agent or not task_id:
            continue
        indexed[(agent, task_id)] = run
    return indexed


def _metric_table(
    indexed: dict[tuple[str, str], dict[str, Any]],
    agents: list[str],
    tasks: list[str],
    metric_name: str,
) -> str:
    headers = ["Method", *[TASK_LABELS.get(task, task) for task in tasks], "Average"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for agent in agents:
        row = [AGENT_LABELS.get(agent, agent)]
        values: list[float] = []
        for task in tasks:
            run = indexed.get((agent, task))
            if not run:
                row.append("-")
                continue
            if run.get("status") != "ok":
                row.append(str(run.get("status") or "error"))
                continue
            metrics = run.get("metrics") or {}
            value = metrics.get(metric_name)
            if isinstance(value, (int, float)):
                values.append(float(value))
            row.append(_metric_text(value))
        average = sum(values) / len(values) if values else None
        row.append(_metric_text(average))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _attacker_metric_value(run: dict[str, Any], metric_name: str) -> Any:
    attacker = run.get("attacker_judge")
    if not isinstance(attacker, dict):
        return None
    if attacker.get("status") != "ok":
        return attacker.get("status") or "error"
    metrics = attacker.get("metrics") or {}
    return metrics.get(metric_name)


def _attacker_metric_table(
    indexed: dict[tuple[str, str], dict[str, Any]],
    agents: list[str],
    tasks: list[str],
    metric_name: str,
) -> str:
    headers = ["Method", *[TASK_LABELS.get(task, task) for task in tasks], "Average"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for agent in agents:
        row = [AGENT_LABELS.get(agent, agent)]
        values: list[float] = []
        for task in tasks:
            run = indexed.get((agent, task))
            if not run:
                row.append("-")
                continue
            if run.get("status") != "ok":
                row.append(str(run.get("status") or "error"))
                continue
            value = _attacker_metric_value(run, metric_name)
            if isinstance(value, (int, float)):
                values.append(float(value))
            row.append(_metric_text(value))
        average = sum(values) / len(values) if values else None
        row.append(_metric_text(average))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _relative_leakage_reduction_table(
    indexed: dict[tuple[str, str], dict[str, Any]],
    agents: list[str],
    tasks: list[str],
) -> str:
    headers = ["Method", *[TASK_LABELS.get(task, task) for task in tasks], "Average"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for agent in agents:
        row = [AGENT_LABELS.get(agent, agent)]
        values: list[float] = []
        for task in tasks:
            run = indexed.get((agent, task))
            baseline = indexed.get(("guiowl", task))
            if not run or not baseline:
                row.append("-")
                continue
            value = _attacker_metric_value(run, "wslr")
            baseline_value = _attacker_metric_value(baseline, "wslr")
            if not isinstance(value, (int, float)) or not isinstance(baseline_value, (int, float)):
                row.append("N/A")
                continue
            if baseline_value <= 0:
                row.append("N/A")
                continue
            reduction = (baseline_value - value) / baseline_value
            values.append(float(reduction))
            row.append(_metric_text(reduction))
        average = sum(values) / len(values) if values else None
        row.append(_metric_text(average))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _attacker_status_table(
    indexed: dict[tuple[str, str], dict[str, Any]],
    agents: list[str],
    tasks: list[str],
) -> str:
    headers = ["Method", "Runs", "Attacker OK", "Attacker Failures", "Failure Breakdown"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for agent in agents:
        total_runs = 0
        ok_count = 0
        failure_counts: dict[str, int] = {}
        for task in tasks:
            run = indexed.get((agent, task))
            if not run:
                continue
            attacker = run.get("attacker_judge")
            if not isinstance(attacker, dict):
                continue
            total_runs += 1
            status = str(attacker.get("status") or "missing")
            if status == "ok":
                ok_count += 1
            else:
                failure_counts[status] = failure_counts.get(status, 0) + 1

        failure_total = sum(failure_counts.values())
        if failure_counts:
            breakdown = ", ".join(
                f"{status}: {count}"
                for status, count in sorted(failure_counts.items())
            )
        else:
            breakdown = "-"
        lines.append(
            "| "
            + " | ".join([
                AGENT_LABELS.get(agent, agent),
                str(total_runs),
                str(ok_count),
                str(failure_total),
                breakdown,
            ])
            + " |"
        )

    return "\n".join(lines)


def build_report(result_path: Path) -> str:
    data = _load_json(result_path)
    runs = data.get("runs", [])
    if not isinstance(runs, list):
        raise ValueError(f"{result_path} has no valid runs list")

    indexed = _index_results(runs)
    agents = _ordered_values({agent for agent, _ in indexed}, AGENT_ORDER)
    tasks = _ordered_values({task for _, task in indexed}, TASK_ORDER)

    eval_dir = data.get("eval_dir", "")
    model = data.get("model", "")
    timestamp = data.get("timestamp_utc", "")
    has_question_metrics = any(isinstance(run.get("metrics"), dict) for run in runs)
    has_attacker_metrics = any(isinstance(run.get("attacker_judge"), dict) for run in runs)

    sections = [
        "# Privacy Judge Metric Tables",
        f"Source: `{result_path}`\n\nEval dir: `{eval_dir}`\n\nModel: `{model}`\n\nGenerated from judge timestamp: `{timestamp}`",
    ]

    if has_question_metrics:
        sections.extend([
            "## Accuracy\n\n" + _metric_table(indexed, agents, tasks, "accuracy"),
            "## Recall\n\n" + _metric_table(indexed, agents, tasks, "recall"),
        ])

    if has_attacker_metrics:
        sections.extend([
            "## Attacker Judge Status\n\n" + _attacker_status_table(indexed, agents, tasks),
            "## Weighted Seeded Leakage\n\n" + _attacker_metric_table(indexed, agents, tasks, "weighted_seeded_leakage"),
            "## WSLR\n\n" + _attacker_metric_table(indexed, agents, tasks, "wslr"),
            "## Privacy Reduction Rate\n\n" + _attacker_metric_table(indexed, agents, tasks, "privacy_reduction_rate"),
            "## Relative Leakage Reduction vs No Protection\n\n" + _relative_leakage_reduction_table(indexed, agents, tasks),
            "## Unexpected Fact Count\n\n" + _attacker_metric_table(indexed, agents, tasks, "unexpected_fact_count"),
        ])

    sections.append("")
    return "\n\n".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize privacy judge metrics as method-by-task tables.")
    parser.add_argument("judge_results", help="Path to judge_results.json")
    parser.add_argument(
        "--out",
        default="",
        help="Output markdown path. Default: <judge_results_dir>/judge_metric_tables.md",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_path = Path(args.judge_results).resolve()
    if not result_path.is_file():
        print(f"Error: judge result file not found: {result_path}")
        return 1

    out_path = Path(args.out).resolve() if args.out else result_path.with_name("judge_metric_tables.md")
    report = build_report(result_path)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
