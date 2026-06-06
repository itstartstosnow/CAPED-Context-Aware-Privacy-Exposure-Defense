#!/usr/bin/env python3
"""Build paper-ready privacy benchmark tables from raw eval directories."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


METHOD_ORDER = ["guiowl", "text_only", "no_elem_verify", "no_task_extract", "privacy"]
METHOD_LABELS = {
    "guiowl": "No Protection",
    "text_only": "Text-Only",
    "no_elem_verify": "CAPED w/o Element Verification",
    "no_task_extract": "CAPED w/o Task Extraction",
    "privacy": "Full CAPED",
}
CATEGORY_ORDER = [
    "Mock apps",
    "AndroidWorld-style local apps",
    "Real connected apps",
]
DEFAULT_EXPERIMENTS = {
    "mock": "evaluation/results_metadata/eval_mock_all_updated_260511_1838",
    "awp": "evaluation/results_metadata/eval_awp_all_updated_260512_1744",
    "extend_other": "evaluation/results_metadata/eval_expandOther_all_20260604_full_expand",
    "extend_google": "evaluation/results_metadata/eval_expandGoogle_all_20260604_full_expand",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def rel_or_abs(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def markdown_table(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        cells = []
        for name in fieldnames:
            value = row.get(name, "")
            cells.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_classification(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {
        "task_id",
        "task_name / instruction",
        "app_name",
        "original_experiment_group",
        "paper_category",
        "classification_reason",
    }
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path} missing classification columns: {sorted(missing)}")
    return {row["task_id"]: row for row in rows}


def parse_success_file(path: Path) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    scores: dict[str, float] = {}
    if not path.is_file():
        return scores, [f"Missing success record file: {path}"]

    line_re = re.compile(
        r"^\s*(?:"
        r"\|\s*(?P<table_score>[01](?:\.5)?|0\.5)\s*\|\s*(?P<table_run>[a-z][^|\s]+)\s*\|"
        r"|(?P<run>[a-z][A-Za-z0-9_]+__[A-Za-z0-9_]+)\s+(?P<score>[01](?:\.5)?|0\.5)\b"
        r")"
    )
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = line_re.search(line)
        if not match:
            continue
        run = match.group("table_run") or match.group("run")
        score_text = match.group("table_score") or match.group("score")
        if not run or score_text is None:
            continue
        try:
            scores[run.strip()] = float(score_text)
        except ValueError:
            warnings.append(f"Could not parse success score at {path}:{lineno}: {line}")
    return scores, warnings


def success_file_for(eval_dir: Path) -> Path:
    for name in ("TaskSuccessRate.md", "successRate.md"):
        candidate = eval_dir / name
        if candidate.is_file():
            return candidate
    return eval_dir / "TaskSuccessRate.md"


def experiment_paths(args: argparse.Namespace, base: Path) -> dict[str, Path]:
    paths = {key: rel_or_abs(value, base) for key, value in DEFAULT_EXPERIMENTS.items()}
    for item in args.experiment:
        if "=" not in item:
            raise ValueError(f"--experiment must use group=path, got: {item}")
        group, path_text = item.split("=", 1)
        paths[group] = rel_or_abs(path_text, base)
    return paths


def index_eval_summary(eval_dir: Path) -> dict[str, dict[str, Any]]:
    summary_path = eval_dir / "eval_summary.json"
    if not summary_path.is_file():
        return {}
    payload = load_json(summary_path)
    runs = payload if isinstance(payload, list) else payload.get("runs", [])
    indexed = {}
    for run in runs if isinstance(runs, list) else []:
        run_name = run.get("run") or f"{run.get('agent')}__{run.get('task_id')}"
        indexed[run_name] = run
    return indexed


def leaked_items(attacker: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(attacker, dict):
        return []
    mapping = attacker.get("seeded_mapping") or {}
    items = mapping.get("items") or []
    leaked = []
    for item in items:
        if item.get("leaked") and not item.get("task_required"):
            leaked.append({
                "id": item.get("id"),
                "category": item.get("category"),
                "weight": item.get("weight"),
                "evidence": item.get("evidence"),
                "screenshot_indices": item.get("screenshot_indices") or [],
            })
    return leaked


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def build_rows(classification: dict[str, dict[str, str]], experiments: dict[str, Path]) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
    warnings: list[str] = []
    task_rows: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()

    for group, eval_dir in experiments.items():
        if not eval_dir.is_dir():
            warnings.append(f"Missing result directory for {group}: {eval_dir}")
            continue
        judge_path = eval_dir / "judge_results.json"
        if not judge_path.is_file():
            warnings.append(f"Missing judge_results.json for {group}: {judge_path}")
            continue
        judge_payload = load_json(judge_path)
        runs = judge_payload.get("runs", [])
        if not isinstance(runs, list):
            warnings.append(f"Invalid runs list in {judge_path}")
            continue
        success_path = success_file_for(eval_dir)
        success_scores, parse_warnings = parse_success_file(success_path)
        warnings.extend(parse_warnings)
        eval_index = index_eval_summary(eval_dir)

        judge_index = {(run.get("agent"), run.get("task_id")): (idx, run) for idx, run in enumerate(runs)}
        expected_tasks = [
            task_id
            for task_id, meta in classification.items()
            if meta["original_experiment_group"] == group
        ]
        for task_id in expected_tasks:
            seen_tasks.add(task_id)
            for agent in METHOD_ORDER:
                if (agent, task_id) not in judge_index:
                    warnings.append(f"Missing judge run for {group}: {agent}__{task_id}")
                    continue
                idx, run = judge_index[(agent, task_id)]
                run_name = f"{agent}__{task_id}"
                utility = success_scores.get(run_name)
                if utility is None:
                    summary_score = eval_index.get(run_name, {}).get("task_successful")
                    if isinstance(summary_score, (int, float)):
                        utility = float(summary_score)
                        warnings.append(f"Using eval_summary task_successful for missing manual success score: {run_name}")
                    else:
                        utility = 0.0
                        warnings.append(f"Missing utility score; defaulting to 0: {run_name}")
                success = utility == 1.0

                attacker = run.get("attacker_judge")
                attacker_ok = isinstance(attacker, dict) and attacker.get("status") == "ok"
                if not attacker_ok:
                    warnings.append(f"Privacy judge not ok for {run_name}: {attacker.get('status') if isinstance(attacker, dict) else 'missing'}")
                metrics = attacker.get("metrics", {}) if attacker_ok else {}
                weighted = metrics.get("weighted_seeded_leakage")
                max_weighted = metrics.get("total_applicable_weight")
                run_wslr = metrics.get("wslr") if success and attacker_ok else None
                run_sps = (1.0 - float(run_wslr)) if isinstance(run_wslr, (int, float)) else None

                meta = classification[task_id]
                run_dir = Path(run.get("run_dir") or eval_dir / run_name)
                if not run_dir.is_absolute():
                    run_dir = (eval_dir / run_dir).resolve()
                screenshot_dir = run.get("screenshot_dir") or str(run_dir / "screenshots")
                raw_result_path = run_dir / "run_summary.json"
                if not raw_result_path.is_file():
                    raw_result_path = run_dir / "run.log"

                task_rows.append({
                    "task_id": task_id,
                    "app_name": run.get("app_name") or meta["app_name"],
                    "paper_category": meta["paper_category"],
                    "method": METHOD_LABELS.get(agent, agent),
                    "utility_score": fmt_num(utility),
                    "success": "1" if success else "0",
                    "weighted_leakage": fmt_num(weighted),
                    "max_weighted_leakage": fmt_num(max_weighted),
                    "wslr": fmt_num(run_wslr),
                    "sps": fmt_num(run_sps),
                    "leaked_seeded_items": json.dumps(leaked_items(attacker), ensure_ascii=False),
                    "raw_result_path": str(raw_result_path),
                    "task_success_record_path": str(success_path),
                    "screenshot_or_trajectory_path": str(screenshot_dir),
                    "judge_output_path": f"{judge_path}#runs[{idx}]",
                    "_method_id": agent,
                    "_utility_float": float(utility),
                    "_wslr_float": float(run_wslr) if isinstance(run_wslr, (int, float)) else None,
                    "_max_float": float(max_weighted) if isinstance(max_weighted, (int, float)) else None,
                    "_weighted_float": float(weighted) if isinstance(weighted, (int, float)) else None,
                    "_original_experiment_group": group,
                })

    unclassified = [
        row for task_id, row in classification.items()
        if task_id not in seen_tasks
    ]
    return task_rows, warnings, unclassified


def aggregate(task_rows: list[dict[str, Any]], category: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        if category is not None and row["paper_category"] != category:
            continue
        key = (row["paper_category"] if category is None else "Overall", row["_method_id"])
        groups[key].append(row)

    category_keys = CATEGORY_ORDER if category is None else ["Overall"]
    for cat in category_keys:
        for method_id in METHOD_ORDER:
            group_rows = groups.get((cat, method_id), [])
            if not group_rows:
                continue
            utility_values = [row["_utility_float"] for row in group_rows]
            success_rows = [row for row in group_rows if row["success"] == "1"]
            privacy_values = [row["_wslr_float"] for row in success_rows if row["_wslr_float"] is not None]
            privacy_count = len(privacy_values)
            wslr = (sum(privacy_values) / len(privacy_values)) if privacy_values else None
            notes = []
            if not success_rows:
                notes.append("No successful runs; WSLR/SPS N/A.")
            elif privacy_count < len(success_rows):
                notes.append(f"{len(success_rows) - privacy_count} successful run(s) missing judge metrics.")
            task_ids = {row["task_id"] for row in group_rows}
            rows.append({
                "paper_category": cat,
                "method": METHOD_LABELS[method_id],
                "n_tasks": len(task_ids),
                "n_runs": len(group_rows),
                "success_count": len(success_rows),
                "mean_utility": fmt_num(sum(utility_values) / len(utility_values)),
                "success_conditioned_wslr": fmt_num(wslr),
                "sps": fmt_num(1.0 - wslr if isinstance(wslr, float) else None),
                "notes": " ".join(notes),
            })
    return rows


def overall(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        by_method[row["_method_id"]].append(row)
    for method_id in METHOD_ORDER:
        group_rows = by_method.get(method_id, [])
        if not group_rows:
            continue
        success_rows = [row for row in group_rows if row["success"] == "1"]
        privacy_values = [row["_wslr_float"] for row in success_rows if row["_wslr_float"] is not None]
        wslr = (sum(privacy_values) / len(privacy_values)) if privacy_values else None
        utility_values = [row["_utility_float"] for row in group_rows]
        rows.append({
            "method": METHOD_LABELS[method_id],
            "n_tasks_total": len({row["task_id"] for row in group_rows}),
            "n_runs_total": len(group_rows),
            "success_count_total": len(success_rows),
            "mean_utility_total": fmt_num(sum(utility_values) / len(utility_values)),
            "success_conditioned_wslr_total": fmt_num(wslr),
            "sps_total": fmt_num(1.0 - wslr if isinstance(wslr, float) else None),
        })
    return rows


def residual_summary(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in task_rows:
        if row["method"] != "Full CAPED" or row["success"] != "1":
            continue
        for item in json.loads(row["leaked_seeded_items"] or "[]"):
            key = (row["paper_category"], item.get("category") or "", item.get("id") or "")
            entry = counts.setdefault(key, {
                "paper_category": row["paper_category"],
                "privacy_surface": item.get("category") or "",
                "seeded_item_id": item.get("id") or "",
                "count": 0,
                "tasks": set(),
            })
            entry["count"] += 1
            entry["tasks"].add(row["task_id"])
    rows = []
    for entry in counts.values():
        rows.append({
            "paper_category": entry["paper_category"],
            "privacy_surface": entry["privacy_surface"],
            "seeded_item_id": entry["seeded_item_id"],
            "count": entry["count"],
            "tasks": ", ".join(sorted(entry["tasks"])),
        })
    return sorted(rows, key=lambda r: (-r["count"], r["paper_category"], r["privacy_surface"], r["seeded_item_id"]))


def narrative(classification_rows: list[dict[str, str]], aggregate_rows: list[dict[str, Any]], overall_rows: list[dict[str, Any]], residual_rows: list[dict[str, Any]]) -> str:
    apps_by_cat: dict[str, set[str]] = defaultdict(set)
    tasks_by_cat: dict[str, list[str]] = defaultdict(list)
    for row in classification_rows:
        apps_by_cat[row["paper_category"]].add(row["app_name"])
        tasks_by_cat[row["paper_category"]].append(row["task_id"])

    full_overall = next((row for row in overall_rows if row["method"] == "Full CAPED"), None)
    full_by_cat = [row for row in aggregate_rows if row["method"] == "Full CAPED"]
    residual_top = residual_rows[:5]
    lines = [
        "# Paper Privacy Evaluation Summary",
        "",
        "## Result Directories",
    ]
    for group, path_text in DEFAULT_EXPERIMENTS.items():
        lines.append(f"- {group}: `{repo_root() / path_text}`")
    lines.extend([
        "",
        "## Category Composition",
    ])
    for cat in CATEGORY_ORDER:
        lines.append(f"- {cat}: apps = {', '.join(sorted(apps_by_cat[cat]))}; tasks = {', '.join(tasks_by_cat[cat])}.")

    lines.extend([
        "",
        "## Paper-Ready Aggregate",
        markdown_table(aggregate_rows, aggregate_fields),
        "",
        "## Overall Summary",
        markdown_table(overall_rows, overall_fields),
        "",
        "## Full CAPED Residual Leakage",
        markdown_table(residual_rows, residual_fields),
        "",
        "## Warnings",
        "\n".join(f"- {warning}" for warning in warnings) if warnings else "_No warnings._",
        "",
        "## Generated Files",
        f"- `{out_dir / 'task_classification.csv'}`",
        f"- `{out_dir / 'task_level_raw_stats.csv'}`",
        f"- `{out_dir / 'paper_ready_aggregate.csv'}`",
        f"- `{out_dir / 'overall_summary.csv'}`",
        f"- `{out_dir / 'full_caped_residual_leakage.csv'}`",
        f"- `{out_dir / 'paper_privacy_tables.json'}`",
        f"- `{out_dir / 'run_metadata.json'}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")

    if warnings:
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Wrote paper privacy tables to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
