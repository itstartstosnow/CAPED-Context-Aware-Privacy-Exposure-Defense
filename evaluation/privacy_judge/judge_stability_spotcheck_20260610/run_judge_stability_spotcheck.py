#!/usr/bin/env python3
"""Random spot-check for privacy judge repeated-run stability.

This script intentionally reuses evaluation/privacy_judge/run_privacy_judge.py
for prompt construction, model calls, seeded mapping validation, and WSLR
computation. The spot-check code only handles sampling, repeat orchestration,
and stability analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
SPOTCHECK_DIR = SCRIPT_PATH.parent
PRIVACY_JUDGE_DIR = SPOTCHECK_DIR.parent
REPO_ROOT = PRIVACY_JUDGE_DIR.parent.parent
ANDROID_WORLD_DIR = REPO_ROOT / "Mobile-Agent-v3" / "android_world_v3"
DIAGNOSTIC_CSV = (
    ANDROID_WORLD_DIR
    / "reviewer1_feedback_diagnostics"
    / "privacy_metric_diagnostics_20260610"
    / "per_run_privacy_metrics.csv"
)

sys.path.insert(0, str(PRIVACY_JUDGE_DIR))

from judge_client import JudgeClient  # noqa: E402
from run_privacy_judge import (  # noqa: E402
    JUDGE_MODEL,
    _check_env,
    _judge_single_with_retries,
)


SAMPLE_SEED = 20260610
SAMPLE_SIZE = 28
REPEAT_COUNT = 3
JUDGE_API = "responses"
MANIFEST_PATH = SPOTCHECK_DIR / "sample_manifest.csv"
REPEAT_RUNS_PATH = SPOTCHECK_DIR / "repeat_runs.json"
VARIANCE_PATH = SPOTCHECK_DIR / "judge_repeat_variance.csv"
ORDERING_PATH = SPOTCHECK_DIR / "method_ordering_by_repeat.csv"
SUMMARY_PATH = SPOTCHECK_DIR / "judge_repeat_summary.md"
RAW_REPEATS_DIR = SPOTCHECK_DIR / "raw_repeats"

EVAL_RESULT_PATHS = [
    ANDROID_WORLD_DIR / "eval_mock_all_updated_260511_1838" / "judge_results.json",
    ANDROID_WORLD_DIR / "eval_awp_all_updated_260512_1744" / "judge_results.json",
    ANDROID_WORLD_DIR / "eval_expandOther_all_20260604_full_expand" / "judge_results.json",
    ANDROID_WORLD_DIR / "eval_expandGoogle_all_20260604_full_expand" / "judge_results.json",
]

METHOD_LABELS = {
    "guiowl": "No Protection",
    "text_only": "Text-Only",
    "no_elem_verify": "CAPED w/o Element Verification",
    "no_task_extract": "CAPED w/o Task Extraction",
    "privacy": "Full CAPED",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_diagnostic_rows() -> dict[str, dict[str, str]]:
    if not DIAGNOSTIC_CSV.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with DIAGNOSTIC_CSV.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            run_name = Path(row.get("run_dir", "")).name
            artifact_suite = row.get("artifact_suite", "")
            key = f"{artifact_suite}::{run_name}"
            rows[key] = row
    return rows


def infer_artifact_suite(result_path: Path) -> str:
    name = result_path.parent.name
    if name.startswith("eval_mock"):
        return "mock"
    if name.startswith("eval_awp"):
        return "awp"
    if name.startswith("eval_expandOther"):
        return "expandOther"
    if name.startswith("eval_expandGoogle"):
        return "expandGoogle"
    return name


def infer_paper_suite(artifact_suite: str) -> str:
    if artifact_suite == "mock":
        return "Mock apps"
    if artifact_suite in {"awp", "expandOther"}:
        return "AndroidWorld-style local apps"
    if artifact_suite == "expandGoogle":
        return "Real connected apps"
    return artifact_suite


def ok_attacker_run(run: dict[str, Any]) -> bool:
    attacker = run.get("attacker_judge")
    return run.get("status") == "ok" and isinstance(attacker, dict) and attacker.get("status") == "ok"


def load_judged_runs() -> list[dict[str, Any]]:
    diagnostics = load_diagnostic_rows()
    candidates: list[dict[str, Any]] = []
    for result_path in EVAL_RESULT_PATHS:
        payload = read_json(result_path)
        artifact_suite = infer_artifact_suite(result_path)
        paper_suite = infer_paper_suite(artifact_suite)
        for run in payload.get("runs", []):
            if not isinstance(run, dict) or not ok_attacker_run(run):
                continue
            method = str(run.get("agent", ""))
            task_id = str(run.get("task_id", ""))
            run_name = f"{method}__{task_id}"
            diag = diagnostics.get(f"{artifact_suite}::{run_name}", {})
            candidates.append({
                "paper_suite": paper_suite,
                "artifact_suite": artifact_suite,
                "eval_dir": str(result_path.parent),
                "run_name": run_name,
                "task_id": task_id,
                "app_name": str(run.get("app_name", "")),
                "task_title": str(run.get("task_title", "")),
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "fully_successful": diag.get("fully_successful", ""),
                "utility": diag.get("utility", ""),
                "original_wslr": str((run.get("attacker_judge", {}).get("metrics") or {}).get("wslr", "")),
                "original_leaked_item_ids": ";".join(
                    item.get("id", "")
                    for item in (run.get("attacker_judge", {}).get("seeded_mapping", {}).get("items") or [])
                    if item.get("leaked") and not item.get("task_required")
                ),
                "run_dir": str(run.get("run_dir", "")),
                "screenshot_dir": str(run.get("screenshot_dir", "")),
                "judge_results_json": str(result_path),
                "num_images": str(run.get("num_images", "")),
            })
    return candidates


def sample_runs(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(SAMPLE_SEED)
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        strata[(row["paper_suite"], row["method"])].append(row)

    for key in sorted(strata):
        rows = sorted(strata[key], key=lambda r: (r["artifact_suite"], r["run_name"]))
        chosen = rng.choice(rows)
        selected.append({**chosen, "selection_reason": "stratum_seed"})
        selected_keys.add(f"{chosen['artifact_suite']}::{chosen['run_name']}")

    remaining = [
        row for row in candidates
        if f"{row['artifact_suite']}::{row['run_name']}" not in selected_keys
    ]
    rng.shuffle(remaining)
    for row in remaining:
        if len(selected) >= SAMPLE_SIZE:
            break
        selected.append({**row, "selection_reason": "random_fill"})
        selected_keys.add(f"{row['artifact_suite']}::{row['run_name']}")

    selected.sort(key=lambda r: (r["paper_suite"], r["method"], r["artifact_suite"], r["run_name"]))
    for index, row in enumerate(selected, 1):
        row["sample_id"] = f"S{index:02d}"
        row["sample_seed"] = str(SAMPLE_SEED)
    return selected


def write_manifest(rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "sample_seed",
        "paper_suite",
        "artifact_suite",
        "eval_dir",
        "run_name",
        "task_id",
        "app_name",
        "task_title",
        "method",
        "method_label",
        "fully_successful",
        "utility",
        "original_wslr",
        "original_leaked_item_ids",
        "run_dir",
        "screenshot_dir",
        "judge_results_json",
        "num_images",
        "selection_reason",
    ]
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def raw_output_path(sample: dict[str, str], repeat_index: int) -> Path:
    safe_name = f"{sample['sample_id']}__{sample['artifact_suite']}__{sample['run_name']}.json"
    return RAW_REPEATS_DIR / f"repeat_{repeat_index}" / safe_name


def extract_repeat_record(
    sample: dict[str, str],
    repeat_index: int,
    result: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    attacker = result.get("attacker_judge") if isinstance(result, dict) else {}
    mapping = attacker.get("seeded_mapping", {}) if isinstance(attacker, dict) else {}
    metrics = attacker.get("metrics", {}) if isinstance(attacker, dict) else {}
    items = mapping.get("items", []) if isinstance(mapping, dict) else []
    return {
        "sample_id": sample["sample_id"],
        "repeat_index": repeat_index,
        "task_id": sample["task_id"],
        "paper_suite": sample["paper_suite"],
        "artifact_suite": sample["artifact_suite"],
        "method": sample["method"],
        "run_name": sample["run_name"],
        "fully_successful": sample.get("fully_successful", ""),
        "judge_model": JUDGE_MODEL,
        "judge_api": JUDGE_API,
        "judge_config": {
            "config_dir": str(PRIVACY_JUDGE_DIR),
            "task_specs": "task_specs.json",
            "seeded_leakage_specs": "seeded_leakage_specs.json",
            "prompt_logic": "evaluation/privacy_judge/run_privacy_judge.py",
        },
        "status": result.get("status"),
        "attacker_status": attacker.get("status") if isinstance(attacker, dict) else None,
        "error": result.get("error"),
        "wslr": metrics.get("wslr") if isinstance(metrics, dict) else None,
        "weighted_seeded_leakage": metrics.get("weighted_seeded_leakage") if isinstance(metrics, dict) else None,
        "total_applicable_weight": metrics.get("total_applicable_weight") if isinstance(metrics, dict) else None,
        "mapped_items": [
            {
                "item_id": item.get("id"),
                "task_required": item.get("task_required"),
                "weight": item.get("weight"),
                "leaked": item.get("leaked"),
                "evidence": item.get("evidence"),
                "screenshot_indices": item.get("screenshot_indices"),
            }
            for item in items
            if isinstance(item, dict)
        ],
        "raw_output_path": str(out_path),
    }


def run_repeats() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST_PATH}. Run --sample first.")
    samples = read_manifest()
    base_url = _check_env()
    client = JudgeClient(model=JUDGE_MODEL, api=JUDGE_API)
    repeat_records: list[dict[str, Any]] = []

    for repeat_index in range(1, REPEAT_COUNT + 1):
        for sample_index, sample in enumerate(samples, 1):
            out_path = raw_output_path(sample, repeat_index)
            print(
                f"[repeat {repeat_index}/{REPEAT_COUNT} sample {sample_index}/{len(samples)}] "
                f"{sample['sample_id']} {sample['run_name']} ... ",
                end="",
                flush=True,
            )
            result = _judge_single_with_retries(
                sample["task_id"],
                Path(sample["screenshot_dir"]),
                PRIVACY_JUDGE_DIR,
                client,
                judge_mode="attacker",
            )
            result["agent"] = sample["method"]
            result["run_dir"] = sample["run_dir"]
            result["model"] = JUDGE_MODEL
            result["base_url"] = base_url
            result["timestamp_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            write_json(out_path, result)
            repeat_record = extract_repeat_record(sample, repeat_index, result, out_path)
            repeat_records.append(repeat_record)
            write_json(REPEAT_RUNS_PATH, {
                "sample_seed": SAMPLE_SEED,
                "sample_size": len(samples),
                "repeat_count": REPEAT_COUNT,
                "judge_model": JUDGE_MODEL,
                "judge_api": JUDGE_API,
                "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "runs": repeat_records,
            })
            status = repeat_record.get("attacker_status") or repeat_record.get("status")
            wslr = repeat_record.get("wslr")
            print(f"{status} WSLR={wslr}")


def bool_label(value: Any) -> bool:
    return bool(value)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stdev(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    return statistics.stdev(values)


def item_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("item_id")): item
        for item in record.get("mapped_items", [])
        if item.get("item_id") and not item.get("task_required")
    }


def compute_majority_wslr(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    maps = [item_map(record) for record in records]
    all_ids = sorted(set().union(*(set(m) for m in maps)))
    leaked_weight = 0.0
    total_weight = 0.0
    for item_id in all_ids:
        observed = [m[item_id] for m in maps if item_id in m]
        if not observed:
            continue
        weight = float(observed[0].get("weight") or 0.0)
        leaked_votes = sum(1 for item in observed if bool_label(item.get("leaked")))
        total_weight += weight
        if leaked_votes >= 2:
            leaked_weight += weight
    return leaked_weight / total_weight if total_weight else None


def analyze() -> None:
    payload = read_json(REPEAT_RUNS_PATH)
    records = payload.get("runs", [])
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sample[record["sample_id"]].append(record)

    variance_rows: list[dict[str, Any]] = []
    for sample_id, sample_records in sorted(by_sample.items()):
        sample_records.sort(key=lambda r: int(r["repeat_index"]))
        first = sample_records[0]
        if len(sample_records) != REPEAT_COUNT:
            status = "incomplete"
        elif any(r.get("attacker_status") != "ok" for r in sample_records):
            status = "has_error"
        else:
            status = "ok"

        repeat_maps = [item_map(record) for record in sample_records]
        all_item_ids = sorted(set().union(*(set(m) for m in repeat_maps)))
        changed_ids: list[str] = []
        for item_id in all_item_ids:
            labels = [m.get(item_id, {}).get("leaked", False) for m in repeat_maps]
            if len(set(bool_label(label) for label in labels)) > 1:
                changed_ids.append(item_id)
        wslrs = [
            float(record["wslr"])
            for record in sample_records
            if isinstance(record.get("wslr"), (int, float))
        ]
        wslr_mean = mean(wslrs)
        wslr_std = stdev(wslrs)
        wslr_range = (max(wslrs) - min(wslrs)) if wslrs else None
        wslr_mae = (
            mean([abs(value - wslr_mean) for value in wslrs])
            if wslrs and wslr_mean is not None
            else None
        )
        variance_rows.append({
            "sample_id": sample_id,
            "status": status,
            "paper_suite": first["paper_suite"],
            "artifact_suite": first["artifact_suite"],
            "method": first["method"],
            "run_name": first["run_name"],
            "task_id": first["task_id"],
            "fully_successful": first.get("fully_successful", ""),
            "incidental_item_count": len(all_item_ids),
            "flipped_item_count": len(changed_ids),
            "item_flip_rate": len(changed_ids) / len(all_item_ids) if all_item_ids else "",
            "wslr_repeat_1": sample_records[0].get("wslr") if len(sample_records) > 0 else "",
            "wslr_repeat_2": sample_records[1].get("wslr") if len(sample_records) > 1 else "",
            "wslr_repeat_3": sample_records[2].get("wslr") if len(sample_records) > 2 else "",
            "wslr_mean": wslr_mean,
            "wslr_std": wslr_std,
            "wslr_range": wslr_range,
            "wslr_mae_to_mean": wslr_mae,
            "majority_wslr": compute_majority_wslr(sample_records),
            "changed_item_ids": ";".join(changed_ids),
        })

    write_variance_csv(variance_rows)
    ordering_rows = compute_orderings(records, variance_rows)
    write_ordering_csv(ordering_rows)
    write_summary(variance_rows, ordering_rows)


def write_variance_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "status",
        "paper_suite",
        "artifact_suite",
        "method",
        "run_name",
        "task_id",
        "fully_successful",
        "incidental_item_count",
        "flipped_item_count",
        "item_flip_rate",
        "wslr_repeat_1",
        "wslr_repeat_2",
        "wslr_repeat_3",
        "wslr_mean",
        "wslr_std",
        "wslr_range",
        "wslr_mae_to_mean",
        "majority_wslr",
        "changed_item_ids",
    ]
    with VARIANCE_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ordering(values: dict[str, float]) -> str:
    return " > ".join(
        method
        for method, _ in sorted(values.items(), key=lambda item: (item[1], item[0]), reverse=True)
    )


def compute_orderings(
    records: list[dict[str, Any]],
    variance_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope_name, sample_filter in [
        ("all_sampled", lambda r: True),
        ("successful_only", lambda r: str(r.get("fully_successful")).lower() == "true"),
    ]:
        for aggregation in ["repeat_1", "repeat_2", "repeat_3"]:
            repeat_index = int(aggregation.rsplit("_", 1)[1])
            grouped: dict[str, list[float]] = defaultdict(list)
            for record in records:
                if int(record["repeat_index"]) != repeat_index or not sample_filter(record):
                    continue
                if isinstance(record.get("wslr"), (int, float)):
                    grouped[record["method"]].append(float(record["wslr"]))
            rows.append(ordering_row(scope_name, aggregation, grouped))

        grouped_majority: dict[str, list[float]] = defaultdict(list)
        for row in variance_rows:
            if not sample_filter(row):
                continue
            if isinstance(row.get("majority_wslr"), (int, float)):
                grouped_majority[row["method"]].append(float(row["majority_wslr"]))
        rows.append(ordering_row(scope_name, "majority_vote", grouped_majority))

    return rows


def ordering_row(scope_name: str, aggregation: str, grouped: dict[str, list[float]]) -> dict[str, Any]:
    means = {method: mean(values) for method, values in grouped.items() if values}
    return {
        "scope": scope_name,
        "aggregation": aggregation,
        "method_count": len(means),
        "method_mean_wslr_json": json.dumps(means, ensure_ascii=False, sort_keys=True),
        "method_ordering_by_mean_wslr_desc": ordering({k: v for k, v in means.items() if v is not None}),
    }


def write_ordering_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "scope",
        "aggregation",
        "method_count",
        "method_mean_wslr_json",
        "method_ordering_by_mean_wslr_desc",
    ]
    with ORDERING_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        return f"{value:.4f}"
    return str(value)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sum(int(row["incidental_item_count"]) for row in rows)
    flips = sum(int(row["flipped_item_count"]) for row in rows)
    stds = [float(row["wslr_std"]) for row in rows if isinstance(row.get("wslr_std"), (int, float))]
    ranges = [float(row["wslr_range"]) for row in rows if isinstance(row.get("wslr_range"), (int, float))]
    return {
        "trajectories": len(rows),
        "labels": labels,
        "item_flip_rate": flips / labels if labels else None,
        "mean_wslr_std": mean(stds),
        "max_wslr_range": max(ranges) if ranges else None,
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary(variance_rows: list[dict[str, Any]], ordering_rows: list[dict[str, Any]]) -> None:
    ok_rows = [row for row in variance_rows if row["status"] == "ok"]
    overall = aggregate_rows(ok_rows)

    by_suite_method: list[list[Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ok_rows:
        grouped[(row["paper_suite"], row["method"])].append(row)
    for (suite, method), rows in sorted(grouped.items()):
        agg = aggregate_rows(rows)
        by_suite_method.append([
            suite,
            method,
            agg["trajectories"],
            agg["labels"],
            agg["item_flip_rate"],
            agg["mean_wslr_std"],
            agg["max_wslr_range"],
        ])

    selection_counts = Counter((row["paper_suite"], row["method"]) for row in variance_rows)
    selection_rows = [
        [suite, method, count]
        for (suite, method), count in sorted(selection_counts.items())
    ]

    ordering_md_rows = [
        [
            row["scope"],
            row["aggregation"],
            row["method_ordering_by_mean_wslr_desc"],
            row["method_mean_wslr_json"],
        ]
        for row in ordering_rows
    ]

    summary = "\n\n".join([
        "# Judge Stability Spot-Check Summary",
        (
            f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z\n\n"
            f"Sample seed: `{SAMPLE_SEED}`\n\n"
            f"Sample size: `{SAMPLE_SIZE}` trajectories, `{REPEAT_COUNT}` repeats each\n\n"
            f"Judge model/API: `{JUDGE_MODEL}` / `{JUDGE_API}`\n\n"
            "Prompt construction, judge calls, seeded mapping validation, and WSLR "
            "computation reuse `evaluation/privacy_judge/run_privacy_judge.py`."
        ),
        "## Overall\n\n" + markdown_table(
            ["Scope", "# trajectories", "# incidental labels", "Item flip rate", "Mean WSLR std", "Max WSLR range"],
            [[
                "all ok repeats",
                overall["trajectories"],
                overall["labels"],
                overall["item_flip_rate"],
                overall["mean_wslr_std"],
                overall["max_wslr_range"],
            ]],
        ),
        "## Sample Coverage\n\n" + markdown_table(
            ["Suite", "Method", "# trajectories"],
            selection_rows,
        ),
        "## Suite And Method Breakdown\n\n" + markdown_table(
            ["Suite", "Method", "# trajectories", "# incidental labels", "Item flip rate", "Mean WSLR std", "Max WSLR range"],
            by_suite_method,
        ),
        "## Method Ordering\n\n" + markdown_table(
            ["Scope", "Aggregation", "Method ordering by mean WSLR", "Method mean WSLR JSON"],
            ordering_md_rows,
        ),
        (
            "## Interpretation Notes\n\n"
            "- This is a random spot-check over existing judged-ok trajectories, not a full re-judging of all benchmark runs.\n"
            "- Method ordering is computed only within the sampled spot-check set.\n"
            "- Human agreement is intentionally skipped for this run."
        ),
    ])
    SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")


def do_sample() -> None:
    candidates = load_judged_runs()
    if len(candidates) < SAMPLE_SIZE:
        raise SystemExit(f"Only {len(candidates)} judged-ok candidates available, need {SAMPLE_SIZE}")
    samples = sample_runs(candidates)
    write_manifest(samples)
    counts = Counter((row["paper_suite"], row["method"]) for row in samples)
    print(f"Wrote {MANIFEST_PATH}")
    print(f"Candidates: {len(candidates)}; sampled: {len(samples)}; seed: {SAMPLE_SEED}")
    for (suite, method), count in sorted(counts.items()):
        print(f"  {suite} / {method}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="Generate sample_manifest.csv")
    parser.add_argument("--run-repeats", action="store_true", help="Run judge repeats via OpenAI Responses API")
    parser.add_argument("--analyze", action="store_true", help="Analyze repeat_runs.json")
    parser.add_argument("--all", action="store_true", help="Run sample, repeats, and analysis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (args.sample or args.run_repeats or args.analyze or args.all):
        args.sample = True
    if args.all or args.sample:
        do_sample()
    if args.all or args.run_repeats:
        run_repeats()
    if args.all or args.analyze:
        analyze()
        print(f"Wrote {VARIANCE_PATH}")
        print(f"Wrote {ORDERING_PATH}")
        print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
