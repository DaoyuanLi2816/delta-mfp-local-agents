"""Extension experiment runner.

This script runs three additional experiments without touching the existing
calibration / natural-failure data:

1. Soft fault injection on existing successful or near-successful traces
   (records to data/processed/soft_fault_results.csv and
   data/traces/soft_fault_traces.jsonl).
2. Repair re-evaluation on persistent fault traces using a reduced method set
   that fits a single-GPU runtime budget.
3. Repair evaluation on soft fault traces.

Run order is incremental: each loop writes results immediately so a long run
can be killed at any time and the partial outputs are still usable.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from analysis.metrics import mean, write_csv, wilson_ci  # noqa: E402
from benchmarks.local_task_suite import load_local_tasks  # noqa: E402
from run_local_llm_suite import (  # noqa: E402
    SOFT_FAULTS,
    classify_delta_mfp,
    collect_success_or_near_success,
    extract_router_features,
    fault_row,
    max_steps_for,
    predict_repair,
    read_csv,
    repair_for_method,
    replay_prefix,
    run_episode,
    write_jsonl,
)


PROCESSED = ROOT / "data" / "processed"
TRACES = ROOT / "data" / "traces"


# Reduced repair method set for the budgeted re-eval.
REPAIR_METHODS = [
    "none",
    "generic_retry",
    "prompted_reflection",
    "predicted_repair",
    "oracle_repair",
    "clarifygate",
    "preconditiongate",
    "evidencegate",
    "rollbackretry",
    "boundaryshield",
]


def _short_replay_prefix(*args, max_steps: Optional[int] = None, **kwargs):
    return replay_prefix(*args, max_steps=max_steps, **kwargs)


def run_repairs(
    label: str,
    trace_path: Path,
    out_csv: Path,
    base_seed_prefix: int,
    replay_n: int = 1,
    max_replay_steps_full: int = 12,
    max_replay_steps_prefix: int = 8,
    max_traces: Optional[int] = None,
) -> List[Dict]:
    tasks_by_id = {task.task_id: task for task in load_local_tasks()}
    rows = read_csv(out_csv)
    seen = {(r["trace_id"], r["method"]) for r in rows}
    new_traces = 0
    if not trace_path.exists():
        return rows
    for line in trace_path.open("r", encoding="utf-8"):
        try:
            trace = json.loads(line)
        except json.JSONDecodeError:
            continue
        task = tasks_by_id.get(trace.get("task_id"))
        if task is None:
            continue
        kf = trace.get("fault_prefix_index")
        if kf is None:
            continue
        kf = int(kf)
        trace_id = (
            f"{trace['model']}::{trace['interface']}::{trace['task_id']}"
            f"::{trace['seed']}::{trace['fault_type']}"
        )
        already = sum(1 for m in REPAIR_METHODS if (trace_id, m) in seen)
        if already >= len(REPAIR_METHODS):
            continue
        new_traces += 1
        if max_traces is not None and new_traces > max_traces:
            break
        for i, method in enumerate(REPAIR_METHODS):
            if (trace_id, method) in seen:
                continue
            repair = repair_for_method(method, trace)
            rollback_to = None
            if method == "rollbackretry" or repair == "rollbackretry":
                rollback_to = max(0, kf - 1)
            if method == "generic_retry":
                prefix = 0
                repair = None
            elif method == "prompted_reflection":
                prefix = 0
                repair = "preconditiongate"
            else:
                prefix = kf
            seeds = [base_seed_prefix + new_traces * 10_000 + i * 100 + j
                     for j in range(replay_n)]
            t0 = time.time()
            # Restart-style methods use up to the task's natural step budget so
            # they can plausibly complete the workflow. Prefix-level repairs
            # only need to finish the remaining steps and are capped lower.
            if method in ("generic_retry", "prompted_reflection"):
                step_cap = min(max_steps_for(task), max_replay_steps_full)
            else:
                step_cap = min(max_steps_for(task), max_replay_steps_prefix)
            result = replay_prefix(
                task, trace, prefix, trace["model"], trace["interface"], seeds,
                repair=repair, max_steps=step_cap,
                rollback_to=rollback_to,
            )
            elapsed = time.time() - t0
            ci = wilson_ci(round(result["success_rate"] * replay_n), replay_n)
            row = {
                "trace_id": trace_id,
                "model": trace["model"],
                "interface": trace["interface"],
                "task_id": trace["task_id"],
                "family": trace["family"],
                "fault_type": trace.get("fault_type"),
                "fault_class": "soft" if trace.get("fault_type", "").startswith("soft_") else "persistent",
                "method": method,
                "repair": repair or "none",
                "success_rate": f"{result['success_rate']:.3f}",
                "success_ci": f"[{ci['lo']:.2f},{ci['hi']:.2f}]",
                "residual_failure": f"{1.0 - result['success_rate']:.3f}",
                "extra_model_calls": replay_n,
                "extra_tool_calls": f"{mean([r.get('tool_calls', 0) for r in result['runs']]):.2f}",
                "elapsed_sec": f"{elapsed:.1f}",
            }
            rows.append(row)
            seen.add((trace_id, method))
            write_csv(out_csv, rows)
            sys.stdout.write(
                f"[{label}] {trace_id[-30:]} {method:<22s} "
                f"sr={row['success_rate']} t={elapsed:.0f}s\n"
            )
            sys.stdout.flush()
    return rows


def run_soft_fault_injection(
    target: int = 25,
    replay_n: int = 1,
    p_fail: float = 0.6,
    delta: float = 0.3,
) -> List[Dict]:
    cal_path = PROCESSED / "calibration_results.csv"
    calibration = read_csv(cal_path)
    base_traces = collect_success_or_near_success(
        calibration, target=max(target * 2, 30), seeds_per_cell=2,
    )
    out_csv = PROCESSED / "soft_fault_results.csv"
    out_jsonl = TRACES / "soft_fault_traces.jsonl"
    rows = read_csv(out_csv)
    existing = {r.get("trace_id") for r in rows}
    soft_types = sorted(SOFT_FAULTS)
    injected = []
    if out_jsonl.exists():
        for line in out_jsonl.open("r", encoding="utf-8"):
            try:
                injected.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    tasks_by_id = {task.task_id: task for task in load_local_tasks()}
    idx = 0
    for base in base_traces:
        if len(rows) >= target:
            break
        task = tasks_by_id[base["task_id"]]
        for fault in soft_types:
            seed = 80_000 + idx
            idx += 1
            t0 = time.time()
            try:
                trace = run_episode(
                    task,
                    base["model"],
                    base["interface"],
                    seed=seed,
                    max_steps=max_steps_for(task),
                    fault=fault,
                    inject_after_step=0,
                )
            except Exception as exc:
                sys.stdout.write(f"[soft] error {exc}\n")
                continue
            if trace["final_success"]:
                continue
            trace_id = (
                f"{trace['model']}::{trace['interface']}::{trace['task_id']}"
                f"::{trace['seed']}::{trace['fault_type']}"
            )
            if trace_id in existing:
                continue
            kf = int(trace.get("fault_prefix_index") or 1)
            candidates = sorted({0, kf, max(1, kf - 1), kf + 1
                                  if kf + 1 < len(trace["snapshots"]) else 0})
            mfp = classify_delta_mfp(
                task, trace, base["model"], base["interface"],
                replay_n, p_fail, delta,
                base_seed=90_000 + idx * 1000,
                candidate_prefixes=candidates,
            )
            row = fault_row(trace, mfp)
            row["fault_class"] = "soft"
            rows.append(row)
            injected.append(trace)
            existing.add(trace_id)
            write_jsonl(out_jsonl, injected)
            write_csv(out_csv, rows)
            elapsed = time.time() - t0
            sys.stdout.write(
                f"[soft-inject] {trace_id[-40:]} cat={mfp['mfp_category']} "
                f"t={elapsed:.0f}s\n"
            )
            sys.stdout.flush()
            if len(rows) >= target:
                break
    return rows


def run_persistent_repairs(replay_n: int = 1, max_traces: Optional[int] = None) -> List[Dict]:
    return run_repairs(
        label="repair-persistent",
        trace_path=TRACES / "fault_injected_traces.jsonl",
        out_csv=PROCESSED / "repair_results.csv",
        base_seed_prefix=70_000,
        replay_n=replay_n,
        max_traces=max_traces,
    )


def run_soft_repairs(replay_n: int = 1, max_traces: Optional[int] = None) -> List[Dict]:
    return run_repairs(
        label="repair-soft",
        trace_path=TRACES / "soft_fault_traces.jsonl",
        out_csv=PROCESSED / "repair_results.csv",
        base_seed_prefix=120_000,
        replay_n=replay_n,
        max_traces=max_traces,
    )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-target", type=int, default=15)
    parser.add_argument("--soft-replay-n", type=int, default=1)
    parser.add_argument("--repair-replay-n", type=int, default=1)
    parser.add_argument("--persistent-trace-cap", type=int, default=20)
    parser.add_argument("--soft-trace-cap", type=int, default=15)
    parser.add_argument("--phase", choices=["soft_inject", "persistent_repairs",
                                            "soft_repairs", "all"], default="all")
    args = parser.parse_args()

    if args.phase in {"persistent_repairs", "all"}:
        run_persistent_repairs(args.repair_replay_n, max_traces=args.persistent_trace_cap)
    if args.phase in {"soft_inject", "all"}:
        run_soft_fault_injection(target=args.soft_target,
                                  replay_n=args.soft_replay_n)
    if args.phase in {"soft_repairs", "all"}:
        run_soft_repairs(args.repair_replay_n, max_traces=args.soft_trace_cap)


if __name__ == "__main__":
    main()
