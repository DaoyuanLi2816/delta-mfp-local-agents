"""Repair re-evaluation at N=3 replay (camera-ready Priority-2 experiment).

The accepted paper evaluated repairs at N=1, which the reviewers correctly
flagged as too noisy. This script re-runs the SAME repair subset (the same
persistent and soft trace-ids, the same 10 methods, the same step budgets and
seed scheme) at N=3 replay so each cell has a tighter Wilson interval.

Same trace-ids as the accepted run:
  persistent (5): qwen2.5:7b tool_compiler calendar_hard_02 seeds 50000..50004
  soft (4):       seeds 200000, 200008, 200009, 200010
The accepted run only evaluated 3/10 methods on persistent seed 50004 (the run
was budget-truncated); this N=3 rerun completes all 10 methods on all 9 traces,
which is a strict superset on the SAME trace-ids (documented in the report).

Outputs:
  data/processed/repair_results_N3.csv   (one row per trace x method, with per-seed successes)
  data/processed/repair_summary_N3.csv   (per method x class: mean success + Wilson CI)

Does NOT regenerate any traces. Reads the existing fault trace JSONLs.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from analysis.metrics import mean, write_csv, wilson_ci  # noqa: E402
from benchmarks.local_task_suite import load_local_tasks  # noqa: E402
from run_local_llm_suite import (  # noqa: E402
    max_steps_for,
    repair_for_method,
    replay_prefix,
)

PROCESSED = ROOT / "data" / "processed"
TRACES = ROOT / "data" / "traces"
PERSISTENT_JSONL = TRACES / "fault_injected_traces.jsonl"
SOFT_JSONL = TRACES / "soft_fault_traces.jsonl"
OUT_CSV = PROCESSED / "repair_results_N3.csv"
SUMMARY_CSV = PROCESSED / "repair_summary_N3.csv"

REPLAY_N = 3
MAX_REPLAY_STEPS_FULL = 12
MAX_REPLAY_STEPS_PREFIX = 8

REPAIR_METHODS = [
    "none", "generic_retry", "prompted_reflection", "predicted_repair",
    "oracle_repair", "clarifygate", "preconditiongate", "evidencegate",
    "rollbackretry", "boundaryshield",
]
METHOD_DISPLAY = {
    "none": "None", "generic_retry": "Retry", "prompted_reflection": "Reflection",
    "predicted_repair": "Predicted", "oracle_repair": "Oracle",
    "clarifygate": "ClarifyGate", "preconditiongate": "PreconditionGate",
    "evidencegate": "EvidenceGate", "rollbackretry": "RollbackRetry",
    "boundaryshield": "BoundaryShield",
}

# Exact trace-ids used by the accepted (N=1) repair run.
PERSISTENT_IDS: Set[str] = {
    "qwen2.5:7b::tool_compiler::calendar_hard_02::50000::wrong_id",
    "qwen2.5:7b::tool_compiler::calendar_hard_02::50001::stale_memory",
    "qwen2.5:7b::tool_compiler::calendar_hard_02::50002::precondition_skip",
    "qwen2.5:7b::tool_compiler::calendar_hard_02::50003::date_ambiguity",
    "qwen2.5:7b::tool_compiler::calendar_hard_02::50004::wrong_id",
}
SOFT_IDS: Set[str] = {
    "llama3.1:8b::tool_compiler::calendar_medium_00::200000::soft_date_ambiguity",
    "qwen2.5:7b::tool_compiler::inventory_medium_00::200008::soft_untrusted_instruction",
    "qwen2.5:7b::tool_compiler::inventory_medium_00::200009::soft_wrong_id",
    "qwen2.5:7b::tool_compiler::calendar_hard_03::200010::soft_date_ambiguity",
}

FIELDS = [
    "trace_id", "model", "interface", "task_id", "family", "fault_type",
    "fault_class", "method", "method_display", "repair", "prefix_used",
    "replay_n", "n_success", "success_rate", "success_ci", "per_seed_success",
    "residual_failure", "extra_model_calls", "extra_tool_calls", "elapsed_sec",
]


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def trace_id_for(t: Dict) -> str:
    return f"{t['model']}::{t['interface']}::{t['task_id']}::{t['seed']}::{t.get('fault_type')}"


def run_class(label: str, jsonl: Path, allow: Set[str], base_seed_prefix: int,
              rows: List[Dict]) -> None:
    tasks_by_id = {task.task_id: task for task in load_local_tasks()}
    traces = [t for t in load_jsonl(jsonl) if trace_id_for(t) in allow]
    # Deterministic order by seed for reproducibility.
    traces.sort(key=lambda t: int(t["seed"]))
    print(f"[{label}] {len(traces)}/{len(allow)} target traces found", flush=True)
    for t_idx, trace in enumerate(traces, 1):
        task = tasks_by_id.get(trace.get("task_id"))
        if task is None:
            print(f"[{label}] SKIP missing task {trace.get('task_id')}", flush=True)
            continue
        kf = int(trace.get("fault_prefix_index"))
        tid = trace_id_for(trace)
        for i, method in enumerate(REPAIR_METHODS):
            repair = repair_for_method(method, trace)
            rollback_to = None
            if method == "rollbackretry" or repair == "rollbackretry":
                rollback_to = max(0, kf - 1)
            if method == "generic_retry":
                prefix, repair = 0, None
            elif method == "prompted_reflection":
                prefix, repair = 0, "preconditiongate"
            else:
                prefix = kf
            seeds = [base_seed_prefix + t_idx * 10_000 + i * 100 + j for j in range(REPLAY_N)]
            if method in ("generic_retry", "prompted_reflection"):
                step_cap = min(max_steps_for(task), MAX_REPLAY_STEPS_FULL)
            else:
                step_cap = min(max_steps_for(task), MAX_REPLAY_STEPS_PREFIX)
            t0 = time.time()
            result = replay_prefix(
                task, trace, prefix, trace["model"], trace["interface"], seeds,
                repair=repair, max_steps=step_cap, rollback_to=rollback_to,
            )
            elapsed = time.time() - t0
            per_seed = [int(bool(r.get("final_success"))) for r in result["runs"]]
            n_success = sum(per_seed)
            ci = wilson_ci(n_success, REPLAY_N)
            row = {
                "trace_id": tid, "model": trace["model"], "interface": trace["interface"],
                "task_id": trace["task_id"], "family": trace["family"],
                "fault_type": trace.get("fault_type"), "fault_class": label,
                "method": method, "method_display": METHOD_DISPLAY[method],
                "repair": repair or "none", "prefix_used": prefix, "replay_n": REPLAY_N,
                "n_success": n_success,
                "success_rate": f"{result['success_rate']:.3f}",
                "success_ci": f"[{ci['lo']:.2f},{ci['hi']:.2f}]",
                "per_seed_success": json.dumps(per_seed),
                "residual_failure": f"{1.0 - result['success_rate']:.3f}",
                "extra_model_calls": REPLAY_N,
                "extra_tool_calls": f"{mean([r.get('tool_calls', 0) for r in result['runs']]):.2f}",
                "elapsed_sec": f"{elapsed:.1f}",
            }
            rows.append(row)
            write_csv(OUT_CSV, rows)
            print(f"[{label}] {tid[-34:]} {method:<20s} sr={row['success_rate']} "
                  f"ci={row['success_ci']} t={elapsed:.0f}s", flush=True)


def build_summary(rows: List[Dict]) -> None:
    # Per (class, method): mean of per-trace success rates + Wilson CI on pooled trials.
    agg: Dict = defaultdict(lambda: {"rates": [], "succ": 0, "trials": 0, "tool": [], "elapsed": []})
    none_rate = {}
    for r in rows:
        key = (r["fault_class"], r["method"])
        a = agg[key]
        a["rates"].append(float(r["success_rate"]))
        a["succ"] += int(r["n_success"])
        a["trials"] += int(r["replay_n"])
        a["tool"].append(float(r["extra_tool_calls"]))
        a["elapsed"].append(float(r["elapsed_sec"]))
    for r in rows:
        if r["method"] == "none":
            none_rate.setdefault(r["fault_class"], []).append(float(r["success_rate"]))
    none_mean = {c: (mean(v) if v else 0.0) for c, v in none_rate.items()}

    out = []
    for (cls, method), a in sorted(agg.items()):
        n_traces = len(a["rates"])
        mean_sr = mean(a["rates"]) if a["rates"] else 0.0
        ci = wilson_ci(a["succ"], a["trials"]) if a["trials"] else {"lo": 0.0, "hi": 0.0}
        lift = mean_sr - none_mean.get(cls, 0.0)
        out.append({
            "fault_class": cls, "method": method, "method_display": METHOD_DISPLAY[method],
            "n_traces": n_traces, "replay_n": REPLAY_N, "n_trials": a["trials"],
            "n_success_pooled": a["succ"],
            "mean_success_rate": f"{mean_sr:.3f}",
            "wilson_ci": f"[{ci['lo']:.2f},{ci['hi']:.2f}]",
            "lift_over_none": f"{lift:+.3f}",
            "mean_extra_tool_calls": f"{mean(a['tool']):.2f}" if a["tool"] else "0.00",
            "mean_elapsed_sec": f"{mean(a['elapsed']):.1f}" if a["elapsed"] else "0.0",
        })
    # order: persistent block then soft block, methods in canonical order
    order = {m: i for i, m in enumerate(REPAIR_METHODS)}
    out.sort(key=lambda r: (r["fault_class"], order[r["method"]]))
    write_csv(SUMMARY_CSV, out)
    print(f"wrote {SUMMARY_CSV} ({len(out)} method-class rows)", flush=True)


def main() -> None:
    rows: List[Dict] = []
    started = time.time()
    run_class("persistent", PERSISTENT_JSONL, PERSISTENT_IDS, 700_000, rows)
    run_class("soft", SOFT_JSONL, SOFT_IDS, 1_200_000, rows)
    build_summary(rows)
    print(f"done. {len(rows)} cells; total elapsed={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
