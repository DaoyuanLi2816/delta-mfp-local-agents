"""Counterfactual prefix replay and repair verification."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from benchmarks.diagnostic_env import DiagnosticEnv, TaskSpec
from replay.trace_logger import run_episode


JsonDict = Dict[str, Any]


def replay_from_prefix(
    task: TaskSpec,
    trace: JsonDict,
    prefix_index: int,
    seeds: List[int],
    mode: str = "imperfect",
    repair_type: Optional[str] = None,
) -> JsonDict:
    snapshots = trace.get("snapshots", [])
    if not seeds:
        raise ValueError("replay requires at least one seed")
    if prefix_index < 0 or prefix_index >= len(snapshots):
        raise IndexError(
            f"prefix_index {prefix_index} is outside saved snapshot range "
            f"[0, {len(snapshots) - 1}]"
        )
    snapshot = deepcopy(snapshots[prefix_index])
    extra_tool_calls = 0
    extra_tokens = 0
    if repair_type:
        extra_tokens += repair_cost_tokens(repair_type)
        snapshot = apply_intervention(task, snapshot, repair_type)
        extra_tool_calls += repair_extra_tool_calls(repair_type)
    runs = []
    for seed in seeds:
        run = run_episode(
            task,
            seed=seed,
            mode=mode,
            env_snapshot=snapshot["env"],
            agent_snapshot=snapshot["agent"],
        )
        runs.append(run)
    failures = [not run["final_success"] for run in runs]
    side_effects = [
        bool(run["evaluation"].get("side_effects")) and not run["final_success"]
        for run in runs
    ]
    return {
        "task_id": task.task_id,
        "prefix_index": prefix_index,
        "mode": mode,
        "repair_type": repair_type or "none",
        "n": len(seeds),
        "failure_rate": sum(failures) / max(1, len(failures)),
        "success_rate": 1.0 - sum(failures) / max(1, len(failures)),
        "side_effect_rate": sum(side_effects) / max(1, len(side_effects)),
        "extra_tool_calls": extra_tool_calls,
        "extra_tokens": extra_tokens,
        "runs": runs,
    }


def find_mfp(
    task: TaskSpec,
    trace: JsonDict,
    seeds: List[int],
    p_fail: float = 0.7,
) -> JsonDict:
    total_prefixes = len(trace.get("snapshots", []))
    tested = []
    mfp = None
    for prefix_index in range(total_prefixes):
        result = replay_from_prefix(task, trace, prefix_index, seeds, mode=trace.get("model", "imperfect"))
        tested.append({
            "prefix_index": prefix_index,
            "failure_rate": result["failure_rate"],
            "n": result["n"],
        })
        if result["failure_rate"] >= p_fail and mfp is None:
            mfp = prefix_index
            break
    trace_len = max(1, len(trace.get("trajectory", [])))
    return {
        "task_id": task.task_id,
        "seed": trace.get("seed"),
        "mfp_index": mfp,
        "mfp_norm": None if mfp is None else mfp / trace_len,
        "p_fail": p_fail,
        "n": len(seeds),
        "detected": mfp is not None,
        "tested_prefixes": tested,
    }


def evaluate_all_prefixes(
    task: TaskSpec,
    trace: JsonDict,
    seeds: List[int],
    p_fail: float = 0.7,
) -> JsonDict:
    """Replay every saved prefix.

    This is intentionally more verbose than ``find_mfp``: it keeps prefix-0,
    later-prefix, unstable, and no-MFP cases in the denominator for analysis.
    """

    tested = []
    for prefix_index in range(len(trace.get("snapshots", []))):
        result = replay_from_prefix(
            task,
            trace,
            prefix_index,
            seeds,
            mode=trace.get("model", "imperfect"),
        )
        tested.append({
            "prefix_index": prefix_index,
            "failure_rate": result["failure_rate"],
            "n": result["n"],
        })
    first = next((row for row in tested if row["failure_rate"] >= p_fail), None)
    trace_len = max(1, len(trace.get("trajectory", [])))
    return {
        "task_id": task.task_id,
        "seed": trace.get("seed"),
        "mfp_index": None if first is None else first["prefix_index"],
        "mfp_norm": None if first is None else first["prefix_index"] / trace_len,
        "p_fail": p_fail,
        "n": len(seeds),
        "detected": first is not None,
        "tested_prefixes": tested,
    }


def find_delta_mfp(
    task: TaskSpec,
    trace: JsonDict,
    seeds: List[int],
    p_fail: float = 0.7,
    delta: float = 0.3,
) -> JsonDict:
    """Find the earliest nontrivial MFP relative to prefix-0 recurrence.

    A prefix is nontrivial only when k > 0, the prefix failure rate exceeds
    p_fail, and it increases failure probability over prefix 0 by at least
    ``delta``. Traces are classified for denominator-safe reporting.
    """

    result = evaluate_all_prefixes(task, trace, seeds, p_fail=p_fail)
    tested = result["tested_prefixes"]
    p0 = tested[0]["failure_rate"] if tested else 0.0
    trace_len = max(1, len(trace.get("trajectory", [])))
    delta_mfp = None
    for row in tested:
        k = row["prefix_index"]
        pk = row["failure_rate"]
        if k > 0 and pk >= p_fail and (pk - p0) >= delta:
            delta_mfp = k
            break
    any_reaches = any(row["failure_rate"] >= p_fail for row in tested)
    if delta_mfp is not None:
        category = "nontrivial_mfp"
    elif p0 >= p_fail:
        category = "prefix0_failure"
    elif not any_reaches:
        category = "unstable"
    else:
        category = "no_delta_mfp"
    result.update({
        "p0_failure_rate": p0,
        "delta": delta,
        "delta_mfp_index": delta_mfp,
        "delta_mfp_norm": None if delta_mfp is None else delta_mfp / trace_len,
        "mfp_category": category,
    })
    return result


def apply_intervention(task: TaskSpec, snapshot: JsonDict, repair_type: str) -> JsonDict:
    snap = deepcopy(snapshot)
    repair_type = repair_type.lower()
    if repair_type == "rollback":
        env = DiagnosticEnv(task, state=snap["env"])
        env.restore_last_verified()
        snap["env"] = env.snapshot()
    if repair_type in task.repair_map:
        snap["agent"]["active_plan"] = "correct"
        snap["agent"]["step_index"] = task.repair_map[repair_type]
        snap["agent"]["done"] = False
        snap["agent"]["final_answer"] = None
        snap["agent"].setdefault("memory", {})
        snap["agent"]["memory"]["repair"] = repair_type
    return snap


def repair_cost_tokens(repair_type: str) -> int:
    return {
        "clarify": 70,
        "statecheck": 90,
        "evidenceground": 100,
        "rollback": 120,
        "untrustedguard": 80,
        "self_reflection": 220,
        "simulated_retry": 220,
        "random_routed": 100,
        "majority_routed": 100,
        "predicted_routed": 100,
        "oracle_routed": 100,
    }.get(repair_type, 50)


def repair_extra_tool_calls(repair_type: str) -> int:
    return {
        "clarify": 1,
        "statecheck": 1,
        "evidenceground": 0,
        "rollback": 1,
        "untrustedguard": 0,
        "self_reflection": 0,
        "simulated_retry": 0,
        "random_routed": 1,
        "majority_routed": 1,
        "predicted_routed": 1,
        "oracle_routed": 1,
    }.get(repair_type, 0)
