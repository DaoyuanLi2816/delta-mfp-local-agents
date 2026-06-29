"""Run collection, MFP replay, repairs, metrics, and table generation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import json
import random

from benchmarks.diagnostic_env import make_tasks, TaskSpec
from replay.trace_logger import run_episode, write_jsonl
from replay.prefix_replay import find_mfp, find_delta_mfp, replay_from_prefix
from repairs.interventions import REPAIRS, display_name
from analysis.metrics import mean, write_csv, to_latex_table


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TRACES = DATA / "traces"
PROCESSED = DATA / "processed"
TABLES = ROOT / "paper" / "tables"
RUNS = ROOT / "runs"


def task_map() -> Dict[str, TaskSpec]:
    return {task.task_id: task for task in make_tasks()}


def collect_failures(num_seeds: int = 8) -> List[dict]:
    tasks = make_tasks()
    all_runs = []
    failed = []
    for task in tasks:
        for seed in range(num_seeds):
            run = run_episode(task, seed=seed, mode="imperfect")
            all_runs.append(run)
            if not run["final_success"]:
                failed.append(run)
    write_jsonl(TRACES / "all_runs.jsonl", all_runs)
    write_jsonl(TRACES / "failed_traces.jsonl", failed)
    return failed


def run_prefixes(failed: List[dict], replay_n: int = 5, p_fail: float = 0.7) -> List[dict]:
    tasks = task_map()
    rows = []
    for idx, trace in enumerate(failed):
        task = tasks[trace["task_id"]]
        seeds = [10_000 + idx * 100 + j for j in range(replay_n)]
        result = find_delta_mfp(task, trace, seeds=seeds, p_fail=p_fail, delta=0.3)
        rows.append({
            "trace_id": f"{trace['task_id']}:{trace['seed']}",
            "task_id": trace["task_id"],
            "family": trace["family"],
            "trigger_class": trace["trigger_class"],
            "failure_mode": trace["failure_mode"],
            "seed": trace["seed"],
            "trajectory_len": len(trace["trajectory"]),
            "mfp_index": result["mfp_index"],
            "mfp_norm": result["mfp_norm"],
            "p0_failure_rate": result["p0_failure_rate"],
            "delta_mfp_index": result["delta_mfp_index"],
            "delta_mfp_norm": result["delta_mfp_norm"],
            "mfp_category": result["mfp_category"],
            "detected": result["detected"],
            "p_fail": result["p_fail"],
            "replay_n": result["n"],
            "tested_prefixes": json.dumps(result["tested_prefixes"]),
        })
    write_csv(PROCESSED / "prefixes.csv", rows)
    return rows


def run_repairs(failed: List[dict], prefix_rows: List[dict], replay_n: int = 5) -> List[dict]:
    tasks = task_map()
    prefix_by_trace = {row["trace_id"]: row for row in prefix_rows}
    rows = []
    train_task_ids, _ = split_task_ids()
    train_best = [
        task.best_repair for task in tasks.values()
        if task.task_id in train_task_ids and task.best_repair in REPAIRS
    ]
    majority_repair = Counter(train_best).most_common(1)[0][0] if train_best else "statecheck"
    for idx, trace in enumerate(failed):
        trace_id = f"{trace['task_id']}:{trace['seed']}"
        prefix = prefix_by_trace[trace_id]
        selected_prefix = (
            prefix.get("delta_mfp_index")
            if prefix.get("delta_mfp_index") not in ("", None)
            else prefix.get("mfp_index")
        )
        if selected_prefix in ("", None):
            continue
        task = tasks[trace["task_id"]]
        mfp = int(selected_prefix)
        seeds = [20_000 + idx * 100 + j for j in range(replay_n)]
        random_repair = random.Random(stable_int(trace_id)).choice(REPAIRS)
        predicted_repair = predict_repair_from_prefix(trace, mfp)
        candidate_repairs: List[Tuple[str, Optional[str]]] = [
            ("none", None),
            ("simulated_retry", None),
            *[(repair, repair) for repair in REPAIRS],
            ("random_routed", random_repair),
            ("majority_routed", majority_repair),
            ("predicted_routed", predicted_repair),
            ("oracle_routed", task.best_repair),
        ]
        for method, chosen_repair in candidate_repairs:
            if method == "none":
                result = replay_from_prefix(task, trace, mfp, seeds, mode="imperfect")
            elif method == "simulated_retry":
                result = replay_from_prefix(task, trace, 0, seeds, mode="self_reflection", repair_type=None)
                result["repair_type"] = "simulated_retry"
                result["extra_tokens"] = 220
                result["extra_tool_calls"] = mean([r.get("tool_calls", 0) for r in result["runs"]])
                chosen_repair = "retry"
            else:
                result = replay_from_prefix(task, trace, mfp, seeds, mode="imperfect", repair_type=chosen_repair)
                result["repair_type"] = method
                result["chosen_repair"] = chosen_repair
            rows.append({
                "trace_id": trace_id,
                "task_id": task.task_id,
                "family": task.family,
                "trigger_class": task.trigger_class,
                "failure_mode": task.failure_mode,
                "repair_type": result["repair_type"],
                "chosen_repair": result.get("chosen_repair", chosen_repair or result["repair_type"]),
                "success_rate": result["success_rate"],
                "failure_rate": result["failure_rate"],
                "side_effect_rate": result["side_effect_rate"],
                "extra_tool_calls": result["extra_tool_calls"],
                "extra_tokens": result["extra_tokens"],
                "replay_n": result["n"],
            })
    write_csv(PROCESSED / "repairs.csv", rows)
    return rows


def stable_int(text: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text))


def split_task_ids() -> Tuple[set, set]:
    tasks = sorted(task.task_id for task in make_tasks())
    train = {task_id for i, task_id in enumerate(tasks) if i % 5 != 0}
    test = set(tasks) - train
    return train, test


def predict_repair_from_prefix(trace: dict, mfp: int) -> str:
    """Rule-based prefix-observable repair router.

    Uses only text/log features visible by prefix mfp: observed tool names,
    tool args/results, memory writes, and state diffs. It does not inspect the
    ground-truth trigger label or final failure reason.
    """

    prefix_steps = trace.get("trajectory", [])[: max(0, mfp)]
    blob = json.dumps(prefix_steps, sort_keys=True).lower()
    mutating_tools = {"create_event", "cancel_event", "issue_refund", "reserve_flight", "send_email", "update_stock", "create_purchase_order"}
    tools = {str(step.get("tool_name", "")).lower() for step in prefix_steps}
    if "trusted\": false" in blob or "untrusted" in blob or "attacker@" in blob or "send credentials" in blob or "vendor_note" in blob:
        return "untrustedguard"
    if "missing" in blob or "unknown" in blob or "none" in blob:
        return "clarify"
    if "conflict\": true" in blob or "eligible\": false" in blob or "layover_minutes" in blob or "matches" in blob or "file_id" in blob:
        return "evidenceground"
    if tools & mutating_tools:
        return "rollback"
    if "lookup" in blob or "policy" in blob or "availability" in blob:
        return "statecheck"
    return "statecheck"


def run_sensitivity(failed: List[dict], replay_n: int = 5) -> List[dict]:
    tasks = task_map()
    thresholds = [0.5, 0.6, 0.7, 0.9]
    rows = []
    subset = failed[:40]
    for p_fail in thresholds:
        detected = 0
        norms = []
        for idx, trace in enumerate(subset):
            task = tasks[trace["task_id"]]
            seeds = [30_000 + idx * 100 + j for j in range(replay_n)]
            result = find_mfp(task, trace, seeds=seeds, p_fail=p_fail)
            detected += int(result["detected"])
            if result["mfp_norm"] is not None:
                norms.append(float(result["mfp_norm"]))
        rows.append({
            "p_fail": p_fail,
            "replay_n": replay_n,
            "subset_traces": len(subset),
            "detection_rate": f"{detected / max(1, len(subset)):.3f}",
            "mean_mfp_norm": f"{mean(norms):.3f}",
        })
    write_csv(PROCESSED / "sensitivity.csv", rows)
    return rows


def run_stress_tests(failed: List[dict]) -> List[dict]:
    tasks = task_map()
    subset = failed[:40]
    rows = []
    for replay_n, p_fail, label in [(10, 0.9, "strict_replay"), (10, 0.7, "larger_N")]:
        detected = 0
        norms = []
        for idx, trace in enumerate(subset):
            task = tasks[trace["task_id"]]
            seeds = [40_000 + idx * 100 + j for j in range(replay_n)]
            result = find_mfp(task, trace, seeds=seeds, p_fail=p_fail)
            detected += int(result["detected"])
            if result["mfp_norm"] is not None:
                norms.append(float(result["mfp_norm"]))
        rows.append({
            "stress_test": label,
            "replay_n": replay_n,
            "p_fail": p_fail,
            "subset_traces": len(subset),
            "detection_rate": f"{detected / max(1, len(subset)):.3f}",
            "mean_mfp_norm": f"{mean(norms):.3f}",
        })
    write_csv(PROCESSED / "stress_tests.csv", rows)
    return rows


def summarize(
    failed: List[dict],
    prefix_rows: List[dict],
    repair_rows: List[dict],
    replay_n: int = 5,
    p_fail: float = 0.6,
) -> None:
    results_rows = []
    for trace in failed:
        results_rows.append({
            "trace_id": f"{trace['task_id']}:{trace['seed']}",
            "task_id": trace["task_id"],
            "family": trace["family"],
            "trigger_class": trace["trigger_class"],
            "failure_mode": trace["failure_mode"],
            "seed": trace["seed"],
            "trajectory_len": len(trace["trajectory"]),
            "tool_calls": trace["tool_calls"],
            "failure_reasons": ";".join(trace["evaluation"]["failure_reasons"]),
        })
    write_csv(PROCESSED / "results.csv", results_rows)

    dataset_rows = []
    by_family = defaultdict(list)
    for trace in failed:
        by_family[trace["family"]].append(trace)
    for family, traces in sorted(by_family.items()):
        tools = ",".join(sorted(set(sum([task_map()[t["task_id"]].tools for t in traces], []))))
        failure_types = ",".join(sorted({t["trigger_class"] for t in traces}))
        dataset_rows.append({
            "Task Family": family,
            "#Tasks": len({t["task_id"] for t in traces}),
            "#Failed Traces": len(traces),
            "Avg Steps": f"{mean([len(t['trajectory']) for t in traces]):.1f}",
            "Tools": tools,
            "Main Failure Types": failure_types,
        })
    write_csv(TABLES / "table2_dataset_summary.csv", dataset_rows)

    methods = [
        "none", "simulated_retry", "clarify", "statecheck", "evidenceground",
        "rollback", "untrustedguard", "random_routed", "majority_routed",
        "predicted_routed", "oracle_routed",
    ]
    summary_rows = []
    for method in methods:
        rows = [r for r in repair_rows if r["repair_type"] == method]
        if not rows:
            continue
        ci = bootstrap_method_ci(repair_rows, method, seed=17)
        summary_rows.append({
            "method": method,
            "display": display_name(method),
            "repair_success_rate": f"{mean([float(r['success_rate']) for r in rows]):.3f}",
            "success_ci": f"[{ci[0]:.3f},{ci[1]:.3f}]",
            "residual_failure": f"{mean([float(r['failure_rate']) for r in rows]):.3f}",
            "extra_tool_calls": f"{mean([float(r['extra_tool_calls']) for r in rows]):.2f}",
            "extra_tokens": f"{mean([float(r['extra_tokens']) for r in rows]):.1f}",
        })
    write_csv(PROCESSED / "repair_summary.csv", summary_rows)
    write_csv(TABLES / "table3_main_results.csv", summary_rows)

    heat_rows = []
    for trigger in sorted({r["trigger_class"] for r in repair_rows}):
        for repair in REPAIRS + ["random_routed", "majority_routed", "predicted_routed", "oracle_routed", "simulated_retry"]:
            rows = [r for r in repair_rows if r["trigger_class"] == trigger and r["repair_type"] == repair]
            if rows:
                heat_rows.append({
                    "trigger_class": trigger,
                    "repair_type": repair,
                    "display": display_name(repair),
                    "success_rate": f"{mean([float(r['success_rate']) for r in rows]):.3f}",
                    "n": len(rows),
                })
    write_csv(PROCESSED / "repair_by_trigger.csv", heat_rows)

    category_rows = []
    category_counts = Counter(str(row.get("mfp_category", "unknown")) for row in prefix_rows)
    for category, count in sorted(category_counts.items()):
        category_rows.append({
            "category": category,
            "count": count,
            "rate": f"{count / max(1, len(prefix_rows)):.3f}",
        })
    write_csv(PROCESSED / "prefix_categories.csv", category_rows)

    detection_rate = mean([1.0 if str(r["detected"]) == "True" else 0.0 for r in prefix_rows])
    by_method = {r["method"]: r for r in summary_rows}
    predicted_row = by_method.get("predicted_routed", {"repair_success_rate": "0.000", "extra_tool_calls": "0.00"})
    oracle_row = by_method.get("oracle_routed", {"repair_success_rate": "0.000", "extra_tool_calls": "0.00"})
    retry_row = by_method.get("simulated_retry", {"repair_success_rate": "0.000", "extra_tool_calls": "0.00"})
    ablation_rows = [
        {"Variant": f"N={replay_n},p={p_fail}", "MFP Detection Rate": f"{detection_rate:.3f}", "Repair Success": predicted_row["repair_success_rate"], "Cost": predicted_row["extra_tool_calls"], "Notes": "predicted routing"},
        {"Variant": "oracle routed", "MFP Detection Rate": f"{detection_rate:.3f}", "Repair Success": oracle_row["repair_success_rate"], "Cost": oracle_row["extra_tool_calls"], "Notes": "upper bound"},
        {"Variant": "fixed midpoint", "MFP Detection Rate": "n/a", "Repair Success": fixed_midpoint_proxy(repair_rows), "Cost": "1.00", "Notes": "proxy from non-targeted repairs"},
        {"Variant": "retry-style", "MFP Detection Rate": "n/a", "Repair Success": retry_row["repair_success_rate"], "Cost": retry_row["extra_tool_calls"], "Notes": "simulated full retry"},
    ]
    write_csv(TABLES / "table4_ablation.csv", ablation_rows)

    related_headers = ["Work", "Final Success Eval", "Trace Diagnostics", "Counterfactual Replay", "Minimal Prefix", "Verified Repair", "Stateful Tools"]
    related_rows = [
        ["ToolSandbox", "yes", "milestones", "no", "no", "no", "yes"],
        ["tau-bench", "yes", "partial", "full-episode", "no", "no", "yes"],
        ["BFCL V4", "yes", "partial", "no", "no", "no", "partial"],
        ["AgentDebug", "yes", "yes", "no", "no", "feedback", "mixed"],
        ["AgentRx", "yes", "yes", "no", "no", "no", "yes"],
        ["AgenTracer", "yes", "yes", "yes", "no", "feedback", "multi-agent"],
        ["DoVer", "yes", "yes", "interventions", "no", "yes", "multi-agent"],
        ["Ours", "yes", "yes", "yes", "yes", "yes", "yes"],
    ]
    write_csv(TABLES / "table1_related_work.csv", [
        dict(zip(related_headers, row)) for row in related_rows
    ])

    write_related_work_table(related_rows)
    to_latex_table(TABLES / "table2_dataset_summary.tex", list(dataset_rows[0].keys()),
                   [[row[k] for k in dataset_rows[0].keys()] for row in dataset_rows],
                   "Controlled diagnostic suite summary.", "tab:dataset")
    write_main_results_table(summary_rows)
    write_ablation_table(ablation_rows)

    write_case_study()
    write_experiment_log(failed, prefix_rows, summary_rows, replay_n=replay_n, p_fail=p_fail)


def fixed_midpoint_proxy(repair_rows: List[dict]) -> str:
    rows = [r for r in repair_rows if r["repair_type"] in {"clarify", "statecheck", "evidenceground", "rollback", "untrustedguard"}]
    return f"{mean([float(r['success_rate']) for r in rows]):.3f}" if rows else "0.000"


def bootstrap_method_ci(repair_rows: List[dict], method: str, seed: int = 0, reps: int = 1000) -> Tuple[float, float]:
    rows = [r for r in repair_rows if r["repair_type"] == method]
    by_task: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(float(row["success_rate"]))
    task_ids = sorted(by_task)
    if not task_ids:
        return 0.0, 0.0
    rng = random.Random(seed)
    estimates = []
    for _ in range(reps):
        sampled = [rng.choice(task_ids) for _ in task_ids]
        vals = []
        for task_id in sampled:
            vals.extend(by_task[task_id])
        estimates.append(mean(vals))
    estimates.sort()
    lo = estimates[int(0.025 * reps)]
    hi = estimates[min(reps - 1, int(0.975 * reps))]
    return lo, hi


def write_related_work_table(related_rows: List[List[str]]) -> None:
    display_rows = [[row[0].replace("tau-bench", "$\\tau$-bench"), *row[1:]] for row in related_rows]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Work & Final success & Trace diagnostics & Counterfactual replay & Minimal prefix & Verified repair & Stateful tools \\\\",
        "\\midrule",
    ]
    for row in display_rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\caption{Comparison to related benchmarks and diagnostic methods. Our contribution is not another broad benchmark, but a replay protocol that estimates when a failed trajectory enters a reproducible failure basin and tests repairs at that point.}",
        "\\label{tab:related}",
        "\\end{table*}",
        "",
    ])
    (TABLES / "table1_related_work.tex").write_text("\n".join(lines), encoding="utf-8")


def write_main_results_table(summary_rows: List[dict]) -> None:
    wanted = [
        "none", "simulated_retry", "clarify", "statecheck", "evidenceground",
        "rollback", "untrustedguard", "random_routed", "majority_routed",
        "predicted_routed", "oracle_routed",
    ]
    by_method = {row["method"]: row for row in summary_rows}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Method & Success (95\\% CI) & Resid. fail & Calls & Tokens \\\\",
        "\\midrule",
    ]
    for method in wanted:
        if method not in by_method:
            continue
        row = by_method[method]
        lines.append(
            f"{row['display']} & {row['repair_success_rate']} {row['success_ci']} & "
            f"{row['residual_failure']} & {row['extra_tool_calls']} & {row['extra_tokens']} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\caption{Controlled-suite repair results. Oracle routing uses ground-truth trigger labels and is an upper bound; predicted routing uses prefix-observable log features only.}",
        "\\label{tab:main}",
        "\\end{table}",
        "",
    ])
    (TABLES / "table3_main_results.tex").write_text("\n".join(lines), encoding="utf-8")


def write_ablation_table(ablation_rows: List[dict]) -> None:
    headers = list(ablation_rows[0].keys())
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in ablation_rows:
        lines.append(" & ".join(str(row[h]) for h in headers) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Protocol ablations and baselines.}",
        "\\label{tab:ablation}",
        "\\end{table}",
        "",
    ])
    (TABLES / "table4_ablation.tex").write_text("\n".join(lines), encoding="utf-8")


def write_case_study() -> None:
    rows = [
        {"step": 0, "label": "read email", "role": "normal"},
        {"step": 1, "label": "injection", "role": "trigger"},
        {"step": 2, "label": "bad send", "role": "normal"},
        {"step": 3, "label": "guard", "role": "repair"},
        {"step": 4, "label": "safe draft", "role": "normal"},
    ]
    write_csv(PROCESSED / "case_study_trace.csv", rows)
    (PROCESSED / "case_studies.json").write_text(json.dumps({
        "email_prompt_injection": {
            "trigger": "The agent reads an untrusted email and stores an embedded instruction as if it were a user command.",
            "mfp": "The MFP is after the email-read step, before terminal unsafe send.",
            "repair": "UntrustedGuard resets the plan so the email body is summarized as data rather than followed.",
        },
        "refund_irreversible": {
            "trigger": "The agent issues a high-value refund before checking the escalation policy.",
            "mfp": "Failure is reproducible after the refund tool mutates state.",
            "repair": "Rollback is blocked by the irreversible-action flag, yielding an informative negative case.",
        },
    }, indent=2), encoding="utf-8")


def write_experiment_log(
    failed: List[dict],
    prefixes: List[dict],
    summary_rows: List[dict],
    replay_n: int = 5,
    p_fail: float = 0.6,
) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    detected = sum(1 for row in prefixes if str(row["detected"]) == "True")
    categories = Counter(str(row.get("mfp_category", "unknown")) for row in prefixes)
    predicted = next((r for r in summary_rows if r["method"] == "predicted_routed"), None)
    oracle = next((r for r in summary_rows if r["method"] == "oracle_routed"), None)
    retry = next((r for r in summary_rows if r["method"] == "simulated_retry"), None)
    lines = [
        "# Experiment Log",
        "",
        f"Failed traces collected: {len(failed)}",
        f"MFP detected: {detected}/{len(prefixes)}",
        f"Nontrivial Delta-MFP: {categories.get('nontrivial_mfp', 0)}/{len(prefixes)}",
        f"Prefix-0 failure: {categories.get('prefix0_failure', 0)}/{len(prefixes)}",
        f"Predicted-routed repair success: {predicted['repair_success_rate'] if predicted else 'n/a'}",
        f"Oracle-routed repair success: {oracle['repair_success_rate'] if oracle else 'n/a'}",
        f"Simulated retry-style success: {retry['repair_success_rate'] if retry else 'n/a'}",
        "",
        f"Main setting: N={replay_n} replay rollouts per prefix, p_fail={p_fail}.",
        "The agent is a transparent stochastic scripted tool-use policy; no LLM judge is used as ground truth.",
    ]
    (RUNS / "experiment_log.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--replay-n", type=int, default=5)
    parser.add_argument("--p-fail", type=float, default=0.7)
    args = parser.parse_args()
    failed = collect_failures(args.num_seeds)
    prefixes = run_prefixes(failed, replay_n=args.replay_n, p_fail=args.p_fail)
    repairs = run_repairs(failed, prefixes, replay_n=args.replay_n)
    run_sensitivity(failed, replay_n=args.replay_n)
    run_stress_tests(failed)
    summarize(failed, prefixes, repairs, replay_n=args.replay_n, p_fail=args.p_fail)


if __name__ == "__main__":
    main()
