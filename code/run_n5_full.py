"""Full N=5 Delta-MFP replay audit for ALL 50 retained soft-fault traces.

Camera-ready Priority-1 experiment (FAGEN @ ICML 2026). Unlike
`run_n5_stability.py` (which subsamples a stratified subset of 27 traces),
this script re-classifies every one of the 50 retained soft traces at N=5
replay, using the SAME task snapshots, per-trace model, p_fail=0.6, delta=0.3,
and candidate prefixes [0, k_f] as the accepted N=2 run.

It does NOT regenerate any soft-fault traces. It reads the existing
`data/traces/soft_fault_traces.jsonl`.

Outputs:
  - data/processed/soft_fault_results_N5_full.csv   (one row per trace, full detail)
  - data/processed/soft_replay_stability_full.csv   (N=2 vs N=5 comparison + transitions)

Technical (execution) errors are recorded with status="execution_error" and are
NEVER converted into an experimental regime ("unstable" et al.). They can be
retried; the failing seeds are logged.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from benchmarks.local_task_suite import load_local_tasks  # noqa: E402
from run_local_llm_suite import classify_delta_mfp  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
TRACES = ROOT / "data" / "traces"
SOFT_TRACES_JSONL = TRACES / "soft_fault_traces.jsonl"
N2_CSV = PROCESSED / "soft_fault_results.csv"
N5_FULL_CSV = PROCESSED / "soft_fault_results_N5_full.csv"
STABILITY_FULL_CSV = PROCESSED / "soft_replay_stability_full.csv"

P_FAIL = 0.6
DELTA = 0.3
REPLAY_N = 5

N5_FIELDS = [
    "trace_id", "model", "interface", "task_id", "family", "difficulty",
    "fault_type", "fault_prefix_index",
    "mfp_category_N5", "delta_mfp_index_N5",
    "p0_failure_rate_N5", "pk_failure_rate_N5", "fault_to_mfp_distance_N5",
    "status", "error",
]
STAB_FIELDS = [
    "trace_id", "fault_type", "family", "difficulty", "fault_prefix_index",
    "mfp_category_N2", "p0_N2", "pk_N2",
    "mfp_category_N5", "p0_N5", "pk_N5",
    "agreement", "transition", "reason", "status",
]


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def trace_id_for(trace: Dict) -> str:
    parts = [
        trace.get("model", "?"), trace.get("interface", "?"),
        trace.get("task_id", "?"), str(trace.get("seed", "?")),
        trace.get("fault_type", "?"),
    ]
    return "::".join(parts)


def pk_from_tested(tested_json: str, kf) -> Optional[float]:
    """Pull p_kf out of the stored tested_prefixes JSON (N=2 baseline)."""
    try:
        tested = json.loads(tested_json)
    except (json.JSONDecodeError, TypeError):
        return None
    for row in tested:
        if int(row["prefix_index"]) == int(kf):
            return float(row["failure_rate"])
    return None


def fault_to_mfp_distance(kf, k_dmfp) -> str:
    if k_dmfp is None or k_dmfp == "":
        return ""
    try:
        return str(abs(int(k_dmfp) - int(kf)))
    except (TypeError, ValueError):
        return ""


def transition_reason(n2_reg, n5_reg, p0, pk) -> str:
    """A diagnosable, rule-grounded reason for an N=2 -> N=5 regime move.

    Grounded directly in the Delta-MFP decision rule (p_fail=0.6, delta=0.3)
    applied to the N=5 estimates (p0, pk over [0, k_f]).
    """
    def f(x):
        return "n/a" if x is None else f"{x:.2f}"
    if n2_reg == n5_reg:
        return f"stable: same regime at N=2 and N=5 (N5 p0={f(p0)}, pk={f(pk)})"
    gap = "" if (p0 is None or pk is None) else f", pk-p0={pk - p0:+.2f}"
    if n5_reg == "nontrivial_delta_mfp":
        why = (f"at N=5 pk={f(pk)} >= p_fail={P_FAIL} and pk-p0 >= delta={DELTA}{gap}: "
               f"the perturbation prefix stabilises into a Delta-MFP once replay noise is averaged over 5 samples")
    elif n5_reg == "prefix0":
        why = (f"at N=5 p0={f(p0)} >= p_fail={P_FAIL}: the failure reproduces from prefix 0 (before the perturbation), "
               f"so it is not attributable to the perturbation prefix")
    elif n5_reg == "unstable":
        why = (f"at N=5 no prefix reaches p_fail={P_FAIL} (p0={f(p0)}, pk={f(pk)}){gap}: "
               f"replay variance dominates; there is no stable failing prefix")
    elif n5_reg == "no_delta":
        why = (f"at N=5 some prefix reaches p_fail={P_FAIL} but no prefix>0 clears the delta gap (p0={f(p0)}, pk={f(pk)}){gap}: "
               f"a failing prefix exists but the perturbation does not add >= delta over prefix 0")
    else:
        why = f"N5 p0={f(p0)}, pk={f(pk)}"
    return f"{n2_reg} -> {n5_reg}: {why}"


def main(limit: int = 0, only_index: Optional[int] = None) -> None:
    n2_rows = load_csv(N2_CSV)
    n2_by_id = {r["trace_id"]: r for r in n2_rows}

    traces = load_jsonl(SOFT_TRACES_JSONL)
    print(f"loaded {len(traces)} soft traces; N2 baseline rows={len(n2_rows)}", flush=True)
    tasks_by_id = {task.task_id: task for task in load_local_tasks()}

    if only_index is not None:
        traces = [traces[only_index]]
    elif limit:
        traces = traces[:limit]

    out_rows: List[Dict] = []
    stab_rows: List[Dict] = []
    started = time.time()
    n_exec_errors = 0

    for i, trace in enumerate(traces, 1):
        tid = trace_id_for(trace)
        kf = trace.get("fault_prefix_index")
        task = tasks_by_id.get(trace["task_id"])
        n2 = n2_by_id.get(tid, {})
        n2_reg = n2.get("mfp_category", "")
        n2_p0 = float(n2["p0_failure_rate"]) if n2.get("p0_failure_rate") not in (None, "") else None
        n2_pk = pk_from_tested(n2.get("tested_prefixes", ""), kf) if kf is not None else None

        status = "ok"
        error = ""
        cls = None
        if task is None:
            status, error = "execution_error", f"task {trace.get('task_id')} not found"
        elif kf is None:
            status, error = "execution_error", "missing fault_prefix_index"
        else:
            try:
                cls = classify_delta_mfp(
                    task, trace, model=trace["model"], interface=trace["interface"],
                    replay_n=REPLAY_N, p_fail=P_FAIL, delta=DELTA,
                    base_seed=5000 + i, candidate_prefixes=[0, kf],
                )
            except Exception as exc:  # noqa: BLE001 - record, never silently drop
                status = "execution_error"
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()

        if cls is not None:
            tested = cls.get("tested_prefixes", [])
            n5_pk = next((r["failure_rate"] for r in tested if r["prefix_index"] == kf), None)
            n5_p0 = cls.get("p0_failure_rate")
            n5_reg = cls.get("mfp_category", "")
            out_rows.append({
                "trace_id": tid, "model": trace.get("model", ""),
                "interface": trace.get("interface", ""), "task_id": trace.get("task_id", ""),
                "family": trace.get("family", ""), "difficulty": trace.get("difficulty", ""),
                "fault_type": trace.get("fault_type", ""), "fault_prefix_index": kf,
                "mfp_category_N5": n5_reg, "delta_mfp_index_N5": cls.get("delta_mfp_index"),
                "p0_failure_rate_N5": n5_p0, "pk_failure_rate_N5": n5_pk,
                "fault_to_mfp_distance_N5": fault_to_mfp_distance(kf, cls.get("delta_mfp_index")),
                "status": status, "error": error,
            })
            agreement = "1" if (n2_reg and n2_reg == n5_reg) else "0"
            transition = "same" if agreement == "1" else f"{n2_reg}->{n5_reg}"
            reason = transition_reason(n2_reg, n5_reg, n5_p0, n5_pk)
        else:
            n_exec_errors += 1
            n5_p0 = n5_pk = None
            n5_reg = ""
            agreement = ""
            transition = "execution_error"
            reason = error
            out_rows.append({
                "trace_id": tid, "model": trace.get("model", ""),
                "interface": trace.get("interface", ""), "task_id": trace.get("task_id", ""),
                "family": trace.get("family", ""), "difficulty": trace.get("difficulty", ""),
                "fault_type": trace.get("fault_type", ""), "fault_prefix_index": kf,
                "mfp_category_N5": "", "delta_mfp_index_N5": "",
                "p0_failure_rate_N5": "", "pk_failure_rate_N5": "",
                "fault_to_mfp_distance_N5": "", "status": status, "error": error,
            })

        stab_rows.append({
            "trace_id": tid, "fault_type": trace.get("fault_type", ""),
            "family": trace.get("family", ""), "difficulty": trace.get("difficulty", ""),
            "fault_prefix_index": kf,
            "mfp_category_N2": n2_reg,
            "p0_N2": "" if n2_p0 is None else f"{n2_p0:.3f}",
            "pk_N2": "" if n2_pk is None else f"{n2_pk:.3f}",
            "mfp_category_N5": n5_reg,
            "p0_N5": "" if n5_p0 is None else f"{n5_p0:.3f}",
            "pk_N5": "" if n5_pk is None else f"{n5_pk:.3f}",
            "agreement": agreement, "transition": transition,
            "reason": reason, "status": status,
        })

        elapsed = time.time() - started
        print(
            f"[{i}/{len(traces)}] {trace.get('fault_type','?')} model={trace.get('model','?')} "
            f"task={trace.get('task_id','?')} kf={kf} | N2={n2_reg}(p0={n2_p0},pk={n2_pk}) "
            f"-> N5={n5_reg}(p0={n5_p0},pk={n5_pk}) agree={agreement} status={status} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        # Incremental write so partial progress survives interruption.
        write_csv(N5_FULL_CSV, out_rows, N5_FIELDS)
        write_csv(STABILITY_FULL_CSV, stab_rows, STAB_FIELDS)

    print(f"done. {len(out_rows)} N=5 rows; {n_exec_errors} execution errors; "
          f"total elapsed={time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="only first K traces (smoke test)")
    parser.add_argument("--only-index", type=int, default=None, help="single trace by 0-based index")
    args = parser.parse_args()
    main(limit=args.limit, only_index=args.only_index)
