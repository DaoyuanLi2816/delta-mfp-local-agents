"""Optional cross-model robustness check (camera-ready Priority-3).

Re-runs the Delta-MFP classification on a balanced subset of the existing soft
scenarios using a LARGER local model (Qwen2.5-14B) as the replay policy, to
address the reviewer concern about limited model scope. This is framed as a
robustness check, NOT a new benchmark or a new primary claim.

Selection: 4 traces per fault subtype (24 total), deterministic by seed order.
Same tool interface, same p_fail=0.6, delta=0.3, candidate prefixes [0, k_f].
Replay N=3.

For each scenario we restore the original (7B/8B) snapshot+history at prefix 0
and at k_f, then let Qwen2.5-14B sample fresh continuations. p0 thus probes
whether 14B can do the clean task; p_kf probes whether the perturbation triggers
failure for 14B. We compare the 14B regime to the original model's N=2 regime.

Output: data/processed/cross_model_robustness.csv

If 14B cannot complete the workflow reliably (degenerate all-fail), that is a
documented capability limitation, not a forced result. Use --model to switch to
llama3.1:8b as a secondary probe if needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from benchmarks.local_task_suite import load_local_tasks  # noqa: E402
from run_local_llm_suite import classify_delta_mfp  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
TRACES = ROOT / "data" / "traces"
SOFT_JSONL = TRACES / "soft_fault_traces.jsonl"
N2_CSV = PROCESSED / "soft_fault_results.csv"
OUT_CSV = PROCESSED / "cross_model_robustness.csv"

P_FAIL, DELTA, REPLAY_N = 0.6, 0.3, 3
PER_SUBTYPE = 4

FIELDS = [
    "trace_id", "fault_type", "original_model", "probe_model", "task_id",
    "family", "difficulty", "fault_prefix_index",
    "orig_regime_N2", "probe_p0_N3", "probe_pk_N3", "probe_regime_N3",
    "agreement_with_original", "status", "error", "elapsed_sec",
]


def load_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def trace_id_for(t: Dict) -> str:
    return f"{t['model']}::{t['interface']}::{t['task_id']}::{t['seed']}::{t.get('fault_type')}"


def model_quant(model: str) -> str:
    try:
        out = subprocess.check_output(["ollama", "show", model], text=True, timeout=30)
        for line in out.splitlines():
            if "quantization" in line.lower():
                return line.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({exc})"
    return "unknown"


def main(model: str, per_subtype: int) -> None:
    print(f"probe model: {model} | quant: {model_quant(model)}", flush=True)
    n2 = {r["trace_id"]: r for r in load_csv(N2_CSV)}
    traces = load_jsonl(SOFT_JSONL)
    # Balanced selection: first `per_subtype` traces of each fault_type (sorted by seed).
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for t in sorted(traces, key=lambda x: int(x["seed"])):
        by_type[t.get("fault_type")].append(t)
    selected: List[Dict] = []
    for ft, lst in sorted(by_type.items()):
        selected.extend(lst[:per_subtype])
    print(f"selected {len(selected)} traces across {len(by_type)} subtypes "
          f"({per_subtype}/subtype target)", flush=True)

    tasks_by_id = {task.task_id: task for task in load_local_tasks()}
    rows: List[Dict] = []
    started = time.time()
    for i, trace in enumerate(selected, 1):
        tid = trace_id_for(trace)
        kf = trace.get("fault_prefix_index")
        task = tasks_by_id.get(trace["task_id"])
        orig_reg = (n2.get(tid, {}) or {}).get("mfp_category", "")
        status, error, cls = "ok", "", None
        t0 = time.time()
        if task is None or kf is None:
            status, error = "execution_error", "missing task or kf"
        else:
            try:
                cls = classify_delta_mfp(
                    task, trace, model=model, interface=trace["interface"],
                    replay_n=REPLAY_N, p_fail=P_FAIL, delta=DELTA,
                    base_seed=900_000 + i * 13, candidate_prefixes=[0, kf],
                )
            except Exception as exc:  # noqa: BLE001
                status, error = "execution_error", f"{type(exc).__name__}: {exc}"
        elapsed = time.time() - t0
        if cls is not None:
            tested = cls.get("tested_prefixes", [])
            pk = next((r["failure_rate"] for r in tested if r["prefix_index"] == kf), None)
            p0 = cls.get("p0_failure_rate")
            reg = cls.get("mfp_category", "")
            agree = "1" if (orig_reg and orig_reg == reg) else "0"
        else:
            pk = p0 = reg = ""
            agree = ""
        rows.append({
            "trace_id": tid, "fault_type": trace.get("fault_type", ""),
            "original_model": trace.get("model", ""), "probe_model": model,
            "task_id": trace.get("task_id", ""), "family": trace.get("family", ""),
            "difficulty": trace.get("difficulty", ""), "fault_prefix_index": kf,
            "orig_regime_N2": orig_reg, "probe_p0_N3": p0, "probe_pk_N3": pk,
            "probe_regime_N3": reg, "agreement_with_original": agree,
            "status": status, "error": error, "elapsed_sec": f"{elapsed:.1f}",
        })
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"[{i}/{len(selected)}] {trace.get('fault_type','?')} "
              f"orig={trace.get('model','?')}({orig_reg}) -> {model} "
              f"p0={p0} pk={pk} reg={reg} agree={agree} status={status} "
              f"t={elapsed:.0f}s", flush=True)

    # Summary
    ok = [r for r in rows if r["status"] == "ok"]
    dist = defaultdict(int)
    for r in ok:
        dist[r["probe_regime_N3"]] += 1
    agree_n = sum(1 for r in ok if r["agreement_with_original"] == "1")
    print(f"\n=== {model} regime distribution (n={len(ok)} ok): {dict(dist)}", flush=True)
    print(f"=== agreement with original-model N=2 regime: {agree_n}/{len(ok)}", flush=True)
    print(f"=== total elapsed {time.time()-started:.1f}s; "
          f"{sum(1 for r in rows if r['status']!='ok')} execution errors", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--per-subtype", type=int, default=PER_SUBTYPE)
    args = ap.parse_args()
    main(model=args.model, per_subtype=args.per_subtype)
