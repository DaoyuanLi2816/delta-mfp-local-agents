"""Smoke checks for replay semantics.

Run from the repository root:

    python code/tests/replay_smoke_checks.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from benchmarks.diagnostic_env import make_tasks  # noqa: E402
from replay.trace_logger import run_episode  # noqa: E402
from replay.prefix_replay import replay_from_prefix, evaluate_all_prefixes, find_delta_mfp  # noqa: E402


def main() -> None:
    task = next(t for t in make_tasks() if t.task_id == "calendar_missing_timezone")
    trace = run_episode(task, seed=1, mode="imperfect")
    assert not trace["final_success"], "seed=1 should choose the faulty branch for this smoke test"

    assert trace["snapshots"][0]["agent"]["step_index"] == 0
    assert trace["snapshots"][0]["env"]["effects"] == []
    assert trace["snapshots"][0]["env"]["violations"] == []

    # Prefix 0 is before any action and samples new continuations.
    seeds = [101, 102, 103, 104, 105]
    prefix0 = replay_from_prefix(task, trace, 0, seeds, mode="imperfect")
    assert 0.0 < prefix0["failure_rate"] < 1.0, prefix0["failure_rate"]

    # Prefix 1 is after the faulty branch has been written to agent state.
    prefix1 = replay_from_prefix(task, trace, 1, seeds, mode="imperfect")
    assert prefix1["failure_rate"] == 1.0, prefix1["failure_rate"]

    full = evaluate_all_prefixes(task, trace, seeds, p_fail=0.7)
    assert len(full["tested_prefixes"]) == len(trace["snapshots"])
    assert full["tested_prefixes"][0]["prefix_index"] == 0

    delta = find_delta_mfp(task, trace, seeds, p_fail=0.7, delta=0.3)
    assert delta["mfp_category"] == "nontrivial_mfp", delta
    assert delta["delta_mfp_index"] == 1, delta

    out = ROOT / "runs" / "replay_smoke_checks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "prefix0_failure_rate": prefix0["failure_rate"],
        "prefix1_failure_rate": prefix1["failure_rate"],
        "delta_category": delta["mfp_category"],
        "delta_mfp_index": delta["delta_mfp_index"],
    }, indent=2), encoding="utf-8")
    print(f"Replay smoke checks passed: {out}")


if __name__ == "__main__":
    main()
