from __future__ import annotations

import pytest

from benchmarks.diagnostic_env import make_tasks
from replay.prefix_replay import (
    evaluate_all_prefixes,
    find_delta_mfp,
    replay_from_prefix,
)
from replay.trace_logger import run_episode


def _failed_trace():
    task = next(t for t in make_tasks()
                if t.task_id == "calendar_missing_timezone")
    trace = run_episode(task, seed=1, mode="imperfect")
    assert trace["final_success"] is False
    return task, trace


def test_prefix_replay_localizes_committed_failure():
    task, trace = _failed_trace()
    seeds = [101, 102, 103, 104, 105]

    prefix0 = replay_from_prefix(task, trace, 0, seeds, mode="imperfect")
    prefix1 = replay_from_prefix(task, trace, 1, seeds, mode="imperfect")

    assert 0.0 < prefix0["failure_rate"] < 1.0
    assert prefix1["failure_rate"] == 1.0

    result = find_delta_mfp(task, trace, seeds, p_fail=0.7, delta=0.3)
    assert result["mfp_category"] == "nontrivial_mfp"
    assert result["delta_mfp_index"] == 1


def test_all_saved_prefixes_are_evaluated():
    task, trace = _failed_trace()
    result = evaluate_all_prefixes(
        task, trace, [101, 102, 103, 104, 105], p_fail=0.7
    )
    assert len(result["tested_prefixes"]) == len(trace["snapshots"])
    assert result["tested_prefixes"][0]["prefix_index"] == 0


@pytest.mark.parametrize("index", [-1, 999])
def test_replay_rejects_invalid_prefix(index):
    task, trace = _failed_trace()
    with pytest.raises(IndexError):
        replay_from_prefix(task, trace, index, [1], mode="imperfect")


def test_replay_requires_at_least_one_seed():
    task, trace = _failed_trace()
    with pytest.raises(ValueError, match="seed"):
        replay_from_prefix(task, trace, 0, [], mode="imperfect")
