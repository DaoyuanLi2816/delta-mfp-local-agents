"""Trace logging utilities."""

from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from agents.base_agent import ScriptedToolAgent
from benchmarks.diagnostic_env import DiagnosticEnv, TaskSpec


JsonDict = Dict[str, Any]


def run_episode(
    task: TaskSpec,
    seed: int,
    mode: str = "imperfect",
    env_snapshot: Optional[JsonDict] = None,
    agent_snapshot: Optional[JsonDict] = None,
    forced_plan: Optional[str] = None,
    max_steps: int = 12,
) -> JsonDict:
    env = DiagnosticEnv(task, state=env_snapshot)
    agent = ScriptedToolAgent(task, seed=seed, mode=mode, snapshot=agent_snapshot, forced_plan=forced_plan)
    steps: List[JsonDict] = []
    snapshots: List[JsonDict] = [{"env": env.snapshot(), "agent": agent.snapshot()}]
    final_answer: Optional[str] = None

    for local_step in range(max_steps):
        obs = env.observe(agent.snapshot_state.memory)
        action, step_spec = agent.act()
        result, state_diff = env.execute(action, step_spec)
        if action.kind == "final":
            final_answer = action.final_answer
        step_record = {
            "step": local_step,
            "observation": obs,
            "agent_state_before": snapshots[-1]["agent"],
            "action": asdict(action),
            "tool_name": action.tool_name,
            "tool_args": action.tool_args,
            "tool_result": result,
            "memory_write": action.memory_write,
            "state_diff": state_diff,
            "plan_step_note": step_spec.note if step_spec else "",
        }
        steps.append(step_record)
        snapshots.append({"env": env.snapshot(), "agent": agent.snapshot()})
        if agent.snapshot_state.done:
            break

    evaluation = env.evaluate(final_answer)
    return {
        "task_id": task.task_id,
        "family": task.family,
        "trigger_class": task.trigger_class,
        "failure_mode": task.failure_mode,
        "model": mode,
        "seed": seed,
        "trajectory": steps,
        "snapshots": snapshots,
        "final_success": evaluation.success,
        "evaluation": evaluation.to_dict(),
        "tool_calls": env.state.get("tool_calls", 0),
        "final_state": env.snapshot(),
        "final_answer": final_answer,
    }


def write_jsonl(path: Path, rows: List[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[JsonDict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compact_trace_for_tokens(trace: JsonDict) -> str:
    rows = []
    for step in trace.get("trajectory", []):
        rows.append(
            f"{step['step']} {step['action'].get('kind')} {step.get('tool_name')} "
            f"{step.get('tool_args')} {step.get('tool_result')}"
        )
    return "\n".join(rows)
