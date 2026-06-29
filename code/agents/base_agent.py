"""Scripted stochastic agents for the diagnostic suite."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
import random

from benchmarks.diagnostic_env import AgentAction, StepSpec, TaskSpec


JsonDict = Dict[str, Any]


@dataclass
class AgentSnapshot:
    step_index: int = 0
    active_plan: Optional[str] = None
    memory: JsonDict = field(default_factory=dict)
    done: bool = False
    final_answer: Optional[str] = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


class ScriptedToolAgent:
    """A transparent stochastic tool-use agent.

    The agent follows a correct or faulty plan. The branch is stochastic before
    the failure trigger and then stored in memory, which lets prefix replay
    distinguish unstable early states from committed failing states.
    """

    def __init__(
        self,
        task: TaskSpec,
        seed: int,
        mode: str = "imperfect",
        snapshot: Optional[JsonDict] = None,
        forced_plan: Optional[str] = None,
    ):
        self.task = task
        self.rng = random.Random(seed)
        self.seed = seed
        self.mode = mode
        if snapshot is None:
            self.snapshot_state = AgentSnapshot()
        else:
            self.snapshot_state = AgentSnapshot(**deepcopy(snapshot))
        if forced_plan is not None:
            self.snapshot_state.active_plan = forced_plan

    def snapshot(self) -> JsonDict:
        return self.snapshot_state.to_dict()

    def choose_plan_if_needed(self) -> None:
        st = self.snapshot_state
        if st.active_plan is not None:
            return
        if st.step_index < self.task.branch_step:
            return
        error_rate = self.task.error_rate
        if self.mode == "self_reflection":
            error_rate = self.task.self_reflection_error_rate
        st.active_plan = "faulty" if self.rng.random() < error_rate else "correct"
        st.memory["branch_seed"] = self.seed
        st.memory["branch_step"] = st.step_index
        st.memory["chosen_plan"] = st.active_plan

    def act(self) -> Tuple[AgentAction, Optional[StepSpec]]:
        st = self.snapshot_state
        if st.done:
            return AgentAction(kind="final", final_answer=st.final_answer or "Done."), None
        self.choose_plan_if_needed()
        if st.active_plan is None:
            plan = self.task.correct_steps
        else:
            plan = self.task.faulty_steps if st.active_plan == "faulty" else self.task.correct_steps
        if st.step_index >= len(plan):
            st.done = True
            st.final_answer = "Done."
            return AgentAction(kind="final", final_answer=st.final_answer), None
        step = plan[st.step_index]
        action = AgentAction(
            kind=step.kind,
            tool_name=step.tool_name,
            tool_args=deepcopy(step.tool_args),
            memory_write=deepcopy(step.memory_write),
            final_answer=step.final_answer,
            note=step.note,
        )
        st.memory.update(deepcopy(step.memory_write))
        st.step_index += 1
        if step.kind == "final":
            st.done = True
            st.final_answer = step.final_answer
        return action, step

    def apply_repair(self, repair_type: str, resume_step: Optional[int] = None) -> None:
        st = self.snapshot_state
        st.active_plan = "correct"
        st.memory["repair"] = repair_type
        if resume_step is not None:
            st.step_index = resume_step
            st.done = False
            st.final_answer = None


def estimate_tokens(trace_text: str) -> int:
    """Cheap token proxy used only for relative repair-cost accounting."""

    return max(1, int(len(trace_text.split()) * 1.3))
