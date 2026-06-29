"""Repair intervention registry."""

from __future__ import annotations

from typing import Dict


REPAIRS = ["clarify", "statecheck", "evidenceground", "rollback", "untrustedguard"]

TARGETED_BY_TRIGGER: Dict[str, str] = {
    "information": "clarify",
    "state": "statecheck",
    "commitment": "statecheck",
    "grounding": "evidenceground",
    "boundary": "untrustedguard",
    "other": "statecheck",
}


def targeted_repair(trigger_class: str) -> str:
    return TARGETED_BY_TRIGGER.get(trigger_class, "statecheck")


DISPLAY_NAMES: Dict[str, str] = {
    "none": "None",
    "simulated_retry": "Retry",
    "self_reflection": "Reflection",
    "clarify": "Clarify",
    "statecheck": "StateCheck",
    "evidenceground": "Evidence",
    "rollback": "Rollback",
    "untrustedguard": "Guard",
    "random_routed": "Random",
    "majority_routed": "Majority",
    "predicted_routed": "Predicted",
    "oracle_routed": "Oracle",
}


def display_name(method: str) -> str:
    return DISPLAY_NAMES.get(method, method)
