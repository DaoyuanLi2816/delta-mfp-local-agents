"""Verify the published headline claims from the bundled artifact.

This module uses only the Python standard library. It does not run a model or
regenerate experimental observations; it checks that the released tasks,
traces, and processed CSVs support the camera-ready paper's stated counts.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PAPER_SHA256 = "e6cfd8175309be22e04de88a644d34997cf8b0713f9cbb00a0ecf235cfc3e6d6"


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _counts(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_claims(root: Path) -> dict[str, Any]:
    """Return expected-versus-actual checks for every published headline."""
    root = root.resolve()
    data = root / "data"
    processed = data / "processed"

    tasks = _jsonl_rows(data / "tasks" / "local_calibrated_tasks.jsonl")
    natural_traces = _jsonl_rows(data / "traces" / "natural_failed_traces.jsonl")
    persistent_traces = _jsonl_rows(data / "traces" / "fault_injected_traces.jsonl")
    soft_traces = _jsonl_rows(data / "traces" / "soft_fault_traces.jsonl")

    calibration = _csv_rows(processed / "calibration_results.csv")
    natural = _csv_rows(processed / "natural_mfp_results.csv")
    persistent = _csv_rows(processed / "fault_injection_results.csv")
    soft_n5 = _csv_rows(processed / "soft_fault_results_N5_full.csv")
    stability = _csv_rows(processed / "soft_replay_stability_full.csv")
    cross_model = _csv_rows(processed / "cross_model_robustness.csv")
    repair = _csv_rows(processed / "repair_summary_N3.csv")

    repair_index = {
        (row["fault_class"], row["method"]): row
        for row in repair
    }
    paper = root / "paper" / "main.pdf"

    actual = {
        "tasks_total": len(tasks),
        "tasks_by_family": _counts(tasks, "family"),
        "tasks_by_difficulty": dict(sorted(Counter(
            str(row.get("metadata", {}).get("difficulty", "")) for row in tasks
        ).items())),
        "calibration_cells": len(calibration),
        "natural_trace_count": len(natural_traces),
        "natural_regimes": _counts(natural, "mfp_category"),
        "persistent_trace_count": len(persistent_traces),
        "persistent_regimes": _counts(persistent, "mfp_category"),
        "persistent_distance": _counts(persistent, "fault_to_mfp_distance"),
        "soft_trace_count": len(soft_traces),
        "soft_n2_regimes": _counts(stability, "mfp_category_N2"),
        "soft_n5_regimes": _counts(soft_n5, "mfp_category_N5"),
        "soft_same_regime": sum(row["agreement"] == "1" for row in stability),
        "soft_n2_nontrivial_survivors": sum(
            row["mfp_category_N2"] == "nontrivial_delta_mfp"
            and row["mfp_category_N5"] == "nontrivial_delta_mfp"
            for row in stability
        ),
        "soft_switched_into_nontrivial": sum(
            row["mfp_category_N2"] != "nontrivial_delta_mfp"
            and row["mfp_category_N5"] == "nontrivial_delta_mfp"
            for row in stability
        ),
        "cross_model_regimes": _counts(cross_model, "probe_regime_N3"),
        "cross_model_agreement": sum(
            row["agreement_with_original"] == "1" for row in cross_model
        ),
        "repair_persistent_none": repair_index[
            ("persistent", "none")
        ]["mean_success_rate"],
        "repair_persistent_retry": repair_index[
            ("persistent", "generic_retry")
        ]["mean_success_rate"],
        "repair_soft_none": repair_index[
            ("soft", "none")
        ]["mean_success_rate"],
        "repair_soft_oracle": repair_index[
            ("soft", "oracle_repair")
        ]["mean_success_rate"],
        "paper_sha256": _sha256(paper) if paper.exists() else None,
    }

    expected = {
        "tasks_total": 120,
        "tasks_by_family": {
            "calendar": 30, "file_email": 30, "inventory": 30, "refund": 30,
        },
        "tasks_by_difficulty": {"easy": 40, "hard": 40, "medium": 40},
        "calibration_cells": 152,
        "natural_trace_count": 25,
        "natural_regimes": {
            "nontrivial_delta_mfp": 13, "prefix0": 5, "unstable": 7,
        },
        "persistent_trace_count": 40,
        "persistent_regimes": {"nontrivial_delta_mfp": 40},
        "persistent_distance": {"0": 40},
        "soft_trace_count": 50,
        "soft_n2_regimes": {
            "nontrivial_delta_mfp": 7, "prefix0": 20, "unstable": 23,
        },
        "soft_n5_regimes": {
            "nontrivial_delta_mfp": 7, "prefix0": 22, "unstable": 21,
        },
        "soft_same_regime": 37,
        "soft_n2_nontrivial_survivors": 1,
        "soft_switched_into_nontrivial": 6,
        "cross_model_regimes": {
            "nontrivial_delta_mfp": 5, "unstable": 19,
        },
        "cross_model_agreement": 14,
        "repair_persistent_none": "0.000",
        "repair_persistent_retry": "1.000",
        "repair_soft_none": "0.667",
        "repair_soft_oracle": "0.083",
        "paper_sha256": PAPER_SHA256,
    }

    checks = {
        name: {
            "expected": value,
            "actual": actual.get(name),
            "ok": actual.get(name) == value,
        }
        for name, value in expected.items()
    }
    return {
        "ok": all(check["ok"] for check in checks.values()),
        "paper": (
            "Before the Fall: Delta Minimal Failing Prefixes for Local "
            "Tool-Use Agent Failures"
        ),
        "checks": checks,
    }
