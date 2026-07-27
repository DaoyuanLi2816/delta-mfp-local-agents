"""Metric aggregation for MFP experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import csv
import math


JsonDict = Dict[str, Any]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Dict[str, float]:
    if n < 0 or successes < 0 or successes > n:
        raise ValueError("wilson_ci requires 0 <= successes <= n")
    if z <= 0:
        raise ValueError("wilson_ci requires z > 0")
    if n == 0:
        return {"lo": 0.0, "hi": 0.0}
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return {"lo": max(0.0, center - margin), "hi": min(1.0, center + margin)}


def write_csv(path: Path, rows: List[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_latex_table(path: Path, headers: List[str], rows: List[List[Any]], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colspec = "l" * len(headers)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(latex_escape(h) for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(x) for x in row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)
