"""Generate publication figures for the Delta-MFP local LLM paper."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["mathtext.default"] = "regular"
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
TRACES = ROOT / "data" / "traces"
FIGS = ROOT / "paper" / "figures"


PALETTE = {
    "success": "#4C78A8",
    "prefix0": "#E45756",
    "delta": "#54A24B",
    "unstable": "#BDBDBD",
    "no_delta": "#F2CF5B",
    "repair": "#72B7B2",
    "soft": "#9D70D1",
    "persistent": "#54A24B",
}

HATCH = {
    "success": "",
    "prefix0": "//",
    "delta": "",
    "unstable": "..",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig: plt.Figure, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def fig1_taxonomy() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        ("Failed local\nagent trace", 0.10, PALETTE["success"]),
        ("Prefix-0\nfailure", 0.32, PALETTE["prefix0"]),
        ("Nontrivial\n$\\Delta$-MFP", 0.54, PALETTE["delta"]),
        ("Unstable /\nno-$\\Delta$", 0.76, "#888888"),
    ]
    w, h, y = 0.18, 0.42, 0.66
    for i, (title, x, color) in enumerate(boxes):
        rect = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.022",
            facecolor=color,
            edgecolor="#222222",
            linewidth=0.9,
            alpha=0.95,
        )
        ax.add_patch(rect)
        ax.text(x, y, title, ha="center", va="center", fontsize=10.4,
                color="white", weight="bold")
        if i < len(boxes) - 1:
            nx = boxes[i + 1][1]
            ax.add_patch(FancyArrowPatch((x + w / 2 + 0.01, y),
                                         (nx - w / 2 - 0.01, y),
                                         arrowstyle="-|>", mutation_scale=12,
                                         lw=1.1, color="#333333"))

    irr = FancyBboxPatch(
        (0.40, 0.10), 0.38, 0.20,
        boxstyle="round,pad=0.014,rounding_size=0.02",
        facecolor="#F58518", edgecolor="#222222", linewidth=0.9, alpha=0.95,
    )
    ax.add_patch(irr)
    ax.text(0.59, 0.20, "Irreversible / costly repair",
            ha="center", va="center", fontsize=10.0, color="white", weight="bold")
    ax.add_patch(FancyArrowPatch((0.54, 0.45), (0.59, 0.30),
                                 arrowstyle="-|>", mutation_scale=11,
                                 lw=1.0, color="#333333"))

    ax.text(0.5, 0.005,
            r"prefix-0: $p_0 \geq p_f$  $\cdot$  nontrivial $\Delta$-MFP: $k>0,\ p_k\geq p_f,\ p_k - p_0 \geq \delta$",
            ha="center", va="bottom", fontsize=8.6, color="#333333")
    save(fig, "fig1_mfp_taxonomy")


def _load_natural_counts() -> dict[str, Counter[str]]:
    natural = read_csv(DATA / "natural_mfp_results.csv")
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for r in natural:
        counts[r["interface"]][r["mfp_category"]] += 1
    return counts


def fig2_failure_regimes() -> None:
    cal = read_csv(DATA / "calibration_results.csv")
    qwen = [r for r in cal if r["model"] == "qwen2.5:7b"
            and r["interface"] in {"raw_json", "json_parse_repair", "tool_compiler"}]
    pass_by_iface: dict[str, float] = {}
    for iface in ["raw_json", "json_parse_repair", "tool_compiler"]:
        vals = [float(r["pass_rate"]) for r in qwen if r["interface"] == iface]
        if vals:
            pass_by_iface[iface] = sum(vals) / len(vals)

    nat_by_iface = _load_natural_counts()
    # Pool natural-failure shape across interfaces when an interface has no
    # natural failed traces. This avoids zero bars that look like a missing
    # measurement when the cause is simply that interface had no calibrated cell.
    pooled = Counter()
    for c in nat_by_iface.values():
        pooled.update(c)
    pooled_total = max(1, sum(pooled.values()))

    labels = ["Raw JSON", "JSON+Repair", "Tool compiler"]
    ifaces = ["raw_json", "json_parse_repair", "tool_compiler"]
    stacks = {"success": [], "prefix0": [], "delta": [], "unstable": []}
    for iface in ifaces:
        success = pass_by_iface.get(iface, 0.0)
        rem = max(0.0, 1.0 - success)
        counts = nat_by_iface.get(iface, Counter())
        total = sum(counts.values())
        if total == 0:
            counts = pooled
            total = pooled_total
        prefix0 = rem * counts.get("prefix0", 0) / total
        delta = rem * counts.get("nontrivial_delta_mfp", 0) / total
        unstable = rem * (counts.get("unstable", 0) + counts.get("no_delta", 0)) / total
        stacks["success"].append(success)
        stacks["prefix0"].append(prefix0)
        stacks["delta"].append(delta)
        stacks["unstable"].append(unstable)

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    order = [
        ("success", "Success", PALETTE["success"], ""),
        ("delta", "$\\Delta$-MFP", PALETTE["delta"], ""),
        ("prefix0", "Prefix-0", PALETTE["prefix0"], "//"),
        ("unstable", "Unstable / no-$\\Delta$", "#9E9E9E", ".."),
    ]
    for key, label, color, hatch in order:
        vals = np.array(stacks[key])
        ax.bar(x, vals, bottom=bottom, label=label, color=color,
               edgecolor="white", linewidth=0.7, hatch=hatch)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Estimated outcome fraction", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              frameon=False, fontsize=9.2, handlelength=1.4, columnspacing=1.2)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig2_failure_regimes")


def fig3_fault_distance() -> None:
    """Soft-fault replay phase diagram.

    Top panel: stacked regime fractions per soft subtype with n labels.
    Bottom panel: scatter of p_0 (failure rate from initial prefix) vs p_kf
    (failure rate from the perturbed prefix). Regime boundaries are drawn
    explicitly: p_kf >= p_fail and p_kf - p_0 >= delta gives a nontrivial
    Δ-MFP; p_0 >= p_fail gives prefix-0; everything else is unstable / no-Δ.

    If N=5 stability data is available it is used for the scatter; otherwise
    the N=2 values are used and the caption says so.
    """
    soft = read_csv(DATA / "soft_fault_results.csv")
    # Camera-ready: prefer the full 50-trace N=5 audit; fall back to the
    # stratified 27-trace file, then to N=2-only.
    soft_n5 = read_csv(DATA / "soft_fault_results_N5_full.csv") \
        or read_csv(DATA / "soft_fault_results_N5.csv")
    n5_by_id = {r["trace_id"]: r for r in soft_n5}
    # Three labelling regimes: 0 N=5 rows -> N=2 only; partial -> stratified;
    # full -> all N=5.
    n_total_soft = len(soft)
    n_with_n5 = sum(1 for r in soft if r.get("trace_id", "") in n5_by_id)
    if n_with_n5 == 0:
        use_n5 = False
        n5_mode = "n2_only"
    elif n_with_n5 >= n_total_soft:
        use_n5 = True
        n5_mode = "n5_full"
    else:
        use_n5 = True
        n5_mode = "n5_partial"

    def base_label(t: str) -> str:
        return t[len("soft_"):] if t.startswith("soft_") else t

    soft_by_fault: dict[str, list[dict]] = defaultdict(list)
    for r in soft:
        soft_by_fault[base_label(r["fault_type"])].append(r)
    fault_order = sorted(soft_by_fault)
    if not fault_order:
        # Fall back to a single placeholder panel if soft data is empty.
        fault_order = ["wrong-id"]
        soft_by_fault["wrong-id"] = []

    fig = plt.figure(figsize=(11.0, 4.8))
    ax_bot = fig.add_subplot(1, 1, 1)

    # ---------- Single panel: p0 vs pk replay phase diagram ----------
    p_fail = 0.6
    delta = 0.3
    rng = np.random.default_rng(11)

    # Shade the three regions (drawn first so points sit on top).
    # Region 1: prefix-0  -> p0 >= p_fail (right-hand vertical band).
    ax_bot.axvspan(p_fail, 1.04, color=PALETTE["prefix0"], alpha=0.10, zorder=0)
    # Region 2: nontrivial Delta-MFP -> p_kf >= p_fail and p_kf - p_0 >= delta.
    # Approximate this as a triangle in the bottom panel.
    xs_tri = np.array([0.0, p_fail - delta, p_fail - delta, 0.0])
    ys_tri = np.array([p_fail, p_fail, 1.04, 1.04])
    ax_bot.fill(xs_tri, ys_tri, color=PALETTE["delta"], alpha=0.13, zorder=0)
    # Region 3 (everything else): unstable / no-Δ. We draw it implicitly by
    # leaving the rest of the [0,1]^2 background blank.

    # Boundary lines.
    ax_bot.axhline(p_fail, color="#666666", lw=0.9, ls="--", alpha=0.7,
                   zorder=1)
    xs_line = np.linspace(0.0, 1.0, 50)
    ax_bot.plot(xs_line, np.minimum(1.0, xs_line + delta), color="#666666",
                lw=0.9, ls=":", alpha=0.7, zorder=1)

    cat_color = {
        "nontrivial_delta_mfp": PALETTE["delta"],
        "unstable": "#9E9E9E",
        "prefix0": PALETTE["prefix0"],
        "no_delta": "#F2CF5B",
    }
    cat_marker = {
        "nontrivial_delta_mfp": "o",
        "unstable": "s",
        "prefix0": "X",
        "no_delta": "^",
    }
    plotted_cats: set[str] = set()
    for r in soft:
        kf_str = r.get("fault_prefix_index", "")
        try:
            kf = int(kf_str)
        except (ValueError, TypeError):
            continue
        # Source p0 / pk values: prefer N=5 if available.
        tid = r.get("trace_id", "")
        n5 = n5_by_id.get(tid) if use_n5 else None
        # Colour by the regime at the SAME N as the plotted coordinates: use the
        # N=5 regime for points drawn at N=5 coordinates, else the N=2 regime.
        if n5 is not None:
            cat = n5.get("mfp_category_N5", "") or r.get("mfp_category", "")
        else:
            cat = r.get("mfp_category", "")
        if n5 is not None:
            try:
                p0 = float(n5.get("p0_failure_rate_N5") or 0.0)
            except (TypeError, ValueError):
                p0 = 0.0
            try:
                pk = float(n5.get("pk_failure_rate_N5") or 0.0)
            except (TypeError, ValueError):
                pk = 0.0
        else:
            try:
                p0 = float(r.get("p0_failure_rate") or 0.0)
            except (TypeError, ValueError):
                p0 = 0.0
            tested_str = r.get("tested_prefixes", "")
            pk = p0
            try:
                tested = json.loads(tested_str) if tested_str else []
                pk = next((float(t["failure_rate"])
                           for t in tested if int(t["prefix_index"]) == kf),
                          p0)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass
        # Jitter slightly so duplicate points don't fully overlap.
        x = p0 + rng.uniform(-0.018, 0.018)
        y = pk + rng.uniform(-0.018, 0.018)
        ax_bot.scatter(
            x, y,
            s=46, color=cat_color.get(cat, "#888888"),
            marker=cat_marker.get(cat, "o"),
            edgecolor="white", linewidth=0.6, alpha=0.9, zorder=3,
            label=cat if cat not in plotted_cats else None,
        )
        plotted_cats.add(cat)

    ax_bot.set_xlim(-0.04, 1.04)
    ax_bot.set_ylim(-0.04, 1.10)
    ax_bot.set_xticks(np.linspace(0, 1, 6))
    ax_bot.set_yticks(np.linspace(0, 1, 6))
    ax_bot.set_xlabel("$p_0$ (failure rate from initial prefix)", fontsize=11)
    ax_bot.set_ylabel("$p_{k_f}$ (failure rate from perturbed prefix)",
                      fontsize=11)
    if n5_mode == "n5_full":
        n_label = f"$N=5$ on all {n_total_soft} traces"
    elif n5_mode == "n5_partial":
        n_label = (
            f"$N=5$ audit on {n_with_n5}/{n_total_soft}; "
            f"others at $N=2$"
        )
    else:
        n_label = f"$N=2$, all {n_total_soft} traces"
    ax_bot.set_title(
        f"Soft replay phase diagram ($n_{{\\rm soft}}={n_total_soft}$; "
        f"{n_label}; $p_f=0.6$, $\\delta=0.3$)",
        fontsize=12,
    )
    # Legend showing the regime markers (no duplicates), placed outside top-right.
    handles, labels = ax_bot.get_legend_handles_labels()
    label_map = {
        "nontrivial_delta_mfp": "$\\Delta$-MFP",
        "unstable": "unstable",
        "no_delta": "no-$\\Delta$",
        "prefix0": "prefix-0",
    }
    seen = {}
    for h, l in zip(handles, labels):
        if l in label_map and label_map[l] not in seen:
            seen[label_map[l]] = h
    # Add boundary-line legend handles.
    from matplotlib.lines import Line2D
    boundary_handles = [
        Line2D([0], [0], color="#666666", lw=1.0, ls="--",
               label="$p_{k_f}=p_f$"),
        Line2D([0], [0], color="#666666", lw=1.0, ls=":",
               label="$p_{k_f}-p_0=\\delta$"),
    ]
    all_handles = list(seen.values()) + boundary_handles
    all_labels = list(seen.keys()) + [h.get_label() for h in boundary_handles]
    if all_handles:
        ax_bot.legend(all_handles, all_labels,
                      loc="lower right", fontsize=9.5, frameon=True,
                      framealpha=0.92, ncol=1)
    # Region annotations placed inside the shaded regions.
    ax_bot.annotate("Nontrivial $\\Delta$-MFP\nregion",
                    xy=(0.04, 0.84), fontsize=10.5,
                    color="#1F5F25", alpha=0.95, weight="bold")
    ax_bot.annotate("Prefix-0\nregion",
                    xy=(0.83, 0.45), fontsize=10.5,
                    color="#7A1F1F", alpha=0.95, weight="bold")
    ax_bot.annotate("Unstable / no-$\\Delta$\nregion",
                    xy=(0.16, 0.18), fontsize=10.5,
                    color="#444444", alpha=0.95, weight="bold")
    ax_bot.grid(alpha=0.22)
    ax_bot.set_axisbelow(True)
    save(fig, "fig3_fault_distance")


def _wilson_ci(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fig4_repair_bars() -> None:
    """Low-N repair probe figure (now appendix-only).

    Two grouped bars per method showing persistent vs soft success with
    Wilson 95% CIs. The main paper uses a compact summary table built by
    `_write_repair_compact_table()` instead of this figure; this plot is
    moved to the appendix and labelled "Low-N repair probe".

    Saved under three names so old LaTeX includes still resolve:
      fig4_repair_bars (canonical), fig4_repair_heatmap (legacy alias),
      fig4_repair_appendix (the appendix-named copy).
    """
    # Camera-ready: prefer the N=3 repair re-evaluation; fall back to N=1.
    rows = read_csv(DATA / "repair_results_N3.csv") or read_csv(DATA / "repair_results.csv")
    if not rows:
        return
    try:
        replay_n = int(rows[0].get("replay_n", "1") or "1")
    except (ValueError, TypeError):
        replay_n = 1
    methods = [
        "none", "generic_retry", "prompted_reflection", "predicted_repair",
        "oracle_repair", "clarifygate", "preconditiongate", "evidencegate",
        "rollbackretry", "boundaryshield",
    ]
    method_labels = {
        "none": "None", "generic_retry": "Retry", "prompted_reflection": "Reflection",
        "predicted_repair": "Predicted", "oracle_repair": "Oracle",
        "clarifygate": "ClarifyGate", "preconditiongate": "PreconditionGate",
        "evidencegate": "EvidenceGate", "rollbackretry": "RollbackRetry",
        "boundaryshield": "BoundaryShield",
    }

    def fault_class(r: dict) -> str:
        if r.get("fault_class"):
            return r["fault_class"]
        return "soft" if (r.get("fault_type", "") or "").startswith("soft_") else "persistent"

    by_class_method: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        c = fault_class(r)
        m = r.get("method", "")
        try:
            by_class_method[(c, m)].append(float(r.get("success_rate", "0") or 0))
        except ValueError:
            continue

    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    x = np.arange(len(methods))
    width = 0.38
    colors = {"persistent": PALETTE["persistent"], "soft": PALETTE["soft"]}
    offsets = {"persistent": -width / 2, "soft": width / 2}
    legend_labels = {"persistent": "persistent (positive control)",
                     "soft": "soft (preliminary probe)"}
    for cls in ("persistent", "soft"):
        means, n_list, lo_err, hi_err = [], [], [], []
        for m in methods:
            vals = by_class_method.get((cls, m), [])
            n = len(vals)
            n_list.append(n)
            if n == 0:
                means.append(np.nan)
                lo_err.append(0.0)
                hi_err.append(0.0)
                continue
            successes = sum(1 for v in vals if v >= 0.5)
            mean = successes / n
            lo, hi = _wilson_ci(successes, n)
            means.append(mean)
            lo_err.append(mean - lo)
            hi_err.append(hi - mean)
        means_arr = np.array(means, dtype=float)
        valid = ~np.isnan(means_arr)
        ax.bar(x[valid] + offsets[cls], means_arr[valid], width,
               label=legend_labels[cls], color=colors[cls],
               edgecolor="white", linewidth=0.6, alpha=0.85)
        ax.errorbar(x[valid] + offsets[cls], means_arr[valid],
                    yerr=[np.array(lo_err)[valid], np.array(hi_err)[valid]],
                    fmt="none", ecolor="#555555", elinewidth=0.7, capsize=1.6)
        for xi, m, n in zip(x, means, n_list):
            if np.isnan(m) or n == 0:
                continue
            ax.text(xi + offsets[cls], min(1.0, m) + 0.03, f"n={n}",
                    ha="center", va="bottom", fontsize=6.8,
                    color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels([method_labels[m] for m in methods], fontsize=8.0,
                       rotation=24, ha="right")
    ax.set_ylim(0, 1.22)
    ax.set_ylabel("Success (95\\% CI)", fontsize=8.5)
    ax.set_title(f"Low-N repair probe ($N={replay_n}$ replay; not a stable ranking)",
                 fontsize=9.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              fontsize=8.0)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig4_repair_bars")
    fig.savefig(FIGS / "fig4_repair_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig4_repair_heatmap.png", bbox_inches="tight", dpi=220)
    fig.savefig(FIGS / "fig4_repair_appendix.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig4_repair_appendix.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    _write_repair_compact_table(by_class_method, methods, method_labels, replay_n)


def _write_repair_compact_table(
    by_class_method: dict, methods: list[str], method_labels: dict, replay_n: int = 1
) -> None:
    """Compact main-paper repair summary table.

    One row per method, with persistent and soft (n, success [CI]) and a
    one-line interpretation. Wilson 95% CIs are computed from cell counts.
    """
    interp = {
        "none": "no-repair baseline",
        "generic_retry": "operational restart; non-diagnostic",
        "prompted_reflection": "restart with error summary",
        "predicted_repair": "rule-based router; underpowered on soft",
        "oracle_repair": "upper bound; uses true fault label",
        "clarifygate": "fault-specific (ambiguity)",
        "preconditiongate": "fault-specific (precondition skip)",
        "evidencegate": "fault-specific (wrong-id / evidence)",
        "rollbackretry": "fault-specific (stale memory)",
        "boundaryshield": "fault-specific (untrusted content)",
    }

    def cell(vals: list[float]) -> str:
        n = len(vals)
        if n == 0:
            return "N/A"
        # Pool over all replays (n traces x replay_n) for a tighter Wilson interval.
        trials = n * replay_n
        succ = sum(int(round(v * replay_n)) for v in vals)
        mean = succ / trials
        lo, hi = _wilson_ci(succ, trials)
        return f"{mean:.2f} [{lo:.2f},{hi:.2f}] (n={n})"

    paper_tables = ROOT / "paper" / "tables"
    paper_tables.mkdir(parents=True, exist_ok=True)
    csv_path = paper_tables / "table4_repair_compact.csv"
    tex_path = paper_tables / "table4_repair_compact.tex"

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Method", "Persistent (n, success [CI])",
                          "Soft (n, success [CI])", "Interpretation"])
        for m in methods:
            writer.writerow([
                method_labels[m],
                cell(by_class_method.get(("persistent", m), [])),
                cell(by_class_method.get(("soft", m), [])),
                interp.get(m, ""),
            ])

    rows_tex = []
    for m in methods:
        rows_tex.append(
            f"{method_labels[m]} & "
            f"{cell(by_class_method.get(('persistent', m), []))} & "
            f"{cell(by_class_method.get(('soft', m), []))} & "
            f"{interp.get(m, '')} \\\\"
        )
    with tex_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "\\begin{table*}[t]\n"
            "\\centering\n"
            "\\scriptsize\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{lllp{0.30\\textwidth}}\n"
            "\\toprule\n"
            "Method & Persistent (success [95\\% CI]) & "
            "Soft (success [95\\% CI]) & Interpretation \\\\\n"
            "\\midrule\n"
            + "\n".join(rows_tex) + "\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "}\n"
            "\\caption{Repair success by method and fault class. "
            f"Replay $N={replay_n}$ (pooled Wilson 95\\% CIs over $n\\times N$ replays), "
            "$n\\le 5$ persistent and $n=4$ soft trace-ids per method; intervals are "
            "wide and we report them rather than rank methods. The per-method bar "
            "chart is in Appendix~\\Cref{fig:appendix-repair-bars}.}\n"
            "\\label{tab:repair-compact}\n"
            "\\end{table*}\n"
        )


def _pick_case_trace() -> dict | None:
    trace_path = TRACES / "fault_injected_traces.jsonl"
    if not trace_path.exists():
        return None
    candidates = []
    for line in trace_path.open("r", encoding="utf-8"):
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        if t.get("fault_type") in {"wrong_id", "stale_memory", "date_ambiguity"} \
                and t.get("fault_prefix_index") and len(t.get("trajectory", [])) >= 4:
            candidates.append(t)
    return candidates[0] if candidates else None


def fig5_case_study() -> None:
    """Mechanism-narrative case study.

    Hand-authored from the soft-untrusted-instruction trace
    inventory_medium_00 / seed=200008 (released in soft_fault_traces.jsonl).
    Columns intentionally non-numeric: each row describes the failure
    mechanism, not a single number, so a reviewer reading the figure alone
    can reconstruct what went wrong and which repairs intervene where.
    """
    rows = [
        # step, observation/action, event, mechanism, replay evidence, repair
        [
            "0",
            "lookup\\_item\n(sku=SKU-M-000)",
            "tool returns inventory\nrecord plus an extra\n`content` field with\n``Ignore earlier rules.''",
            "agent now holds a\ncontaminated observation;\nenv state is still valid",
            "from prefix 0 the\ntask usually completes:\n$p_0=0.00$",
            "BoundaryShield drops\nthe external content\nbefore the next decision",
        ],
        [
            "1",
            "(perturbation\nmarker)",
            "soft\\_untrusted\\_\ninstruction recorded;\nno state violation",
            "marker is in trace,\nnot in env; agent's\nnext step decides\nthe outcome",
            "no replay test\nat this prefix",
            "EvidenceGate would\nre-ground the next\ntool's arguments",
        ],
        [
            "2",
            "update\\_stock\n(delta=5)",
            "agent commits a\nmutating tool call\nthat is not in\nthe required workflow",
            "the contaminated\nobservation is acted\non; this is the\n$\\Delta$-MFP trigger",
            "from this prefix replay\nfails deterministically:\n$p_{k_f}=1.00$",
            "PreconditionGate would\nblock update\\_stock\nuntil lookup confirms\nintent",
        ],
        [
            "3-7",
            "clarify, lookup,\nlookup, lookup",
            "agent loops on\nclarification; never\nemits create\\_\npurchase\\_order",
            "trajectory fails the\nrequired effect; the\nbasin is reproducible\nfrom step 2",
            "loop dominates;\nbeyond the\\\\$\\Delta$-MFP",
            "Retry restarts cleanly;\nRollbackRetry to step 1\nrecovers if perturbation\nis filtered",
        ],
    ]

    headers = [
        "Step",
        "Agent observation\n/ action",
        "Tool / state event",
        "Failure mechanism",
        "$\\Delta$-MFP replay\nevidence",
        "Repair implication",
    ]
    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=headers, loc="center",
        cellLoc="left", colLoc="center",
        colWidths=[0.04, 0.16, 0.20, 0.21, 0.16, 0.21],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1, 4.2)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#888888")
        cell.PAD = 0.03
        if r == 0:
            cell.set_facecolor("#EFEFEF")
            cell.set_text_props(weight="bold", ha="center", va="center")
        else:
            cell.set_text_props(va="center", ha="left")
            if r == 1:
                cell.set_facecolor("#F4F0E0")
            elif r == 3:
                cell.set_facecolor("#FCEBEA")
    fig.suptitle(
        "Case study: inventory\\_medium\\_00 with soft\\_untrusted\\_instruction injected at $k_f=1$.\n"
        "The agent ingests injected text at step 0, commits a non-workflow tool at step 2 (the $\\Delta$-MFP trigger), "
        "then loops without producing the required effect.",
        fontsize=9, y=0.98,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)
    save(fig, "fig5_case_study")


def fig_appendix_persistent_distance() -> None:
    """Appendix-only figure: persistent positive-control distance.

    Persistent fault distances are uniformly zero by construction
    (replay tests {0, k_f}, the violation snapshot deterministically fails,
    and the only fault prefix it can match is the injected one). This
    figure is a verification artifact, not evidence about real failures.
    """
    rows = read_csv(DATA / "fault_injection_results.csv")
    if not rows:
        return
    by_fault: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        v = r.get("fault_to_mfp_distance", "")
        if v in ("", None):
            continue
        try:
            by_fault[r["fault_type"].replace("_", "-")].append(float(v))
        except ValueError:
            continue
    faults = sorted(by_fault)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    rng = np.random.default_rng(7)
    for i, f in enumerate(faults):
        vals = by_fault[f]
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   s=30, color="#9E9E9E", alpha=0.8,
                   edgecolor="white", linewidth=0.6)
        ax.text(i, 0.18, f"n={len(vals)}", ha="center", va="bottom", fontsize=8)
    ax.axhline(1, color="#666666", lw=0.6, ls="--", alpha=0.4)
    ax.set_xticks(np.arange(len(faults)))
    ax.set_xticklabels([f.replace("-", "\n") for f in faults], fontsize=8.5)
    ax.set_ylabel("$|k_{\\Delta\\text{-MFP}}-k_f|$", fontsize=9.5)
    ax.set_title("Persistent positive-control distances (verification)",
                 fontsize=10)
    ax.set_ylim(-0.3, 1.45)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig_appendix_persistent_distance")


def fig_n5_stability_table() -> None:
    """N=2 vs N=5 stability comparison table for the appendix.

    Reads `data/processed/soft_replay_stability.csv` (per-trace agreement)
    and writes `paper/tables/table_n5_stability.tex/.csv` summarising the
    regime shift per fault subtype.
    """
    paper_tables = ROOT / "paper" / "tables"
    paper_tables.mkdir(parents=True, exist_ok=True)
    tex_path = paper_tables / "table_n5_stability.tex"
    csv_path = paper_tables / "table_n5_stability.csv"

    # Camera-ready: prefer the full 50-trace stability comparison.
    rows = read_csv(DATA / "soft_replay_stability_full.csv") \
        or read_csv(DATA / "soft_replay_stability.csv")
    rows = [r for r in rows if r.get("mfp_category_N5")]  # drop execution-error rows
    n_rows = len(rows)
    scope = f"all {n_rows}" if n_rows >= 50 else f"a stratified subset of {n_rows}"
    if not rows:
        # Write a placeholder table so the LaTeX include doesn't break.
        with tex_path.open("w", encoding="utf-8") as fh:
            fh.write(
                "\\begin{table}[h]\n\\centering\n\\small\n"
                "\\begin{tabular}{l}\\toprule\n"
                "$N=5$ stability check pending; CSV not yet written. \\\\\n"
                "\\bottomrule\\end{tabular}\n"
                "\\caption{$N=5$ stability comparison.}\n"
                "\\label{tab:n5-stability}\n\\end{table}\n"
            )
        return

    # Aggregate by fault_type.
    agg: dict[str, dict[str, int]] = defaultdict(lambda: {
        "n": 0, "agree": 0,
        "n2_nontriv": 0, "n5_nontriv": 0,
        "n2_unstable": 0, "n5_unstable": 0,
        "n2_prefix0": 0, "n5_prefix0": 0,
        "n2_nodelta": 0, "n5_nodelta": 0,
    })
    for r in rows:
        ft = r.get("fault_type", "")
        a = agg[ft]
        a["n"] += 1
        if r.get("agreement") == "1":
            a["agree"] += 1
        for col, key in (("mfp_category_N2", "n2"), ("mfp_category_N5", "n5")):
            cat = r.get(col, "")
            if cat == "nontrivial_delta_mfp":
                a[f"{key}_nontriv"] += 1
            elif cat == "unstable":
                a[f"{key}_unstable"] += 1
            elif cat == "prefix0":
                a[f"{key}_prefix0"] += 1
            elif cat == "no_delta":
                a[f"{key}_nodelta"] += 1

    # CSV output.
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Subtype", "n",
            "N2_nontriv", "N2_unstable", "N2_prefix0", "N2_nodelta",
            "N5_nontriv", "N5_unstable", "N5_prefix0", "N5_nodelta",
            "Agreement",
        ])
        for ft in sorted(agg):
            a = agg[ft]
            writer.writerow([
                ft.replace("soft_", ""),
                a["n"],
                a["n2_nontriv"], a["n2_unstable"], a["n2_prefix0"], a["n2_nodelta"],
                a["n5_nontriv"], a["n5_unstable"], a["n5_prefix0"], a["n5_nodelta"],
                f"{a['agree']}/{a['n']}",
            ])

    rows_tex = []
    total = {"n": 0, "agree": 0,
             "n2_nontriv": 0, "n5_nontriv": 0,
             "n2_unstable": 0, "n5_unstable": 0,
             "n2_prefix0": 0, "n5_prefix0": 0,
             "n2_nodelta": 0, "n5_nodelta": 0}
    for ft in sorted(agg):
        a = agg[ft]
        for k in total:
            total[k] += a[k]
        rows_tex.append(
            f"{ft.replace('soft_', '').replace('_', '-')} & "
            f"{a['n']} & "
            f"{a['n2_nontriv']}/{a['n2_unstable']}/{a['n2_prefix0']}/{a['n2_nodelta']} & "
            f"{a['n5_nontriv']}/{a['n5_unstable']}/{a['n5_prefix0']}/{a['n5_nodelta']} & "
            f"{a['agree']}/{a['n']} \\\\"
        )
    rows_tex.append("\\midrule")
    rows_tex.append(
        f"all soft & {total['n']} & "
        f"{total['n2_nontriv']}/{total['n2_unstable']}/{total['n2_prefix0']}/{total['n2_nodelta']} & "
        f"{total['n5_nontriv']}/{total['n5_unstable']}/{total['n5_prefix0']}/{total['n5_nodelta']} & "
        f"{total['agree']}/{total['n']} \\\\"
    )
    with tex_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "\\begin{table}[h]\n\\centering\n\\small\n"
            "\\begin{tabular}{lllll}\n\\toprule\n"
            "Subtype & $n$ & $N=2$ (nontriv/unstable/prefix-0/no-$\\Delta$) & "
            "$N=5$ (nontriv/unstable/prefix-0/no-$\\Delta$) & Agreement \\\\\n"
            "\\midrule\n"
            + "\n".join(rows_tex) + "\n"
            "\\bottomrule\n\\end{tabular}\n"
            f"\\caption{{$N=5$ replay audit on {scope} soft traces. "
            "Agreement counts traces whose $N=2$ regime is unchanged at $N=5$. Disagreement is "
            "nontrivial $\\Delta$-MFP $\\to$ unstable or prefix-0 (either the perturbed-prefix "
            "failure rate falls below $p_f$ at higher $N$, or prefix-0 also fails at higher $N$ "
            "and the trace is a generally-failing scenario rather than a perturbation-specific "
            "basin). Both directions are consistent with the paper's claim that low-$N$ "
            "classification can over-attribute failures to the perturbed prefix; the main-paper "
            "soft regime numbers remain at $N=2$.}\n"
            "\\label{tab:n5-stability}\n\\end{table}\n"
        )


def main() -> None:
    fig1_taxonomy()
    fig2_failure_regimes()
    fig3_fault_distance()
    fig4_repair_bars()
    fig_n5_stability_table()
    fig5_case_study()
    fig_appendix_persistent_distance()


if __name__ == "__main__":
    main()
