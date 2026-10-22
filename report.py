"""Generate MACPE-K figures and placement tables from recorded runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from numbers import Real
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
from scipy.stats import t

from experiments import (
    timing_inputs as inputs,
    timing_planner_count as planner_count,
    physical_completion_view,
    read_json,
    rounded,
    write_json,
    SAFETY_METRICS,
)
from theory import pathwise

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TEXT_WIDTH = 516 / 72.27
COLUMN_WIDTH = 252 / 72.27
COST_CONTROLLERS = (("sequential", 1), ("macpe", 1), ("macpe", 3))


def interval(values):
    n = len(values)
    mean = statistics.mean(values)
    half = float(t.ppf(0.975, n - 1)) * statistics.stdev(values) / math.sqrt(n) if n > 1 else None
    return dict(
        n=n,
        mean=mean,
        half_width=half,
        ci_low=mean - half if half is not None else None,
        ci_high=mean + half if half is not None else None,
    )


def cost_interval(values):
    result = interval(values)
    if result["half_width"] is None:
        result.update(half_width=0.0, ci_low=result["mean"], ci_high=result["mean"])
    return result


# Figure styles and page bounds.
RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#DEDEDE",
    "grid.linewidth": 0.4,
    "axes.titlepad": 3.0,
    "axes.labelpad": 2.0,
    "axes.titlelocation": "left",
    "lines.linewidth": 1.0,
    "lines.markersize": 4.2,
    "lines.markeredgewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.pad": 1.8,
    "ytick.major.pad": 1.8,
    "legend.frameon": False,
    "legend.handlelength": 2.0,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.2,
    "errorbar.capsize": 0,
    "figure.dpi": 150,
    "savefig.dpi": 600,
}


def configure(**overrides):
    """Set the common figure style, with per-figure overrides."""
    plt.rcParams.update({**RC, **overrides})


def verify_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [
        item
        for item in fig.findobj(Text)
        if item.get_visible()
        and item.get_text().strip()
        and item.get_window_extent(renderer).width > 0
    ]
    outside = [
        item.get_text()
        for item in texts
        if not fig.bbox.contains(*item.get_window_extent(renderer).p0)
        or not fig.bbox.contains(*item.get_window_extent(renderer).p1)
    ]
    if outside:
        raise ValueError(f"Text outside the exported page: {outside}")
    if min(item.get_fontsize() for item in texts) < 7:
        raise ValueError("Figure text must be at least 7 pt at figure layout width.")
    return dict(
        all_text_inside_page=True, minimum_font_size_pt=min(item.get_fontsize() for item in texts)
    )


# A: timing figures.

STYLES = {
    "sequential": ("#C62828", "o", "-", "Sequential MAPE-K"),
    "macpe1": ("#191919", "s", "--", r"MACPE-K ($k_P=1$)"),
    "macpe2": ("#1565C0", "D", "-.", r"MACPE-K ($k_P=2$)"),
    "macpe3": ("#DA7900", "v", ":", r"MACPE-K ($k_P=3$)"),
    "sufficient": ("#247B39", "^", "-", "MACPE-K (sufficient Planning capacity)"),
}

FIXED = ("sequential", "macpe1", "sufficient")

VARIABLE = ("sequential", "macpe1", "macpe2", "macpe3", "sufficient")


def handles(controllers):
    return [
        Line2D(
            [],
            [],
            color=STYLES[c][0],
            marker=STYLES[c][1],
            linestyle=STYLES[c][2],
            linewidth=1,
            markersize=3.2,
            markerfacecolor="white",
            markeredgewidth=0.7,
            label=STYLES[c][3],
        )
        for c in controllers
    ]


def ranges(rows, envelope):
    if envelope == "requests":
        return ([r["request_TW_min"] for r in rows], [r["request_TW_max"] for r in rows])
    return ([r["mean_TW"]["min"] for r in rows], [r["mean_TW"]["max"] for r in rows])


def observed(ax, rows, x, controller, envelope, *, extrema=False, confidence=False):
    if not rows:
        return
    color, marker, _, _ = STYLES[controller]
    mean = [r["mean_TW"]["mean"] for r in rows]
    if extrema:
        low, high = ranges(rows, envelope)
        ax.errorbar(
            x,
            mean,
            yerr=[np.array(mean) - low, np.array(high) - mean],
            fmt="none",
            ecolor=color,
            alpha=0.30,
            elinewidth=0.55,
            capsize=1.7,
            capthick=0.55,
            zorder=1,
        )
    err = (
        [max(0, (r["mean_TW"]["ci_high"] or m) - m) for r, m in zip(rows, mean)]
        if confidence
        else None
    )
    # Draw sufficient capacity last and above the axis, including its zero line.
    ax.errorbar(
        x,
        mean,
        yerr=err,
        fmt=marker,
        color=color,
        ms=3.3 if controller == "sufficient" else 2.5,
        mfc="white",
        mew=0.65,
        elinewidth=0.65,
        capsize=1.2,
        zorder=7 if controller == "sufficient" else 5,
    )


def dense_fixed():
    """Integer-duration references preserve discontinuities between simulations."""
    curves = []
    for window in (0, 100):
        for arrival in ("periodic", "poisson"):
            arrivals = [
                inputs(
                    dict(
                        family="fixed", arrival=arrival, planning=25, execution=25, replication=rep
                    )
                )[1]
                for rep in ((1,) if arrival == "periodic" else range(1, 31))
            ]
            for controller in FIXED:
                means = []
                for p in range(25, 501):
                    k = planner_count(
                        dict(controller=controller, window=window, arrival=arrival, planning=p)
                    )
                    means.append(
                        statistics.mean(
                            pathwise(
                                a, [p] * 200, [[p] * 4] * 200, k, window, controller == "sequential"
                            )["mean_TW"]
                            for a in arrivals
                        )
                    )
                curves.append(
                    dict(
                        window=window,
                        arrival=arrival,
                        controller=controller,
                        planning=list(range(25, 501)),
                        mean_TW=means,
                        finite_input_traces=len(arrivals),
                    )
                )
    return curves


def finish(fig, path):
    """Check the layout, save the PDF and a PNG preview, and close the figure."""
    check = verify_layout(fig)
    fig.savefig(
        path, metadata={"Creator": "MACPE-K simulation", "CreationDate": None, "ModDate": None}
    )
    fig.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(fig)
    return dict(**check, file=path.name)


def fixed_figure(summary, theory, curves, figures, envelope):
    fig = plt.figure(figsize=(TEXT_WIDTH, 2.65))
    grid = fig.add_gridspec(
        2, 2, left=0.065, right=0.61, bottom=0.14, top=0.86, hspace=0.60, wspace=0.24
    )
    for row, window in enumerate((0, 100)):
        for col, arrival in enumerate(("periodic", "poisson")):
            ax = fig.add_subplot(grid[row, col])
            for controller in FIXED:
                color, _, line, _ = STYLES[controller]
                curve = next(
                    c
                    for c in curves
                    if c["window"] == window
                    and c["arrival"] == arrival
                    and c["controller"] == controller
                )
                ax.plot(
                    [2 * p for p in curve["planning"]],
                    curve["mean_TW"],
                    color=color,
                    ls=line,
                    lw=1.05,
                    zorder=4 if controller == "sufficient" else 3,
                    drawstyle="steps-post" if window else "default",
                )
                rows = sorted(
                    [
                        r
                        for r in summary["summary"]
                        if r["family"] == "fixed"
                        and r["window"] == window
                        and r["arrival"] == arrival
                        and r["controller"] == controller
                    ],
                    key=lambda r: r["planning"],
                )
                for r in rows:
                    assert abs(curve["mean_TW"][r["planning"] - 25] - r["reference_TW"]) <= 0.0101
                observed(
                    ax,
                    rows,
                    [2 * r["planning"] for r in rows],
                    controller,
                    envelope,
                    extrema=True,
                    confidence=True,
                )
            load = r"$\Pi=2$" if arrival == "periodic" else r"Poisson $\lambda=1/120$"
            title = f"({'abde'[row*2+col]}) " + (
                ("Event-driven; " if not window else "Window-based; ") + load if col == 0 else load
            )
            ax.set_title(title, loc="left", fontsize=7.5)
            ax.set_xlabel(r"$T_G=T_P+T_E$", labelpad=0)
            ax.set_ylabel(r"Mean $T_W$" if col == 0 else "", labelpad=1)
            if col:
                ax.tick_params(labelleft=False)
            ax.set_xlim(35, 1015)
            ax.set_xticks([50, 500, 1000])
            # Visible zero values; no shift is applied to observations.
            ax.set_ylim(-30, 1030)
            ax.set_yticks([0, 500, 1000])
    for col, window in enumerate((0, 100)):
        # Fixed-size 3D axes keep the spacing between panels even.
        side = 1.00
        width, height = side / TEXT_WIDTH, side / 2.65
        center_y = (0.73, 0.29)[col]
        ax = fig.add_axes(
            [0.765 - width / 2, center_y - height / 2, width, height],
            projection="3d",
            computed_zorder=False,
        )
        # Transparent axes leave the neighbouring panel's labels visible.
        ax.patch.set_alpha(0)
        pp, ee = np.meshgrid(np.arange(50, 501, 50), np.arange(50, 501, 50))
        for controller in FIXED:
            color, marker, _, _ = STYLES[controller]
            refs = {
                (r["T_P"], r["T_E"]): r
                for r in theory["surface_references"]
                if r["window"] == window and r["controller"] == controller
            }
            zz = np.array(
                [refs[int(p), int(e)]["mean_TW"] for p, e in zip(pp.ravel(), ee.ravel())]
            ).reshape(pp.shape)
            rows = [
                r
                for r in summary["summary"]
                if r["family"] == "surface"
                and r["window"] == window
                and r["controller"] == controller
            ]
            ax.plot_wireframe(
                pp,
                ee,
                zz,
                color=color,
                linewidth=0.65,
                alpha=0.85,
                rstride=1,
                cstride=1,
                zorder=3,
                clip_on=False,
            )
            if rows:
                ax.scatter(
                    [r["planning"] for r in rows],
                    [r["execution"] for r in rows],
                    [r["mean_TW"]["mean"] for r in rows],
                    s=5.5 if controller != "sufficient" else 7,
                    marker=marker,
                    facecolors=color,
                    edgecolors="white",
                    linewidths=0.2,
                    depthshade=False,
                    zorder=5 if controller == "sufficient" else 4,
                    clip_on=False,
                )
        ax.set_title(
            f"({'c' if not window else 'f'}) $\\Pi=2$",
            loc="left",
            x=-0.23,
            y=0.95,
            pad=-1,
            fontsize=7.5,
        )
        ax.set_xlabel(r"$T_P$", labelpad=-6)
        ax.set_ylabel(r"$T_E$", labelpad=-8)
        ax.set_zlabel("")
        ax.text2D(
            -0.23,
            0.55,
            r"Mean $T_W$",
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="center",
        )
        ax.set_xticks([50, 250, 500])
        ax.set_yticks([50, 250, 500])
        ax.tick_params(labelsize=7, pad=-1)
        ax.view_init(elev=22, azim=-65)
        ax.set_box_aspect((2.0, 1.0, 0.8), zoom=1.55)
        ax.set_zlim(-30, 850)
        ax.set_zticks([0, 400, 800])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor("#dddddd")
            axis._axinfo["grid"].update(color="#dddddd", linewidth=0.35)
    fig.legend(
        handles=handles(FIXED),
        loc="upper center",
        bbox_to_anchor=(0.51, 1),
        ncol=3,
        columnspacing=1.5,
    )
    return finish(fig, figures / "s1_metrics.pdf")


def variable_figure(summary, theory, figures, envelope):
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.15, right=0.975, bottom=0.12, top=0.77, hspace=0.4)
    for ax, window in zip(axes, (0, 100)):
        for controller in VARIABLE:
            color, _, line, _ = STYLES[controller]
            refs = sorted(
                [
                    r
                    for r in theory["references"]
                    if r["window"] == window and r["controller"] == controller
                ],
                key=lambda r: r["delta_P"],
            )
            ax.plot(
                [r["delta_P"] + 100 for r in refs],
                [r["mean_TW"] for r in refs],
                color=color,
                ls=line,
                lw=1.05,
                zorder=4 if controller == "sufficient" else 3,
            )
            rows = sorted(
                [
                    r
                    for r in summary["summary"]
                    if r["family"] == "variable"
                    and r["window"] == window
                    and r["controller"] == controller
                ],
                key=lambda r: r["planning"],
            )
            observed(ax, rows, [r["planning"] + 100 for r in rows], controller, envelope)
        ax.set_title(
            "(a) Event-driven" if not window else r"(b) Window-based ($w=100$)", loc="left"
        )
        ax.set_ylabel(r"Mean $T_W$")
        ax.set_xlim(130, 1020)
        ax.set_ylim(-15, 480)
        ax.set_yticks([0, 200, 400])
        ax.set_xticks([150, 400, 600, 800, 1000])
    axes[1].set_xlabel(r"$\delta_G=\delta_P+\delta_E$")
    fig.legend(
        handles=handles(("sequential", "macpe2", "macpe1", "macpe3")),
        loc="upper center",
        bbox_to_anchor=(0.53, 1.005),
        ncol=2,
        columnspacing=1.3,
    )
    fig.legend(
        handles=handles(("sufficient",)), loc="upper center", bbox_to_anchor=(0.52, 0.9), ncol=1
    )
    return finish(fig, figures / "s2_conf.pdf")


def timing_figures(data, output):
    summary = json.loads((data / "summary.json").read_text())
    theory = json.loads((data / "theory.json").read_text())
    assert summary["runs"] == 9720 and len(summary["summary"]) == 1020
    assert all(
        0 <= row["request_TW_min"] <= row["request_TW_max"] <= 1000 for row in summary["summary"]
    )
    assert all(
        row["mean_TW"]["n"]
        == (
            1
            if row["family"] == "surface"
            or row["family"] == "fixed"
            and row["arrival"] == "periodic"
            else 30
        )
        for row in summary["summary"]
    )
    configure(**{"axes.titlepad": 4})
    curves = dense_fixed()
    checks = dict(
        fixed=fixed_figure(summary, theory, curves, output, "requests"),
        variable=variable_figure(summary, theory, output, "requests"),
    )
    metadata = dict(
        inputs=["summary.json", "theory.json"],
        runs=summary["runs"],
        envelope="requests",
        checks=checks,
        extrema="Individual-request minima and maxima in fixed 2D panels",
        confidence_intervals="95% Student-t intervals across run means in fixed 2D panels",
    )
    (output / "timing.json").write_text(json.dumps(rounded(metadata), indent=2) + "\n")
    return checks


# B: capacity and coordination cost.


def group_ranges(runs):
    groups = defaultdict(list)
    for run in runs:
        key = (
            run["family"],
            run["capacity_scale"],
            run["coordination"],
            run["policy"],
            run["planners"],
        )
        groups[key].append(run)
    result = []
    for (family, scale, cost, policy, planners), records in sorted(groups.items()):
        item = dict(
            family=family,
            capacity_scale=scale,
            coordination=cost,
            policy=policy,
            planners=planners,
            replications=len(records),
        )
        successful = [o for r in records for o in r["outcomes"] if o["TD"] is not None]
        for metric in ("mean_TD", "completion_percent"):
            means = [r["metrics"][metric] for r in records]
            item[metric] = dict(
                **cost_interval(means), replication_min=min(means), replication_max=max(means)
            )
        item["mean_TD"].update(
            request_min=min(o["TD"] for o in successful),
            request_max=max(o["TD"] for o in successful),
        )
        item["successful_requests"] = len(successful)
        # Successful requests have no resource or executor delay.
        extra = [o["TD"] - o["TW"] - 90 - cost for o in successful]
        assert set(extra) == {0}, (family, scale, cost, policy, planners, set(extra))
        assert all(r["metrics"]["admission_latency_mean"] == cost for r in records)
        assert all(r["metrics"]["executor_wait_max"] == 0 for r in records)
        categories = sorted({c for r in records for c in r["exit_counts"]})
        item["mean_exit_counts"] = {
            category: statistics.mean(r["exit_counts"].get(category, 0) for r in records)
            for category in categories
        }
        item["mean_delayed_candidates"] = statistics.mean(r["metrics"]["delayed"] for r in records)
        item["successful_resource_delays"] = 0
        result.append(item)
    return result


def draw(ranges, output, envelope):
    configure(**{"legend.fontsize": 8})
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 1.75))
    fig.subplots_adjust(left=0.062, right=0.99, bottom=0.255, top=0.785, wspace=0.37)
    styles = (
        ("#C62828", "o", "Sequential MAPE-K", "-"),
        ("#191919", "s", r"MACPE-K ($k_P=1$)", "--"),
        ("#DA7900", "^", r"MACPE-K ($k_P=3$)", "-"),
    )

    def panel(ax, rows, metric, xkey, color, marker, line, baseline=False):
        xx = [r[xkey] for r in rows]
        if baseline:
            xx = [0, 60]
            rows = [rows[0], rows[0]]
        prefix = "request" if metric == "mean_TD" and envelope == "requests" else "replication"
        if envelope != "none":
            ax.fill_between(
                xx,
                [r[metric][prefix + "_min"] for r in rows],
                [r[metric][prefix + "_max"] for r in rows],
                color=color,
                alpha=0.12,
                edgecolor="none",
                zorder=1,
            )
        yy = [r[metric]["mean"] for r in rows]
        ax.errorbar(
            xx,
            yy,
            yerr=[r[metric]["half_width"] for r in rows],
            color=color,
            marker=None if baseline else marker,
            markersize=3.4 if color == "#C62828" else 2.8,
            markerfacecolor="white",
            markeredgewidth=0.65,
            linestyle=line,
            linewidth=0.85,
            elinewidth=0.6,
            capsize=1.4,
            zorder=3,
        )

    for (policy, planners), (color, marker, label, line) in zip(COST_CONTROLLERS, styles):
        rows = sorted(
            [
                r
                for r in ranges
                if r["family"] == "capacity" and r["policy"] == policy and r["planners"] == planners
            ],
            key=lambda r: r["capacity_scale"],
        )
        for ax, metric in zip(axes[:2], ("mean_TD", "completion_percent")):
            panel(ax, rows, metric, "capacity_scale", color, marker, line)
        rows = sorted(
            [
                r
                for r in ranges
                if r["family"] == "coordination"
                and r["policy"] == policy
                and r["planners"] == planners
            ],
            key=lambda r: r["coordination"],
        )
        panel(
            axes[2],
            rows,
            "mean_TD",
            "coordination",
            color,
            marker,
            line,
            baseline=policy == "sequential",
        )
    scales = sorted({r["capacity_scale"] for r in ranges if r["family"] == "capacity"})
    for ax in axes[:2]:
        ax.set_xlabel("Capacity multiplier")
        ax.set_xscale("log", base=2)
        ax.set_xticks(scales, labels=[f"{v:g}" for v in scales])
        ax.set_xlim(0.45, 17.8)
    axes[0].set_title("(a) Successful deployments", loc="left")
    axes[0].set_ylabel(r"Mean deployment time $T_D$")
    axes[0].set_yticks([90, 100, 110, 120, 130])
    axes[1].set_title("(b) Completed applications", loc="left")
    axes[1].set_ylabel("Completed (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_yticks([0, 25, 50, 75, 100])
    axes[2].set_title("(c) Coordination cost", loc="left")
    axes[2].set_ylabel(r"Mean deployment time $T_D$")
    axes[2].set_xlabel(r"Additional delay $\delta_C$")
    axes[2].set_xlim(-1, 61)
    axes[2].set_xticks([0, 20, 40, 60])
    axes[2].set_yticks([100, 130, 160, 190])
    handles = [
        Line2D(
            [],
            [],
            color=color,
            marker=marker,
            markersize=3,
            markerfacecolor="white",
            markeredgewidth=0.65,
            linewidth=0.85,
            linestyle=line,
            label=label,
        )
        for color, marker, label, line in styles
    ]
    fig.legend(
        handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.53, 1.005), columnspacing=2.2
    )
    checks = verify_layout(fig)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output, metadata={"Creator": "MACPE-K simulation", "CreationDate": None, "ModDate": None}
    )
    fig.savefig(output.with_suffix(".png"), dpi=240)
    plt.close(fig)
    return dict(checks=checks, geometry_inches=[TEXT_WIDTH, 1.75], file=output.name)


def cost_figures(data, output):
    path = data / "runs.json.gz"
    records = read_json(path)
    ranges = group_ranges(records)
    figure = output / "conflicts_coordination.pdf"
    result = draw(ranges, figure, "none")
    result.update(
        input=path.name,
        runs=len(records),
        confidence_intervals="95% Student-t intervals across run means",
        data=ranges,
    )
    figure.with_suffix(".json").write_text(json.dumps(rounded(result), indent=2) + "\n")
    return result


# C: physical completion, paired outcomes, and mechanism ablations.


def placement_key(arrival_mode, planners, policy):
    return f"arrival={arrival_mode},planners={planners},{policy}"


def placement_summary(runs):
    """Compute run-level intervals and paired gains against sequential control.

    Positive savings mean MACPE-K is faster. Common-success comparisons use
    requests completed by both controllers in the same replication.
    """
    grouped = {}
    identities = set()
    for run in runs:
        label = placement_key(run["arrival_mode"], run["planners"], run["policy"])
        identity = (label, run["replication"])
        if identity in identities:
            raise ValueError(f"Duplicate run: {identity}")
        identities.add(identity)
        if run["policy"] == "sequential" and run["planners"] != 1:
            raise ValueError("This campaign has exactly one sequential controller.")
        grouped.setdefault(label, []).append(run)
    summary, rejection_summary = {}, {}
    all_exits = sorted({reason for run in runs for reason in run["exit_counts"]})
    for label, group in sorted(grouped.items()):
        names = sorted(set.intersection(*(set(run["metrics"]) for run in group)))
        summary[label] = {
            name: interval([run["metrics"][name] for run in group])
            for name in names
            if all(
                isinstance(run["metrics"][name], Real)
                and not isinstance(run["metrics"][name], bool)
                for run in group
            )
        }
        rejection_summary[label] = {
            reason: interval([run["exit_counts"].get(reason, 0) for run in group])
            for reason in all_exits
        }
    paired = {}
    for label, group in sorted(grouped.items()):
        exemplar = group[0]
        if exemplar["policy"] != "macpe":
            continue
        baseline_key = placement_key(exemplar["arrival_mode"], 1, "sequential")
        baseline = {run["replication"]: run for run in grouped[baseline_key]}
        if set(baseline) != {run["replication"] for run in group}:
            raise ValueError(f"Unpaired replication sets: {label}")
        observations = {
            name: []
            for name in (
                "completed_difference_macpe_minus_sequential",
                "conditional_response_saving_percent",
                "common_success_response_saving_percent",
                "common_success_response_difference",
                "common_success_count",
            )
        }
        for mac in sorted(group, key=lambda run: run["replication"]):
            seq = baseline[mac["replication"]]
            assert seq["arrival_mode"] == mac["arrival_mode"]
            observations["completed_difference_macpe_minus_sequential"].append(
                mac["metrics"]["completed_requests"] - seq["metrics"]["completed_requests"]
            )
            sr, mr = seq["metrics"].get("response_mean"), mac["metrics"].get("response_mean")
            if sr is not None and mr is not None and sr > 0:
                observations["conditional_response_saving_percent"].append(100 * (1 - mr / sr))
            success_seq = {row["id"]: row["response"] for row in seq["outcomes"] if not row["lost"]}
            success_mac = {row["id"]: row["response"] for row in mac["outcomes"] if not row["lost"]}
            common = sorted(set(success_seq) & set(success_mac))
            observations["common_success_count"].append(len(common))
            if common:
                a = statistics.mean(success_seq[index] for index in common)
                b = statistics.mean(success_mac[index] for index in common)
                observations["common_success_response_difference"].append(a - b)
                if a > 0:
                    observations["common_success_response_saving_percent"].append(100 * (1 - b / a))
        paired[label] = dict(
            baseline=baseline_key,
            **{name: interval(values) if values else None for name, values in observations.items()},
        )
    return dict(summary=summary, paired=paired, terminal_outcomes=rejection_summary)


def load_runs(directory):
    """Read a campaign's runs and protocol and check the recorded run count."""
    protocol = json.loads((directory / "protocol.json").read_text())
    runs = read_json(directory / "runs.json.gz")
    if len(runs) != protocol["run_count"]:
        raise ValueError(f"Incomplete campaign: {directory}")
    return runs, protocol


def pm(stats, digits=1):
    if stats is None or stats.get("mean") is None:
        return r"\textemdash"
    if stats.get("half_width") is None:
        return f"${stats['mean']:.{digits}f}$"
    return f"${stats['mean']:.{digits}f}\\,\\pm\\,{stats['half_width']:.{digits}f}$"


def defined_interval(values):
    defined = [value for value in values if value is not None]
    result = interval(defined) if defined else dict(n=0, mean=None, half_width=None)
    result["undefined_runs"] = len(values) - len(defined)
    return result


def placement_tables(full, ablations, output):
    raw_runs, _ = load_runs(full)
    runs = [physical_completion_view(run) for run in raw_runs]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "effective_runs.json.gz", runs)
    audit = dict(
        input="runs.json.gz",
        output="effective_runs.json.gz",
        runs=len(runs),
        affected_runs=sum(bool(r["observation"]["completed_after_controller_stop"]) for r in runs),
        completed_after_controller_stop=sum(
            r["metrics"]["completed_after_controller_stop"] for r in runs
        ),
        per_run=[
            dict(
                replication=r["replication"],
                arrival_mode=r["arrival_mode"],
                policy=r["policy"],
                planners=r["planners"],
                window=r["window"],
                controller_completed=r["controller_metrics"]["completed_requests"],
                physically_completed=r["metrics"]["completed_requests"],
                controller_mean_TD=r["controller_metrics"]["response_mean"],
                physically_completed_mean_TD=r["metrics"]["mean_TD"],
                affected_requests=r["observation"]["completed_after_controller_stop"],
            )
            for r in runs
        ],
        definition="Application completion includes successful component actions after the controller stops.",
    )
    (output / "observation_audit.json").write_text(json.dumps(rounded(audit), indent=2) + "\n")
    summary = {
        str(w): placement_summary([r for r in runs if r["window"] == w])
        for w in sorted({r["window"] for r in runs})
    }
    for window, values in summary.items():
        for label, stats in values["summary"].items():
            group = [
                r
                for r in runs
                if r["window"] == int(window)
                and f"arrival={r['arrival_mode']},planners={r['planners']},{r['policy']}" == label
            ]
            for metric in ("mean_TD", "response_mean", "mean_TW_success"):
                stats[metric] = defined_interval([r["metrics"][metric] for r in group])
    (output / "summary.json").write_text(json.dumps(rounded(summary), indent=2) + "\n")
    assert "0" in summary, "main placement table uses event-driven control"
    result = summary["0"]
    performance = [
        "% Generated by report.py.",
        r"{\footnotesize\setlength{\tabcolsep}{2pt}%",
        r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrr@{}}",
        r"\toprule",
        r"Controller & Mean $T_D$ & Completed (/200) \\",
        r"\midrule",
    ]
    csv_rows = []
    for arrival in ("frequent", "sparse"):
        if arrival == "sparse":
            performance.append(r"\midrule")
        performance.append(r"\multicolumn{3}{l}{\textit{" + arrival.title() + r" arrivals}} \\")
        for k, policy in ((1, "sequential"), (1, "macpe"), (4, "macpe")):
            key = f"arrival={arrival},planners={k},{policy}"
            stats = result["summary"][key]
            label = "MAPE-K (seq.)" if policy == "sequential" else f"MACPE-K ($k_P={k}$)"
            performance.append(
                f"{label} & {pm(stats['mean_TD'])} & {pm(stats['completed_requests'],2)} \\\\"
            )
            row = dict(arrivals=arrival, controller=policy, planners=k)
            for metric in ("mean_TD", "mean_TW", "mean_TW_success", "completed_requests"):
                row[metric] = stats[metric]["mean"]
                row[metric + "_ci_half"] = stats[metric]["half_width"]
            csv_rows.append(row)
    performance += [r"\bottomrule", r"\end{tabular*}}%"]
    (output / "integrated_table.tex").write_text("\n".join(performance) + "\n")
    csv_rows = rounded(csv_rows)
    with (output / "integrated_table.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    if ablations:
        raw_ablations, _ = load_runs(ablations)
        ablations = [physical_completion_view(run) for run in raw_ablations]
        full_lookup = {
            (r["replication"], r["window"]): r
            for r in runs
            if r["arrival_mode"] == "frequent" and r["planners"] == 3 and r["policy"] == "macpe"
        }
        for run in ablations:
            baseline = full_lookup[(run["replication"], run["window"])]
            assert run["arrival_mode"] == baseline["arrival_mode"]
            if run["ablation"] == "full":
                assert run == baseline
        groups = {
            variant: [r for r in ablations if r["ablation"] == variant and r["window"] == 0]
            for variant in ("full", "no_aggregate", "no_fresh", "no_compensation")
        }
        observed = {}
        for variant, group in groups.items():
            observed[variant] = {}
            for metric in (
                "capacity_violations",
                "network_capacity_violations",
                "network_reservation_violations",
                "stale_admissions",
                "leaked_reservations",
                "completed_requests",
                "mean_TD",
            ):
                observed[variant][metric] = defined_interval([r["metrics"][metric] for r in group])
        assert all(r["metrics"][m] == 0 for r in groups["full"] for m in SAFETY_METRICS)
        (output / "ablation_summary.json").write_text(
            json.dumps(rounded(observed), indent=2) + "\n"
        )
        text = [
            "% Generated by report.py.",
            r"{\footnotesize\setlength{\tabcolsep}{2pt}%",
            r"\begin{tabularx}{\columnwidth}{@{}>{\raggedright\arraybackslash}p{.29\columnwidth}>{\raggedright\arraybackslash}Xr@{}}",
            r"\toprule",
            r"Disabled mechanism & Measured outcome & Count per run \\",
            r"\midrule",
            r"None & All monitored outcomes & $0.0\,\pm\,0.0$ \\",
        ]
        for label, outcome, variant, metric in (
            ("Aggregate capacity check", "Host violations", "no_aggregate", "capacity_violations"),
            ("", "Link violations", "no_aggregate", "network_capacity_violations"),
            ("Freshness", "Stale admissions", "no_fresh", "stale_admissions"),
            ("Compensation", "Leaked host reservations", "no_compensation", "leaked_reservations"),
        ):
            text.append(f"{label} & {outcome} & {pm(observed[variant][metric])} \\\\")
        text += [r"\bottomrule", r"\end{tabularx}}%"]
        (output / "ablation_table.tex").write_text("\n".join(text) + "\n")
    print(f"Wrote placement summaries and tables to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("figures", "tables", "all"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction/outputs")
    args = parser.parse_args()
    if args.output.resolve() == (ROOT / "reference").resolve():
        parser.error("Choose an output directory other than reference/")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command in ("figures", "all"):
        timing_figures(args.data / "timing", args.output)
        cost_figures(args.data / "costs", args.output)
    if args.command in ("tables", "all"):
        placement_tables(
            args.data / "placement/full", args.data / "placement/ablations", args.output
        )
    print(f"Wrote outputs to {args.output}")


if __name__ == "__main__":
    main()
