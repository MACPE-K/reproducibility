"""Check recorded results and simulation behavior."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from dataclasses import dataclass, field
import gzip
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
import unittest

from scipy.stats import t

import experiments as exp
from experiments import physical_completion_view, timing_inputs, timing_planner_count
from simulation import (
    ADMITTED,
    CANCELLED,
    DEP,
    REJECTED_STALE,
    REJECTED_SUPERSEDED,
    SAFETY_METRICS,
    BatchedNetworkDagSimulator,
    Candidate,
    ContractSimulator,
    DagRequest,
    ExactTimingSimulator,
    GeneratedCandidate,
    PlacementUnavailable,
    Request,
    ScopedGenerationContract,
    BatchAblationSimulator,
    completed_action_observations,
    make_workload,
    make_network_dag_plan,
)
from theory import pathwise

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


ROUNDING = 0.0051  # recorded summaries are rounded to two decimals
# Paired response-time savings shown in the paper's performance table, in percent,
# by (duration profile, k_P), for frequent arrivals.
PAIRED_SAVINGS = {
    ("short_short", 1): 0.0,
    ("short_short", 4): 0.7,
    ("long_short", 1): -9.3,
    ("long_short", 4): 14.1,
    ("short_long", 1): 31.6,
    ("short_long", 4): 31.9,
    ("long_long", 1): 7.3,
    ("long_long", 4): 27.4,
}
PLACEMENT_COMPARISONS = (("sequential", 1), ("macpe", 1), ("macpe", 2), ("macpe", 4))


def close(actual, expected, label="value", tolerance=1e-10):
    if actual is None or expected is None:
        require(actual is expected, f"{label}: undefined value differs")
    else:
        require(
            abs(actual - expected) <= tolerance, f"{label}: {actual!r} differs from {expected!r}"
        )


def interval(values):
    mean = statistics.mean(values)
    half = (
        float(t.ppf(0.975, len(values) - 1)) * statistics.stdev(values) / math.sqrt(len(values))
        if len(values) > 1
        else 0.0
    )
    return mean, half


# Recorded campaigns.


def timing_records():
    with gzip.open(DATA / "timing/runs.jsonl.gz", "rb") as stream:
        for line in stream:
            yield json.loads(line)


def data_integrity():
    keys = [exp.timing_key(row["spec"]) for row in timing_records()]
    expected = {exp.timing_key(spec) for spec in exp.timing_specifications()}
    require(len(keys) == len(set(keys)), "Duplicate timing records")
    require(set(keys) == expected, "Missing or unexpected timing records")
    return dict(timing_records=len(keys), status="passed")


def campaign(relative, count):
    directory = DATA / relative
    protocol = read(directory / "protocol.json")
    rows = exp.read_json(directory / "runs.json.gz")
    require(len(rows) == count, f"Incomplete campaign: {relative}")
    if "run_count" in protocol:
        require(protocol["run_count"] == count, f"Wrong run count: {relative}")
    if "specs" in protocol:
        require(len(protocol["specs"]) == count, f"Wrong specification count: {relative}")
        for spec, row in zip(protocol["specs"], rows):
            require(all(row[k] == v for k, v in spec.items()), f"Run/spec mismatch: {relative}")
    return protocol, rows


# A: every recorded input and individual timing recurrence.


def timing():
    specs = exp.timing_specifications()
    config, summary = read(DATA / "timing/configuration.json"), read(DATA / "timing/summary.json")
    require(config["specifications"] == specs, "Timing specification drift")
    rows = {exp.timing_key(row["spec"]): row for row in timing_records()}
    require(len(rows) == len(specs) == summary["runs"] == 9720, "Wrong timing run count")
    groups = defaultdict(list)
    names = ("family", "arrival", "window", "planning", "execution", "controller")
    for spec in specs:
        row = rows[exp.timing_key(spec)]
        require(row["spec"] == spec, "Timing run/spec mismatch")
        _, arrivals, planning, execution = exp.timing_inputs(spec)
        require(
            row["input"] == dict(arrivals=arrivals, planning=planning, execution=execution),
            "Timing input mismatch",
        )
        require(row["planners"] == exp.timing_planner_count(spec), "Wrong planner count")
        expected = pathwise(
            arrivals,
            planning,
            execution,
            row["planners"],
            spec["window"],
            spec["controller"] == "sequential",
        )
        require(row["observations"]["waiting"] == expected["waits"], "Waiting recurrence mismatch")
        require(
            row["observations"]["deployment"] == expected["deployment"],
            "Deployment recurrence mismatch",
        )
        require(row["batches"] == len(expected["batches"]), "Batch count differs")
        require(
            row["metrics"]["completed_requests"]
            == row["metrics"]["final_deployed_applications"]
            == 200
            and row["metrics"]["lost_requests"] == 0
            and row["metrics"]["final_consistent"] == 1,
            "Incomplete timing run",
        )
        require(
            all(row["metrics"][metric] == 0 for metric in SAFETY_METRICS), "Timing safety violation"
        )
        for metric, observations in (("TW", "waiting"), ("TD", "deployment")):
            close(row["mean_" + metric], statistics.mean(row["observations"][observations]), metric)
        groups[tuple(spec[name] for name in names)].append(row)
    require(len(summary["summary"]) == len(groups) == 1020, "Wrong timing group count")
    indexed = {}
    for row in summary["summary"]:
        identity = tuple(row[name] for name in names)
        indexed[identity] = row
        group = groups[identity]
        for metric, observations in (("TW", "waiting"), ("TD", "deployment")):
            values = [item["mean_" + metric] for item in group]
            mean, half = interval(values)
            stats = row["mean_" + metric]
            require(stats["n"] == len(values), "Timing sample count differs")
            close(stats["mean"], mean, tolerance=ROUNDING)
            close(stats["ci_low"], mean - half if len(values) > 1 else None, tolerance=ROUNDING)
            close(stats["ci_high"], mean + half if len(values) > 1 else None, tolerance=ROUNDING)
            close(stats["min"], min(values), tolerance=ROUNDING)
            close(stats["max"], max(values), tolerance=ROUNDING)
            close(row["reference_" + metric], mean, tolerance=ROUNDING)
            require(
                row["request_" + metric + "_min"]
                == min(min(item["observations"][observations]) for item in group),
                "Request minimum differs",
            )
            require(
                row["request_" + metric + "_max"]
                == max(max(item["observations"][observations]) for item in group),
                "Request maximum differs",
            )
    theory = read(DATA / "timing/theory.json")
    for section, family in (
        ("fixed_periodic_references", "fixed"),
        ("surface_references", "surface"),
    ):
        for ref in theory[section]:
            row = indexed[
                (family, "periodic", ref["window"], ref["T_P"], ref["T_E"], ref["controller"])
            ]
            close(row["mean_TW"]["mean"], ref["mean_TW"], tolerance=2 * ROUNDING)
            close(row["mean_TD"]["mean"], ref["mean_TD"], tolerance=2 * ROUNDING)
    inside = comparisons = 0
    for ref in theory["references"]:
        row = indexed[
            ("variable", "periodic", ref["window"], ref["delta_P"], 100, ref["controller"])
        ]
        for metric in ("TW", "TD"):
            observed = row["mean_" + metric]
            inside += (
                observed["ci_low"] - ROUNDING
                <= ref["mean_" + metric]
                <= observed["ci_high"] + ROUNDING
            )
            comparisons += 1
    return dict(
        runs=len(rows),
        applications=len(rows) * 200,
        groups=len(groups),
        all_individual_recurrences_match=True,
        safety_violations=0,
        expectations_inside_pointwise_95_intervals=inside,
        comparisons=comparisons,
    )


# B and C: statistics, completion observations, and paired comparisons.


def costs():
    protocol, rows = campaign("costs", 555)
    groups = defaultdict(list)
    for row in rows:
        groups[
            row["family"],
            row["capacity_scale"],
            row["coordination"],
            row["policy"],
            row["planners"],
        ].append(row)
        successes = [r for r in row["outcomes"] if r["completed"] is not None]
        close(row["metrics"]["mean_TD"], statistics.mean(r["TD"] for r in successes), "Cost TD")
        require(
            row["metrics"]["completed_requests"] == len(successes), "Cost completion count differs"
        )
        require(row["metrics"]["executor_wait_mean"] == 0, "Unexpected executor wait")
    for summary in read(DATA / "costs/summary.json"):
        group = groups[
            summary["family"],
            summary["capacity_scale"],
            summary["coordination"],
            summary["policy"],
            summary["planners"],
        ]
        for metric in ("mean_TD", "mean_TW", "completion_percent", "completed_requests"):
            mean, half = interval([r["metrics"][metric] for r in group])
            close(summary[metric]["mean"], mean, "Cost group mean", tolerance=ROUNDING)
            close(summary[metric]["half_width"], half, "Cost group interval", tolerance=ROUNDING)
    return {"runs": 555, "capacity_runs": 540, "coordination_runs": 15, "statistics": "passed"}


def placement():
    protocol, raw = campaign("placement/full", 300)
    ablation_protocol, raw_ablations = campaign("placement/ablations", 120)
    identities = [exp.placement_identity(row) for row in raw]
    require(len(set(identities)) == 300, "Duplicate full-placement runs")
    require(
        set(identities)
        == {exp.placement_identity(spec) for spec in exp.placement_specifications()},
        "Full placement campaign does not match the complete paired design",
    )
    runs = [exp.physical_completion_view(row) for row in raw]
    ablations = [exp.physical_completion_view(row) for row in raw_ablations]
    lookup = {(r["arrival_mode"], r["policy"], r["planners"], r["replication"]): r for r in runs}
    result = []
    for arrival in ("frequent", "sparse"):
        for policy, planners in PLACEMENT_COMPARISONS:
            group = [lookup[arrival, policy, planners, rep] for rep in range(1, 31)]
            td, td_half = interval([r["metrics"]["mean_TD"] for r in group])
            completed, half = interval([r["metrics"]["completed_requests"] for r in group])
            result.append(
                dict(
                    arrivals=arrival,
                    policy=policy,
                    planners=planners,
                    mean_TD=td,
                    TD_ci_half=td_half,
                    completed=completed,
                    completed_ci_half=half,
                )
            )
    cells = [("0.0", "0.0")]
    for variant, metric in (
        ("no_aggregate", "capacity_violations"),
        ("no_aggregate", "network_capacity_violations"),
        ("no_fresh", "stale_admissions"),
        ("no_compensation", "leaked_reservations"),
    ):
        mean, half = interval([r["metrics"][metric] for r in ablations if r["ablation"] == variant])
        cells.append((f"{mean:.1f}", f"{half:.1f}"))
    published = re.findall(
        r"\$([0-9.]+)\\,\\pm\\,([0-9.]+)\$", (ROOT / "reference/ablation_table.tex").read_text()
    )
    require(published == cells, "Reference ablation table differs")
    require(
        all(r["metrics"][metric] == 0 for r in runs for metric in SAFETY_METRICS),
        "Full-mechanism safety violation",
    )
    full = {
        r["replication"]: r
        for r in runs
        if r["arrival_mode"] == "frequent" and r["policy"] == "macpe" and r["planners"] == 3
    }
    for row in ablations:
        require(
            all(row[key] == full[row["replication"]][key] for key in ("arrival_mode", "window")),
            "Ablation workload differs",
        )
        if row["ablation"] == "full":
            require(row == full[row["replication"]], "Ablation full-mechanism records differ")
    return {
        "full_runs": 300,
        "ablation_records": 120,
        "unique_ablation_only_runs": 90,
        "performance_table": result,
        "completion": "passed",
    }


def duration_observations(raw):
    """Reconstruct completion and initial waiting from the recorded actions and batches."""
    outcomes = raw["outcomes"]
    require(len(outcomes) == 200, "Duration run must contain 200 original requests")
    require({row["id"] for row in outcomes} == set(range(200)), "Invalid request identities")
    starts = defaultdict(list)
    for batch in raw["batches"]:
        for identity in batch["members"]:
            starts[identity].append(batch["start"])
    completed = defaultdict(dict)
    action_ids = set()
    for action in raw["completed_action_outcomes"]:
        identity, component = action["request"], action["component"]
        require(action["kind"] == "deploy", "Unexpected duration-campaign action")
        require(action["id"] not in action_ids, "Duplicate completed action")
        require(component not in completed[identity], "Duplicate component deployment")
        require(action["finished"] > action["started"], "Invalid action timestamps")
        completed[identity][component] = action["finished"]
        action_ids.add(action["id"])
    responses, waits, completion_times = {}, {}, {}
    for row in outcomes:
        identity = row["id"]
        require(bool(starts[identity]), "Request never reached Planning")
        first_start = min(starts[identity])
        close(row["planning_started"], first_start, "First Planning start")
        wait = first_start - row["arrival"]
        require(wait >= 0, "Planning starts before request arrival")
        close(row["first_planner_wait"], wait, "First Planning wait")
        waits[identity] = wait
        scope = {f"{row['application']}.c{index}" for index in range(4)}
        require(set(completed[identity]) <= scope, "Unexpected application component")
        finish = max(completed[identity].values()) if set(completed[identity]) == scope else None
        completion_times[identity] = finish
        if not row["lost"]:
            require(finish == row["completed"], "Controller success disagrees with actions")
        if finish is not None:
            require(finish > row["arrival"], "Completion precedes arrival")
            responses[identity] = finish - row["arrival"]
    require(set(completed) <= set(waits), "Completed action has no original request")
    metrics = dict(
        mean_TW=statistics.mean(waits.values()),
        mean_TD=statistics.mean(responses.values()) if responses else None,
        completed_requests=len(responses),
        lost_requests=200 - len(responses),
        completion_percent=100 * len(responses) / 200,
    )
    close(raw["metrics"]["mean_TW"], metrics["mean_TW"], "All-request waiting mean")
    close(raw["metrics"]["final_deployed_applications"], len(responses), "Final deployment audit")
    observed = physical_completion_view(raw)
    for name in ("mean_TW", "mean_TD", "completed_requests", "lost_requests"):
        close(observed["metrics"][name], metrics[name], f"Physical completion: {name}")
    for row in observed["outcomes"]:
        close(row["completed"], completion_times[row["id"]], "Physical completion timestamp")
        close(row["response"], responses.get(row["id"]), "Physical response time")
    return dict(metrics=metrics, responses=responses, waits=waits)


def sensitivity():
    protocol, raw = campaign("placement/sensitivity", 480)
    specs = exp.sensitivity_specifications()
    fields = tuple(specs[0])
    identities = [tuple(row[name] for name in fields) for row in raw]
    require(len(set(identities)) == 480, "Duplicate duration runs")
    require(
        set(identities) == {tuple(spec[name] for name in fields) for spec in specs},
        "Duration campaign does not match the complete paired design",
    )
    require(protocol["seeds"] == list(range(1, 31)), "Duration campaign seeds differ")
    require(
        protocol["duration_profiles"] == json.loads(json.dumps(exp.DURATION_PROFILES)),
        "Recorded duration distributions differ",
    )
    for name, expected in (
        ("failure_probability", exp.FAILURE_PROBABILITY),
        ("executors", exp.EXECUTORS),
        ("max_rechecks", exp.MAX_RECHECKS),
        ("max_recomputations", exp.MAX_RECOMPUTATIONS),
        ("window", 0),
    ):
        require(protocol[name] == expected, f"Duration protocol differs: {name}")
    require(
        all(r["metrics"][metric] == 0 for r in raw for metric in SAFETY_METRICS),
        "Duration campaign safety violation",
    )
    observed = {
        (
            r["duration_profile"],
            r["policy"],
            r["planners"],
            r["replication"],
        ): duration_observations(r)
        for r in raw
    }
    full_runs = exp.read_json(DATA / "placement/full/runs.json.gz")
    baseline = {
        (r["policy"], r["planners"], r["replication"]): r
        for r in full_runs
        if r["arrival_mode"] == "frequent"
    }
    for row in raw:
        if row["duration_profile"] == "long_long":
            copied = {name: value for name, value in row.items() if name != "duration_profile"}
            original = baseline[row["policy"], row["planners"], row["replication"]]
            require(
                normalized(copied) == normalized(original),
                "Long/long duration run differs from the frequent-arrival baseline",
            )
    sparse = {
        (r["policy"], r["planners"], r["replication"]): duration_observations(r)
        for r in full_runs
        if r["arrival_mode"] == "sparse" and r["planners"] in (1, 2, 4)
    }
    with (ROOT / "reference/placement_performance_table.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == 20, "The performance table must contain twenty rows")
    blocks = [("frequent", profile) for profile in exp.DURATION_PROFILES] + [
        ("sparse", "long_long")
    ]
    expected_cells, gains, index = [], {}, 0
    for arrival, profile in blocks:
        for policy, planners in PLACEMENT_COMPARISONS:
            if arrival == "frequent":
                group = [observed[profile, policy, planners, seed] for seed in range(1, 31)]
                reference = [observed[profile, "sequential", 1, seed] for seed in range(1, 31)]
            else:
                group = [sparse[policy, planners, seed] for seed in range(1, 31)]
                reference = [sparse["sequential", 1, seed] for seed in range(1, 31)]
            csv_row = csv_rows[index]
            index += 1
            labels = (csv_row["duration_profile"], csv_row["arrivals"], csv_row["controller"])
            require(labels == (profile, arrival, policy), "Performance table row labels differ")
            require(int(csv_row["planners"]) == planners, "Performance table planner count differs")
            for metric in ("mean_TW", "mean_TD", "completed_requests"):
                mean, half = interval([row["metrics"][metric] for row in group])
                close(float(csv_row[metric]), mean, metric, tolerance=ROUNDING)
                close(float(csv_row[metric + "_ci_half"]), half, metric, tolerance=ROUNDING)
                expected_cells.append((f"{mean:.1f}", f"{half:.1f}"))
            if policy == "sequential":
                require(
                    csv_row["paired_saving_percent"] == "", "Sequential rows have no paired saving"
                )
                continue
            paired = []
            for mac, seq in zip(group, reference):
                common = sorted(set(seq["responses"]) & set(mac["responses"]))
                require(bool(common), "Duration pair has no common completed request")
                before = statistics.mean(seq["responses"][identity] for identity in common)
                after = statistics.mean(mac["responses"][identity] for identity in common)
                paired.append(100 * (before - after) / before)
            saving = statistics.mean(paired)
            close(
                float(csv_row["paired_saving_percent"]), saving, "paired saving", tolerance=ROUNDING
            )
            expected_cells.append((f"{saving:.1f}", None))
            if arrival == "frequent" and (profile, planners) in PAIRED_SAVINGS:
                gains[profile, planners] = round(saving, 1)
    table = (ROOT / "reference/placement_performance_table.tex").read_text()
    cells = re.findall(r"\$(-?[0-9.]+)(?:\\,\\pm\\,([0-9.]+))?\$", table)
    cells = [(mean, half or None) for mean, half in cells]
    require(cells == expected_cells, "Performance table differs from the recorded runs")
    require(gains == PAIRED_SAVINGS, "Paired savings differ from the values quoted in the paper")
    return dict(
        runs=480,
        baseline_matches=120,
        table_rows=20,
        paired_savings=list(gains.values()),
        status="passed",
    )


# Replay selected runs; elapsed wall time is excluded from comparisons.


def normalized(value):
    value = json.loads(json.dumps(value))
    if isinstance(value, dict):
        return {
            k: normalized(v)
            for k, v in value.items()
            if k not in ("runtime_seconds", "elapsed_seconds")
        }
    if isinstance(value, list):
        return [normalized(v) for v in value]
    return value


def differences(a, b, path=""):
    if type(a) is not type(b):
        return [(path, type(a).__name__, type(b).__name__)]
    if isinstance(a, dict):
        result = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                result.append((path + "/" + k, "missing in one record"))
            else:
                result.extend(differences(a[k], b[k], path + "/" + k))
        return result
    if isinstance(a, list):
        if len(a) != len(b):
            return [(path, len(a), len(b))]
        return [
            d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, path + "/" + str(i))
        ]
    return [] if a == b else [(path, a, b)]


def replay(output):
    output.mkdir(parents=True, exist_ok=True)
    base = DATA
    checks = []

    def check(label, actual, expected):
        diffs = differences(normalized(actual), normalized(expected))
        row = dict(
            case=label,
            exact_numerical_match=not diffs,
            difference_count=len(diffs),
            first_differences=diffs[:10],
        )
        checks.append(row)
        (output / "replay.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        assert not diffs, label

    specs = exp.timing_specifications()
    selectors = [
        dict(
            family="fixed",
            arrival="periodic",
            window=0,
            planning=500,
            execution=500,
            controller="macpe1",
        ),
        dict(family="variable", window=0, planning=900, controller="macpe2", replication=1),
        dict(family="variable", window=100, planning=900, controller="sufficient", replication=1),
        dict(family="surface", window=100, planning=500, execution=50, controller="sequential"),
    ]
    for selector in selectors:
        spec = next(s for s in specs if all(s.get(k) == v for k, v in selector.items()))
        name = exp.timing_key(spec)
        expected = next(row for row in timing_records() if row["spec"] == spec)
        actual = exp.run_timing(spec)
        check("A:" + name, actual, expected)

    rows = exp.read_json(base / "costs/runs.json.gz")
    selectors = [
        dict(family="capacity", capacity_scale=0.5, policy="sequential", planners=1),
        dict(family="capacity", capacity_scale=0.5, policy="macpe", planners=3),
        dict(family="coordination", coordination=60, policy="macpe", planners=3),
    ]
    fields = ("family", "capacity_scale", "coordination", "replication", "policy", "planners")
    for selector in selectors:
        expected = next(
            r
            for r in rows
            if r["replication"] == 1 and all(r.get(k) == v for k, v in selector.items())
        )
        spec = {k: expected[k] for k in fields}
        check("B:" + json.dumps(spec, sort_keys=True), exp.run_cost(spec), expected)

    fields = ("replication", "window", "arrival_mode", "planners", "policy", "ablation")
    for folder, selectors in [
        (
            "full",
            [
                dict(arrival_mode="frequent", policy="sequential", planners=1),
                dict(arrival_mode="frequent", policy="macpe", planners=4),
                dict(arrival_mode="sparse", policy="macpe", planners=2),
                dict(arrival_mode="sparse", policy="macpe", planners=4),
            ],
        ),
        ("ablations", [dict(ablation=x) for x in ("no_aggregate", "no_fresh", "no_compensation")]),
    ]:
        rows = exp.read_json(base / "placement" / folder / "runs.json.gz")
        for selector in selectors:
            expected = next(
                r
                for r in rows
                if r["replication"] == 1 and all(r.get(k) == v for k, v in selector.items())
            )
            spec = {k: expected[k] for k in fields}
            actual = exp.run_placement(spec)
            label = "C:" + json.dumps(spec, sort_keys=True)
            check(label, actual, expected)
            check(
                label + ":effective",
                exp.physical_completion_view(actual),
                exp.physical_completion_view(expected),
            )
    rows = exp.read_json(base / "placement/sensitivity/runs.json.gz")
    for profile, policy, planners in (
        ("short_short", "sequential", 1),
        ("long_short", "macpe", 1),
        ("long_short", "macpe", 2),
        ("short_long", "macpe", 4),
    ):
        expected = next(
            row
            for row in rows
            if (row["duration_profile"], row["policy"], row["planners"], row["replication"])
            == (profile, policy, planners, 1)
        )
        spec = {name: expected[name] for name in fields + ("duration_profile",)}
        actual = exp.run_placement(spec)
        label = "C-duration:" + json.dumps(spec, sort_keys=True)
        check(label, actual, expected)
        check(
            label + ":effective",
            physical_completion_view(actual),
            physical_completion_view(expected),
        )
    return dict(comparisons=len(checks), checks=checks, status="passed")


# Behavioral regressions.


# Contract conformance: controlled successive migration candidates.


@dataclass
class _TestPlanningSession:
    """One controlled Planning invocation, possibly emitting several revisions."""

    id: int
    started: int
    candidate_ids: list = field(default_factory=list)
    invalidated_at: object = None
    closed_at: object = None


class _RevisionHarness(ContractSimulator):
    """Exercise successive revisions within one planning session."""

    def __init__(self, executor_opens=5, duration=4, capacity=4, fail_first=False):
        request = Request(0, 0, ("c",), 1, lambda *args: ({"c": "h0"}, set()))
        failure = (
            (lambda action, request, attempt, rng: 2 if action.candidate == 0 else None)
            if fail_first
            else None
        )
        super().__init__(
            {h: capacity for h in ("h0", "h1", "h2")},
            {"c": dict(host="h0", demand=2, duration=duration)},
            [],
            planners=1,
            policy="macpe",
            failure=failure,
            log=True,
        )
        self.shared_request = request
        self.executor_opens = executor_opens
        self.sessions = []
        self.session_for_candidate = {}
        self.revision_completions = []
        self.recovery_obligations = []
        self.checkpoints = []
        self.final_intent = "h0"
        self.begin_session(0)

    def begin_session(self, time):
        """Start a new planning session after invalidation."""
        if self.sessions:
            if self.sessions[-1].invalidated_at is None:
                raise ValueError("an active Planning session cannot be replaced")
            self.sessions[-1].closed_at = time
        session = _TestPlanningSession(len(self.sessions), time)
        self.sessions.append(session)
        self.shared_request.attempt = session.id
        return session

    def emit(self, time, destination, snapshot=None):
        """Submit a candidate using the given snapshot."""
        session = self.sessions[-1]
        if session.invalidated_at is not None:
            raise ValueError("an invalidated Planning session must be restarted")
        self.final_intent = destination
        before = self.state(time, "before_emission")
        snapshot = self.snapshot(time) if snapshot is None else snapshot
        cand = Candidate(
            len(self.candidates),
            self.shared_request,
            0,
            session.id,
            time,
            snapshot,
            {"c": destination},
            set(),
        )
        self.candidates.append(cand)
        session.candidate_ids.append(cand.id)
        self.session_for_candidate[cand.id] = session.id
        self.counts["candidates"] += 1
        self._admit(cand, time)
        self._observe(time)
        self.checkpoints.append(
            dict(
                before=before,
                after=self.state(time, "after_emission"),
                emitted_candidate=cand.id,
                snapshot_time=snapshot.time,
                snapshot_view=dict(snapshot.view),
                snapshot_component_version=snapshot.versions[("comp", "c")],
            )
        )
        return cand

    def _start_ready_waves(self, time):
        if time >= self.executor_opens:
            super()._start_ready_waves(time)

    def _finish_request(self, cand, time):
        # An intermediate revision being effective is not the final adaptation.
        self.revision_completions.append(
            dict(
                candidate=cand.id,
                time=time,
                session=self.session_for_candidate[cand.id],
                actual=dict(self.m_act),
            )
        )

    def _recompute(self, request, time, cause):
        # Preserve the core's recovery signal; the controlled driver, rather than
        # the one-emission request queue, decides the next Planning invocation.
        session = self.sessions[-1]
        if session.invalidated_at is None:
            session.invalidated_at = time
        self.recovery_obligations.append(dict(time=time, cause=cause, session=session.id))
        self.log(time, "Planning session %d requires recovery after %s" % (session.id, cause))

    def step(self, time):
        self._execution_events(time)
        self._start_ready_waves(time)
        self._observe(time)
        self.checkpoints.append(dict(after=self.state(time, "execution_step")))
        self.end_time = time

    def drain(self, time):
        """Finish execution without starting another planning session."""
        self.step(time)
        while self.active:
            times = [
                a.fail_at if a.fail_at is not None else a.finish
                for a in self.active.values()
                if a.start is not None
            ]
            if time < self.executor_opens:
                times.append(self.executor_opens)
            times = [t for t in times if t is not None and t > time]
            if not times:
                raise AssertionError("controlled scenario left a wave without a wakeup")
            time = min(times)
            self.step(time)
        if self.m_act["c"] == self.final_intent and self.final_audit()["final_consistent"]:
            self.shared_request.finished = time
            self.sessions[-1].closed_at = time

    def state(self, time, phase):
        return dict(
            time=time,
            phase=phase,
            actual=dict(self.m_act),
            target=dict(self.m_tgt),
            conservative={c: self.view(c) for c in self.dem},
            lifecycle=dict(self.lc),
            loads=self.loads(),
            component_version=self.versions[("comp", "c")],
            ledger={str(key): list(entry) for key, entry in sorted(self.ledger.items())},
            active=[
                dict(
                    id=a.id,
                    source=a.source,
                    destination=a.destination,
                    start=a.start,
                    finish=a.finish,
                    predecessors=sorted(a.predecessors),
                )
                for a in sorted(self.active.values(), key=lambda a: a.id)
            ],
        )


def _test_round_trip(**kwargs):
    """Both migrations are admitted before the first executor becomes available."""
    sim = _RevisionHarness(**kwargs)
    sim.emit(1, "h1")
    sim.emit(2, "h0")
    sim.drain(2)
    return sim


def _test_three_fresh_revisions():
    sim = _RevisionHarness()
    for instant, destination in ((1, "h1"), (2, "h2"), (3, "h0")):
        sim.emit(instant, destination)
    sim.drain(3)
    return sim


def _test_stale_revision():
    sim = _RevisionHarness()
    original = sim.snapshot(0)
    sim.emit(1, "h1", original)
    sim.emit(2, "h2", original)
    sim.begin_session(2)
    sim.emit(3, "h2")
    sim.drain(3)
    return sim


def _test_failed_predecessor():
    sim = _RevisionHarness(fail_first=True)
    sim.emit(1, "h1")
    sim.emit(2, "h2")
    sim.step(5)
    sim.step(7)  # first migration fails; the second fails its action-local guard
    sim.begin_session(7)
    sim.emit(8, "h2")  # fresh replan from actual h0, not the obsolete source h1
    sim.drain(8)
    return sim


def _test_stale_after_completion():
    sim = _RevisionHarness(executor_opens=1)
    sim.emit(1, "h1")
    retained = sim.snapshot(2)
    sim.step(5)
    sim.emit(6, "h2", retained)
    sim.begin_session(6)
    sim.emit(7, "h2")
    sim.drain(7)
    return sim


_REVISION_SCENARIOS = (
    (
        "R1_round_trip",
        _test_round_trip,
        "One Planning instance emits h0->h1 and h1->h0 before either action starts.",
    ),
    (
        "R2_three_fresh",
        _test_three_fresh_revisions,
        "Three candidates from one Planning instance, re-reading versions before each emission.",
    ),
    (
        "R3_stale_original_snapshot",
        _test_stale_revision,
        "The original read snapshot is stale after revision 1; a new Planning instance recomputes revision 2.",
    ),
    (
        "R4_failed_predecessor",
        _test_failed_predecessor,
        "A failed first migration is compensated; its successor guard fails; a new Planning instance replans.",
    ),
    (
        "R5_stale_after_completion",
        _test_stale_after_completion,
        "Completion changes the read version even when it confirms the reserved destination.",
    ),
)


# Batch feasibility


def _test_workload(count=2, host_capacity=2, bandwidth=2, arrivals=None):
    hosts = {"h0": host_capacity, "h1": host_capacity}
    distances = {left: {right: int(left != right) for right in hosts} for left in hosts}
    applications, components, demands, edges = {}, {}, {}, {}
    for i in range(count):
        app = f"app{i}"
        members = (f"{app}.a", f"{app}.b")
        applications[app] = dict(
            components=members, edges=((members[0], members[1], 1),), distinct_hosts=True
        )
        edges[(app, *members)] = 1
        for component in members:
            components[component] = dict(host=None, demand=1, duration=4)
            demands[component] = 1
    requests = []
    for i, (app, spec) in enumerate(applications.items()):
        scope = spec["components"]
        request = DagRequest(
            i,
            0 if arrivals is None else arrivals[i],
            scope,
            2,
            make_network_dag_plan(app, applications, demands, distances),
            read_scope=scope,
            application=app,
            operation="deploy",
        )
        request.durations = dict.fromkeys(scope, 4)
        requests.append(request)
    return dict(
        hosts=hosts,
        components=components,
        requests=requests,
        applications=applications,
        distances=distances,
        link_capacities={("h0", "h1"): bandwidth},
        edge_demands=edges,
    )


class BatchedPlacementSafetyTests(unittest.TestCase):
    def run_case(self, work, **options):
        sim = BatchedNetworkDagSimulator(**work, **options)
        metrics = sim.run()
        self.assertEqual(
            metrics["completed_requests"] + metrics["lost_requests"], len(work["requests"])
        )
        for name in (
            "capacity_violations",
            "network_capacity_violations",
            "network_reservation_violations",
            "distinct_host_violations",
            "executor_capacity_violations",
            "stale_admissions",
            "leaked_reservations",
            "duplicate_placements",
            "invalid_transitions",
        ):
            self.assertEqual(metrics[name], 0, name)
        self.assertEqual(metrics["final_consistent"], 1, sim.final_audit())
        self.assertFalse(sim.active)
        self.assertFalse(sim.ledger)
        self.assertFalse(sim.queue)
        self.assertFalse(sim.waiting_waves)
        for request in work["requests"]:
            self.assertEqual(request.finished is None, request.lost)
        return sim, metrics

    def test_anti_affinity_is_per_application_and_links_count_once(self):
        work = _test_workload()
        sim, result = self.run_case(work, planners=3)
        self.assertEqual(result["completed_requests"], 2)
        self.assertEqual(sim.batches[0]["members"], [0, 1])
        self.assertEqual(sim.network_loads(True), {("h0", "h1"): 2})
        self.assertEqual(sim.loads(), {"h0": 2, "h1": 2})
        self.assertEqual(result["executor_peak_running_waves"], 1)

    def test_intra_batch_host_budget_does_not_discard_feasible_member(self):
        work = _test_workload(host_capacity=1)
        _, result = self.run_case(work, planners=3)
        self.assertEqual(result["completed_requests"], 1)
        self.assertIsNotNone(work["requests"][0].finished)
        self.assertTrue(work["requests"][1].lost)

    def test_intra_batch_link_budget_is_planned_cumulatively(self):
        work = _test_workload(bandwidth=1)
        _, result = self.run_case(work, planners=3)
        self.assertEqual(result["completed_requests"], 1)
        self.assertEqual(result["lost_requests"], 1)

    def test_failed_member_recovers_without_redeploying_completed_peer(self):
        work = _test_workload()
        work["requests"][1].durations = dict.fromkeys(work["requests"][1].scope, 1)

        def failure(action, request, attempt, rng):
            return (
                2 if request.id == 0 and attempt == 0 and action.component.endswith(".a") else None
            )

        sim, result = self.run_case(
            work, planners=3, failure=failure, max_recomputations=12, max_retries=12
        )
        self.assertEqual(result["completed_requests"], 2)
        self.assertEqual(work["requests"][1].finished, 3)
        done = [a.component for c in sim.candidates for a in c.actions if a.done]
        self.assertEqual(len(done), len(set(done)))
        self.assertEqual(result["failures"], 1)
        self.assertEqual(result["compensations"], 1)

    def test_exhausted_retries_do_not_erase_successful_application(self):
        work = _test_workload()
        work["requests"][1].durations = dict.fromkeys(work["requests"][1].scope, 1)

        def failure(action, request, attempt, rng):
            return 2 if request.id == 0 and action.component.endswith(".a") else None

        _, result = self.run_case(
            work, planners=1, failure=failure, max_recomputations=2, max_retries=10
        )
        self.assertEqual(result["completed_requests"], 1)
        self.assertTrue(work["requests"][0].lost)
        self.assertEqual(work["requests"][1].finished, 3)

    def test_sequential_waits_for_siblings_after_partial_failure(self):
        work = _test_workload(count=3, host_capacity=3, bandwidth=3, arrivals=[0, 0, 1])
        work["requests"][1].durations = dict.fromkeys(work["requests"][1].scope, 12)

        def failure(action, request, attempt, rng):
            return (
                1 if request.id == 0 and attempt == 0 and action.component.endswith(".a") else None
            )

        sim, _ = self.run_case(
            work, policy="sequential", planners=1, failure=failure, max_recomputations=10
        )
        self.assertGreaterEqual(sim.batches[1]["start"], 14)
        self.assertGreaterEqual(work["requests"][2].started, 14)

    def test_actual_admission_rechecks_capacity_after_concurrent_planning(self):
        work = _test_workload(host_capacity=1, arrivals=[0, 1])
        _, result = self.run_case(work, planners=3)
        self.assertEqual(result["completed_requests"], 1)
        self.assertEqual(result["lost_requests"], 1)

    def test_batch_duration_can_depend_on_size_without_dividing_by_planners(self):
        work = _test_workload()
        sim, _ = self.run_case(
            work, planners=3, batch_duration=lambda members, snapshot: 3 * len(members)
        )
        self.assertEqual(sim.batches[0]["planning_duration"], 6)
        self.assertEqual([r.finished for r in work["requests"]], [10, 10])


# Recovery


class BatchRecoveryTests(unittest.TestCase):
    @staticmethod
    def simulator(work, **kwargs):
        return BatchedNetworkDagSimulator(
            work["hosts"],
            work["components"],
            work["requests"],
            applications=work["applications"],
            distances=work["distances"],
            link_capacities=work["link_capacities"],
            edge_demands=work["edge_demands"],
            planners=1,
            policy="macpe",
            **kwargs,
        )

    def test_retry_wave_waits_for_other_applications_retained_placements(self):
        work = _test_workload(count=2, host_capacity=3, bandwidth=3, arrivals=[0, 0])
        work["requests"][1].durations = dict.fromkeys(work["requests"][1].scope, 12)

        def failure(action, request, attempt, rng):
            return (
                1
                if (request.id == 0 and attempt == 0 and action.component.endswith(".a"))
                else None
            )

        sim = self.simulator(work, failure=failure, max_recomputations=1)
        result = sim.run()
        self.assertEqual(result["completed_requests"], 2)
        self.assertEqual(result["lost_requests"], 0)
        self.assertEqual(result["precondition_failures"], 0)
        self.assertEqual(result["recomputed_after_precondition"], 0)
        self.assertEqual([b["start"] for b in sim.batches], [0, 3])
        self.assertEqual([r.finished for r in sim.requests], [18, 14])
        self.assertEqual(result["final_consistent"], 1)
        self.assertFalse(sim.active or sim.ledger or sim.waiting_waves)

    def test_all_infeasible_batch_has_no_admission_check(self):
        work = _test_workload(count=2, arrivals=[0, 0])

        def unavailable(snapshot, scope, attempt, rng):
            raise PlacementUnavailable("search_exhausted")

        for request in work["requests"]:
            request.plan = unavailable
        sim = self.simulator(work, coordination=5)
        result = sim.run()
        self.assertEqual(result["lost_requests"], 2)
        self.assertEqual(result["predicate_evaluations"], 0)
        self.assertEqual(result["end_time"], 2)
        self.assertIsNone(sim.batches[0]["check_time"])
        self.assertFalse(sim.batches[0]["candidate_created"])
        self.assertFalse(sim.candidates)


# Mechanism ablations


class BatchAblationTests(unittest.TestCase):
    def test_full_is_exact_common_simulator(self):
        first = BatchedNetworkDagSimulator(
            **_test_workload(count=3, arrivals=[0, 1, 1]), planners=3
        )
        second = BatchAblationSimulator(**_test_workload(count=3, arrivals=[0, 1, 1]), planners=3)
        self.assertEqual(first.run(), second.run())
        self.assertEqual(first.batches, second.batches)
        self.assertEqual(first.m_act, second.m_act)
        self.assertEqual(
            [(r.started, r.finished, r.lost) for r in first.requests],
            [(r.started, r.finished, r.lost) for r in second.requests],
        )

    def test_no_aggregate_omits_host_and_link_budgets_only(self):
        for host_capacity, bandwidth in ((1, 2), (2, 1)):
            with self.subTest(host_capacity=host_capacity, bandwidth=bandwidth):
                sim = BatchAblationSimulator(
                    **_test_workload(
                        host_capacity=host_capacity, bandwidth=bandwidth, arrivals=[0, 1]
                    ),
                    planners=3,
                    ablation="no_aggregate",
                )
                result = sim.run()
                metric = (
                    "capacity_violations" if host_capacity == 1 else "network_capacity_violations"
                )
                self.assertGreater(result[metric], 0)
                for preserved in (
                    "invalid_transitions",
                    "duplicate_placements",
                    "dag_latency_violations",
                    "distinct_host_violations",
                    "leaked_reservations",
                ):
                    self.assertEqual(result[preserved], 0)
                self.assertEqual(result["completed_requests"], 2)

    def test_no_fresh_accepts_changed_read_but_keeps_execution_guards(self):
        def failure(action, request, attempt, rng):
            return 1 if attempt == 0 and action.component.endswith(".a") else None

        outcomes = {}
        for ablation in ("full", "no_fresh"):
            work = _test_workload(count=1)
            work["requests"][0].planning = 5
            sim = BatchAblationSimulator(
                **work, planners=1, ablation=ablation, failure=failure, max_recomputations=8
            )
            outcomes[ablation] = sim.run()
        self.assertGreater(outcomes["full"]["rejected_stale"], 0)
        self.assertGreater(outcomes["no_fresh"]["stale_admissions"], 0)
        for result in outcomes.values():
            self.assertEqual(result["invalid_transitions"], 0)
            self.assertEqual(result["completed_requests"], 1)
            self.assertEqual(result["final_consistent"], 1)

    def test_no_compensation_keeps_failed_commitment(self):
        def failure(action, request, attempt, rng):
            return 1 if action.component.endswith(".a") else None

        sim = BatchAblationSimulator(
            **_test_workload(count=1), ablation="no_compensation", failure=failure, planners=1
        )
        result = sim.run()
        self.assertEqual(result["compensations"], 0)
        self.assertGreater(result["leaked_reservations"], 0)
        self.assertEqual(result["completed_requests"], 0)
        self.assertEqual(result["lost_requests"], 1)
        self.assertEqual(result["final_consistent"], 0)
        self.assertEqual(result["invalid_transitions"], 0)

    def test_no_aggregate_still_rejects_anti_affinity_violation(self):
        work = _test_workload(count=1)

        def invalid(snapshot, scope, attempt, rng):
            return dict.fromkeys(scope, "h0"), set()

        work["requests"][0].plan = invalid
        sim = BatchAblationSimulator(**work, ablation="no_aggregate", planners=1)
        result = sim.run()
        self.assertEqual(result["completed_requests"], 0)
        self.assertGreater(result["dag_admission_rejections"], 0)
        self.assertEqual(result["distinct_host_violations"], 0)

    def test_no_aggregate_still_rejects_latency_violation(self):
        work = _test_workload(count=1)
        app = work["applications"]["app0"]
        source, target = app["components"]
        app["edges"] = ((source, target, 0),)

        def invalid(snapshot, scope, attempt, rng):
            return {source: "h0", target: "h1"}, set()

        work["requests"][0].plan = invalid
        sim = BatchAblationSimulator(**work, ablation="no_aggregate", planners=1)
        result = sim.run()
        self.assertEqual(result["completed_requests"], 0)
        self.assertGreater(result["dag_admission_rejections"], 0)
        self.assertEqual(result["dag_latency_violations"], 0)


# Exact timing optimizations


def _test_simulate(cls, spec, capacity=1000):
    seed, arrivals, planning, execution = timing_inputs(spec)
    work = make_workload(seed)
    for key in ("hosts", "link_capacities"):
        work[key] = {k: capacity * v for k, v in work[key].items()}
    for i, request in enumerate(work["requests"]):
        request.arrival, request.planning = arrivals[i], planning[i]
        request.durations = dict(zip(request.scope, execution[i]))
    simulator = cls(
        work["hosts"],
        work["components"],
        work["requests"],
        applications=work["applications"],
        distances=work["distances"],
        link_capacities=work["link_capacities"],
        edge_demands=work["edge_demands"],
        planners=timing_planner_count(spec),
        policy="sequential" if spec["controller"] == "sequential" else "macpe",
        executors=200,
        window=spec["window"],
        seed=seed,
        log=True,
    )
    metrics = simulator.run()
    return simulator, metrics


class ExactOptimizationTests(unittest.TestCase):
    def test_entire_execution_and_metrics_match(self):
        cases = [
            ("fixed", "poisson", 0, 500, 500, "sufficient", 1000),
            ("fixed", "periodic", 100, 350, 350, "sequential", 1000),
            ("variable", "periodic", 0, 900, 100, "macpe3", 1000),
            ("variable", "periodic", 0, 900, 100, "sufficient", 1000),
            ("variable", "periodic", 100, 350, 100, "sufficient", 1000),
            ("fixed", "periodic", 0, 50, 50, "sufficient", 0.5),
        ]
        for family, arrival, window, planning, execution, controller, capacity in cases:
            spec = dict(
                family=family,
                arrival=arrival,
                window=window,
                planning=planning,
                execution=execution,
                controller=controller,
                replication=2,
            )
            with self.subTest(spec=spec, capacity=capacity):
                original, a = _test_simulate(BatchedNetworkDagSimulator, spec, capacity)
                optimized, b = _test_simulate(ExactTimingSimulator, spec, capacity)
                self.assertEqual(a, b)
                self.assertEqual(original.events, optimized.events)
                self.assertEqual(original.batches, optimized.batches)
                self.assertEqual(original.m_act, optimized.m_act)
                self.assertEqual(original.ledger, optimized.ledger)
                self.assertEqual(original.versions, optimized.versions)
                self.assertEqual(
                    [vars(c.snapshot) for c in original.candidates],
                    [vars(c.snapshot) for c in optimized.candidates],
                )
                self.assertEqual(
                    [(r.started, r.finished, r.lost) for r in original.requests],
                    [(r.started, r.finished, r.lost) for r in optimized.requests],
                )

    def test_snapshot_clone_has_independent_mutable_dictionaries(self):
        spec = dict(
            family="fixed",
            arrival="periodic",
            window=100,
            planning=500,
            execution=500,
            controller="sequential",
            replication=0,
        )
        simulator, _ = _test_simulate(ExactTimingSimulator, spec)
        source = simulator.snapshot(simulator.end_time)
        cloned = deepcopy(source)
        self.assertEqual(vars(source), vars(cloned))
        for name, value in vars(source).items():
            if isinstance(value, dict):
                self.assertIsNot(value, getattr(cloned, name))
        for app, values in source.network_application_loads.items():
            self.assertIsNot(values, cloned.network_application_loads[app])


# Physical completion


def _test_example(ablation="full", count=2, fail_all=False):
    work = _test_workload(count=count, host_capacity=3, bandwidth=3, arrivals=[0] * count)
    if count > 1:
        work["requests"][1].durations = dict.fromkeys(work["requests"][1].scope, 12)

    def failure(action, request, attempt, rng):
        return 1 if fail_all or (request.id == 0 and action.component.endswith(".a")) else None

    sim = BatchAblationSimulator(
        **work,
        planners=1,
        policy="macpe",
        failure=failure,
        max_recomputations=0,
        ablation=ablation,
        log=True,
    )
    metrics = sim.run()
    _, actions = completed_action_observations(sim)
    owners = {c: r.id for r in sim.requests for c in r.scope}
    for action in actions:
        action["batch"] = action.pop("request")
        action["request"] = owners[action["component"]]
    outcomes = [
        dict(
            id=r.id,
            application=r.application,
            components=list(r.scope),
            arrival=r.arrival,
            planning_started=r.started,
            completed=r.finished,
            lost=r.lost,
            response=None if r.finished is None else r.finished - r.arrival,
            first_planner_wait=r.started - r.arrival,
            exit=getattr(r, "outcome_reason", None) or "completed",
        )
        for r in sim.requests
    ]
    raw = dict(
        ablation=ablation,
        outcomes=outcomes,
        metrics=metrics,
        completed_action_outcomes=actions,
        batches=sim.batches,
        exit_counts=dict(Counter(r["exit"] for r in outcomes)),
    )
    return sim, raw


class PhysicalObservationTests(unittest.TestCase):
    def test_late_deployment_counted_but_partial_failed_application_is_not(self):
        sim, raw = _test_example()
        self.assertEqual(raw["metrics"]["completed_requests"], 0)
        view = physical_completion_view(raw)
        self.assertEqual(view["metrics"]["completed_requests"], 1)
        self.assertEqual(view["metrics"]["lost_requests"], 1)
        self.assertEqual(view["outcomes"][1]["completed"], 14)
        self.assertEqual(view["metrics"]["mean_TD"], 14)
        self.assertIsNone(view["outcomes"][0]["completed"])
        self.assertEqual(view["observation"]["completed_after_controller_stop"], [1])
        self.assertTrue(view["outcomes"][1]["controller_lost"])
        self.assertTrue(sim._member_complete(sim.requests[1]))
        self.assertFalse(sim._member_complete(sim.requests[0]))
        self.assertEqual(sim.final_audit()["final_consistent"], 1)

    def test_observer_cannot_change_actions_or_scheduling_and_preserves_raw_record(self):
        first, raw = _test_example()
        before = deepcopy(raw)
        trace = deepcopy(first.events)
        batches = deepcopy(first.batches)
        placements = dict(first.m_act)
        statuses = [(r.started, r.finished, r.lost) for r in first.requests]
        physical_completion_view(raw)
        second, equivalent = _test_example()
        self.assertEqual(raw, before)
        self.assertEqual(first.events, trace)
        self.assertEqual(first.events, second.events)
        self.assertEqual(first.batches, batches)
        self.assertEqual(first.batches, second.batches)
        self.assertEqual(first.m_act, placements)
        self.assertEqual(first.m_act, second.m_act)
        self.assertEqual(statuses, [(r.started, r.finished, r.lost) for r in first.requests])
        self.assertEqual(raw["completed_action_outcomes"], equivalent["completed_action_outcomes"])

    def test_same_completion_rule_applies_to_ablation_and_preserves_violation_counts(self):
        _, raw = _test_example("no_compensation")
        view = physical_completion_view(raw)
        self.assertEqual(view["metrics"]["completed_requests"], 1)
        self.assertEqual(view["outcomes"][1]["completed"], 14)
        self.assertEqual(
            view["metrics"]["leaked_reservations"], raw["metrics"]["leaked_reservations"]
        )
        self.assertGreater(view["metrics"]["leaked_reservations"], 0)

    def test_missing_or_duplicate_component_completion_cannot_create_success(self):
        _, raw = _test_example()
        raw["completed_action_outcomes"] = [
            a for a in raw["completed_action_outcomes"] if a["component"] != "app1.a"
        ]
        with self.assertRaises(AssertionError):
            physical_completion_view(raw)
        raw["metrics"]["final_deployed_applications"] = 0
        self.assertEqual(physical_completion_view(raw)["metrics"]["completed_requests"], 0)
        raw["completed_action_outcomes"].append(deepcopy(raw["completed_action_outcomes"][0]))
        with self.assertRaises(ValueError):
            physical_completion_view(raw)

    def test_no_success_has_undefined_deployment_mean_and_percentile(self):
        _, raw = _test_example(count=1, fail_all=True)
        observed = physical_completion_view(raw)
        self.assertEqual(observed["metrics"]["completed_requests"], 0)
        for metric in (
            "mean_TD",
            "response_mean",
            "deployment_mean",
            "response_p95",
            "mean_TW_success",
        ):
            self.assertIsNone(observed["metrics"][metric], metric)


# Paired workload


class WorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frequent = make_workload(7, "frequent")
        cls.sparse = make_workload(7, "sparse")

    def test_every_host_and_link_starts_unused(self):
        work = self.frequent
        self.assertEqual((len(work["hosts"]), len(work["link_capacities"])), (100, 4950))
        self.assertTrue(all(spec["host"] is None for spec in work["components"].values()))
        self.assertTrue(all(capacity > 0 for capacity in work["hosts"].values()))
        self.assertTrue(all(capacity > 0 for capacity in work["link_capacities"].values()))

    def test_all_requests_deploy_distinct_four_component_dags(self):
        work = self.frequent
        self.assertEqual(
            (len(work["requests"]), len(work["applications"]), len(work["components"])),
            (200, 200, 800),
        )
        self.assertEqual(len({r.application for r in work["requests"]}), 200)
        all_scopes = []
        edge_counts = set()
        for request in work["requests"]:
            self.assertEqual(request.operation, "deploy")
            self.assertEqual(len(request.scope), 4)
            self.assertEqual(request.scope, request.read_scope)
            all_scopes.extend(request.scope)
            app = work["applications"][request.application]
            edge_counts.add(len(app["edges"]))
            self.assertTrue(app["distinct_hosts"])
            outgoing = {component: [] for component in app["components"]}
            indegree = dict.fromkeys(outgoing, 0)
            for left, right, limit in app["edges"]:
                outgoing[left].append(right)
                indegree[right] += 1
                self.assertTrue(2 <= limit <= 6)
            ready = [component for component, count in indegree.items() if count == 0]
            visited = []
            while ready:
                component = ready.pop()
                visited.append(component)
                for target in outgoing[component]:
                    indegree[target] -= 1
                    if indegree[target] == 0:
                        ready.append(target)
            self.assertEqual(len(visited), 4)
        self.assertEqual(len(set(all_scopes)), 800)
        self.assertGreater(len(edge_counts), 1)

    def test_arrival_regimes_change_only_arrivals(self):
        a, b = self.frequent, self.sparse
        for key in ("hosts", "distances", "link_capacities", "applications", "components"):
            self.assertEqual(a[key], b[key])
        self.assertEqual(a["edge_demands"], b["edge_demands"])
        for x, y in zip(a["requests"], b["requests"]):
            self.assertEqual(
                (x.id, x.scope, x.planning, x.durations), (y.id, y.scope, y.planning, y.durations)
            )
        self.assertNotEqual([r.arrival for r in a["requests"]], [r.arrival for r in b["requests"]])
        for work, bounds in ((a, (10, 30)), (b, (100, 300))):
            previous = 0
            for request in work["requests"]:
                self.assertTrue(bounds[0] <= request.arrival - previous <= bounds[1])
                self.assertTrue(20 <= request.planning <= 40)
                self.assertTrue(
                    all(20 <= duration <= 60 for duration in request.durations.values())
                )
                previous = request.arrival

    def test_workloads_are_reproducible_and_seed_specific(self):
        regenerated = make_workload(7, "frequent")
        self.assertEqual(self.frequent["applications"], regenerated["applications"])
        self.assertEqual(self.frequent["components"], regenerated["components"])
        self.assertEqual(
            [(r.arrival, r.planning, r.durations) for r in self.frequent["requests"]],
            [(r.arrival, r.planning, r.durations) for r in regenerated["requests"]],
        )
        self.assertNotEqual(
            self.frequent["applications"], make_workload(8, "frequent")["applications"]
        )

    def test_invalid_arrival_mode_rejected(self):
        with self.assertRaises(ValueError):
            make_workload(1, "unknown")


class DurationSensitivityTests(unittest.TestCase):
    @staticmethod
    def spec(profile="long_long", seed=7, policy="macpe", planners=1):
        return dict(
            replication=seed,
            arrival_mode="frequent",
            window=0,
            policy=policy,
            planners=planners,
            ablation="full",
            duration_profile=profile,
        )

    @staticmethod
    def inputs(work, durations=True):
        result = {name: value for name, value in work.items() if name != "requests"}
        result = deepcopy(result)
        result["requests"] = [
            {name: value for name, value in vars(request).items() if name != "plan"}
            for request in work["requests"]
        ]
        if not durations:
            for component in result["components"].values():
                component.pop("duration")
            result["requests"] = [
                {
                    name: value
                    for name, value in request.items()
                    if name not in ("planning", "durations")
                }
                for request in result["requests"]
            ]
        return result

    def test_profiles_preserve_arrivals_dags_demands_and_resource_capacities(self):
        for seed in (1, 7, 30):
            baseline = self.inputs(make_workload(seed, "frequent"), durations=False)
            for profile in exp.DURATION_PROFILES:
                with self.subTest(seed=seed, profile=profile):
                    work = exp.placement_workload(self.spec(profile, seed))
                    self.assertEqual(self.inputs(work, durations=False), baseline)

    def test_long_long_and_absent_profile_preserve_original_workload_exactly(self):
        for seed in (1, 7, 30):
            original = self.inputs(make_workload(seed, "frequent"))
            spec = self.spec(seed=seed)
            self.assertEqual(self.inputs(exp.placement_workload(spec)), original)
            spec.pop("duration_profile")
            self.assertEqual(self.inputs(exp.placement_workload(spec)), original)

    def test_duration_ranges_and_component_request_consistency(self):
        for profile, bounds in exp.DURATION_PROFILES.items():
            work = exp.placement_workload(self.spec(profile))
            planning = [request.planning for request in work["requests"]]
            actions = [component["duration"] for component in work["components"].values()]
            for values, limits in (
                (planning, bounds["planning"]),
                (actions, bounds["action_duration"]),
            ):
                self.assertTrue(all(limits[0] <= value <= limits[1] for value in values))
                self.assertGreater(len(set(values)), 1)
            for request in work["requests"]:
                self.assertEqual(
                    request.durations,
                    {name: work["components"][name]["duration"] for name in request.scope},
                )

    def test_planning_and_action_duration_factors_are_independently_paired(self):
        work = {
            profile: exp.placement_workload(self.spec(profile)) for profile in exp.DURATION_PROFILES
        }
        for a, b in (("short_short", "short_long"), ("long_short", "long_long")):
            self.assertEqual(
                [r.planning for r in work[a]["requests"]], [r.planning for r in work[b]["requests"]]
            )
        for a, b in (("short_short", "long_short"), ("short_long", "long_long")):
            self.assertEqual(
                [r.durations for r in work[a]["requests"]],
                [r.durations for r in work[b]["requests"]],
            )

    def test_controllers_share_inputs_and_replications_are_independent(self):
        for profile in exp.DURATION_PROFILES:
            baseline = self.inputs(exp.placement_workload(self.spec(profile)))
            for policy, planners in (("sequential", 1), ("macpe", 2), ("macpe", 4)):
                self.assertEqual(
                    self.inputs(
                        exp.placement_workload(self.spec(profile, policy=policy, planners=planners))
                    ),
                    baseline,
                )
            other = self.inputs(exp.placement_workload(self.spec(profile, seed=8)))
            self.assertNotEqual(baseline["applications"], other["applications"])
            self.assertNotEqual(baseline["requests"], other["requests"])

    def test_complete_factorial_design_has_unique_checkpoints(self):
        specs = exp.sensitivity_specifications()
        expected = {
            (profile, policy, k, seed)
            for profile in exp.DURATION_PROFILES
            for policy, k in PLACEMENT_COMPARISONS
            for seed in range(1, 31)
        }
        self.assertEqual(len(specs), 480)
        self.assertEqual(
            {(s["duration_profile"], s["policy"], s["planners"], s["replication"]) for s in specs},
            expected,
        )
        self.assertEqual(len({exp.placement_identity(spec) for spec in specs}), 480)
        self.assertTrue(
            all(
                s["window"] == 0 and s["arrival_mode"] == "frequent" and s["ablation"] == "full"
                for s in specs
            )
        )

    def test_full_design_includes_two_planners_and_preserves_ablation_controller(self):
        specs = exp.placement_specifications()
        expected = {
            (arrival, policy, k, seed)
            for arrival in ("frequent", "sparse")
            for policy, k in PLACEMENT_COMPARISONS + (("macpe", 3),)
            for seed in range(1, 31)
        }
        self.assertEqual(len(specs), 300)
        self.assertEqual(
            {(s["arrival_mode"], s["policy"], s["planners"], s["replication"]) for s in specs},
            expected,
        )
        self.assertEqual(len({exp.placement_identity(spec) for spec in specs}), 300)
        ablations = exp.placement_specifications(ablations=True)
        self.assertEqual(len(ablations), 120)
        self.assertTrue(all(s["planners"] == 3 and s["policy"] == "macpe" for s in ablations))

    def test_unknown_duration_profile_is_rejected(self):
        for profile in ("unknown", None):
            with self.assertRaises(ValueError):
                exp.placement_workload(self.spec(profile))

    def test_waiting_counts_first_planning_start_and_excludes_retries(self):
        work = _test_workload(count=3, host_capacity=3, bandwidth=3, arrivals=[0, 1, 2])

        def failure(action, request, attempt, rng):
            return (
                1 if request.id == 0 and attempt == 0 and action.component.endswith(".a") else None
            )

        sim = BatchedNetworkDagSimulator(
            **work, planners=1, policy="sequential", failure=failure, max_recomputations=10
        )
        metrics = sim.run()
        starts = [batch["start"] for batch in sim.batches if 0 in batch["members"]]
        self.assertGreater(len(starts), 1)
        self.assertGreater(max(starts), min(starts))
        self.assertEqual(work["requests"][0].started, min(starts))
        self.assertGreater(work["requests"][0].recomputations, 0)
        self.assertEqual([r.started - r.arrival for r in work["requests"]], [0, 5, 4])
        self.assertEqual(metrics["first_planner_wait_mean"], 3)


# Successive migration candidates


class SuccessiveRevisionTests(unittest.TestCase):
    def test_round_trip_is_admitted_before_first_start(self):
        sim = _test_round_trip()
        first, second = sim.candidates
        self.assertEqual((first.admitted_at, second.admitted_at), (1, 2))
        self.assertEqual((first.actions[0].start, second.actions[0].start), (5, 9))
        self.assertEqual((first.actions[0].finish, second.actions[0].finish), (9, 13))
        self.assertEqual((first.status, second.status), (ADMITTED, ADMITTED))
        self.assertEqual((sim.m_act["c"], sim.shared_request.finished), ("h0", 13))

    def test_action_endpoints_survive_the_final_projection_returning_to_source(self):
        sim = _test_round_trip()
        after_both = sim.checkpoints[1]["after"]
        self.assertEqual((after_both["actual"]["c"], after_both["conservative"]["c"]), ("h0", "h0"))
        # A guard requiring M_res != M_act would incorrectly stop
        # the first migration here. Its own destination is h1, so it must run.
        self.assertEqual(
            [(a["source"], a["destination"], a["start"]) for a in after_both["active"]],
            [("h0", "h1", None), ("h1", "h0", None)],
        )
        self.assertEqual([row["actual"]["c"] for row in sim.revision_completions], ["h1", "h0"])

    def test_serialized_admission_uses_projected_not_current_precondition(self):
        sim = _test_round_trip()
        second = sim.candidates[1]
        before = sim.checkpoints[1]["before"]
        self.assertNotEqual(
            (before["actual"]["c"], before["lifecycle"]["c"]), second.actions[0].pre_state
        )
        self.assertEqual(sim.candidates[0].actions[0].post_state, second.actions[0].pre_state)
        self.assertEqual(second.actions[0].predecessors, {sim.candidates[0].actions[0].id})
        self.assertTrue(second.serialized)
        self.assertEqual(second.status, ADMITTED)

    def test_ledger_keeps_both_action_destinations_and_current_usage(self):
        sim = _test_round_trip()
        after_both = sim.checkpoints[1]["after"]
        self.assertEqual(
            after_both["ledger"], {"0": ["c", "h1", 2, False], "1": ["c", "h0", 2, False]}
        )
        self.assertEqual(after_both["loads"], {"h0": 4, "h1": 2, "h2": 0})
        self.assertEqual(sim.counts["capacity_violations"], 0)
        self.assertEqual(sim.ledger, {})

    def test_three_candidates_are_one_instance_not_three_requests_or_split_waves(self):
        sim = _test_three_fresh_revisions()
        self.assertEqual(len(sim.sessions), 1)
        self.assertEqual(sim.sessions[0].candidate_ids, [0, 1, 2])
        self.assertTrue(all(c.request is sim.shared_request for c in sim.candidates))
        self.assertEqual({c.planner for c in sim.candidates}, {0})
        self.assertEqual({c.attempt for c in sim.candidates}, {0})
        self.assertEqual([len(c.waves) for c in sim.candidates], [1, 1, 1])
        self.assertEqual(sim.counts["split_waves"], 0)
        self.assertEqual(
            [row["actual"]["c"] for row in sim.revision_completions], ["h1", "h2", "h0"]
        )
        self.assertEqual(sim.shared_request.finished, 17)

    def test_fresh_revisions_reread_versions_within_the_active_session(self):
        sim = _test_three_fresh_revisions()
        self.assertEqual([c.snapshot.versions[("comp", "c")] for c in sim.candidates], [0, 1, 2])
        self.assertEqual([c.snapshot.view["c"] for c in sim.candidates], ["h0", "h1", "h2"])
        self.assertEqual(sim.counts["rejected_stale"], 0)
        self.assertEqual(sim.recovery_obligations, [])

    def test_retained_original_snapshot_is_rejected_then_recomputed_in_new_instance(self):
        sim = _test_stale_revision()
        self.assertEqual([c.status for c in sim.candidates], [ADMITTED, REJECTED_STALE, ADMITTED])
        self.assertIs(sim.candidates[0].snapshot, sim.candidates[1].snapshot)
        self.assertEqual(sim.candidates[1].actions, [])
        self.assertEqual([s.candidate_ids for s in sim.sessions], [[0, 1], [2]])
        self.assertEqual([row["cause"] for row in sim.recovery_obligations], ["stale"])
        self.assertEqual(
            (sim.candidates[2].attempt, sim.candidates[2].snapshot.view["c"]), (1, "h1")
        )
        self.assertEqual((sim.m_act["c"], sim.final_audit()["final_consistent"]), ("h2", 1))

    def test_completion_invalidates_retained_snapshot_even_if_placement_matches(self):
        sim = _test_stale_after_completion()
        old = sim.candidates[1]
        self.assertEqual(old.snapshot.view["c"], "h1")
        self.assertEqual(sim.checkpoints[2]["before"]["actual"]["c"], "h1")
        self.assertEqual(old.status, REJECTED_STALE)
        self.assertEqual(sim.counts["rejected_stale"], 1)
        self.assertEqual(sim.m_act["c"], "h2")

    def test_failure_compensates_before_invalid_successor_can_start(self):
        sim = _test_failed_predecessor()
        first, successor, retry = sim.candidates
        self.assertEqual(
            (first.status, successor.status, retry.status), (CANCELLED, CANCELLED, ADMITTED)
        )
        self.assertIsNone(successor.actions[0].start)
        self.assertFalse(successor.actions[0].done)
        state_after_failure = next(
            row["after"] for row in sim.checkpoints if row["after"]["time"] == 7
        )
        self.assertEqual(state_after_failure["actual"], {"c": "h0"})
        self.assertEqual(state_after_failure["lifecycle"], {"c": DEP})
        self.assertEqual(state_after_failure["ledger"], {})
        self.assertEqual(state_after_failure["active"], [])
        self.assertEqual(
            (
                sim.counts["failures"],
                sim.counts["compensations"],
                sim.counts["precondition_failures"],
            ),
            (1, 1, 1),
        )

    def test_failure_recovery_replans_from_actual_source_in_new_instance(self):
        sim = _test_failed_predecessor()
        self.assertEqual(
            [row["cause"] for row in sim.recovery_obligations], ["failure", "precondition"]
        )
        self.assertEqual([s.candidate_ids for s in sim.sessions], [[0, 1], [2]])
        retry = sim.candidates[2].actions[0]
        self.assertEqual(
            (retry.source, retry.destination, retry.start, retry.finish), ("h0", "h2", 8, 12)
        )
        self.assertEqual(sim.shared_request.finished, 12)
        self.assertEqual((sim.counts["completed_actions"], sim.m_act["c"]), (1, "h2"))

    def test_round_trip_remains_ordered_across_duration_and_executor_delay(self):
        for opens, duration in ((3, 1), (5, 4), (9, 7)):
            with self.subTest(executor_opens=opens, duration=duration):
                sim = _test_round_trip(executor_opens=opens, duration=duration)
                self.assertEqual(
                    [c.actions[0].start for c in sim.candidates], [opens, opens + duration]
                )
                self.assertEqual(sim.shared_request.finished, opens + 2 * duration)
                self.assertEqual(sim.final_audit()["final_consistent"], 1)

    def test_every_controlled_trace_finishes_without_capacity_or_reservation_leak(self):
        for name, function, _ in _REVISION_SCENARIOS:
            with self.subTest(scenario=name):
                sim = function()
                self.assertEqual(sim.counts["capacity_violations"], 0)
                self.assertEqual(sim.counts["leaked_reservations"], 0)
                self.assertEqual(sim.final_audit()["final_consistent"], 1)
                self.assertIsNotNone(sim.shared_request.finished)


# Revision generations


class GenerationFrontierTests(unittest.TestCase):
    @staticmethod
    def simulator():
        sim = ScopedGenerationContract(
            {"h0": 4, "h1": 4, "h2": 4},
            {"c": dict(host="h0", demand=2, duration=3)},
            [],
            planners=1,
        )
        request = Request(0, 0, ("c",), 1, lambda *args: ({"c": "h1"}, set()))
        return sim, request

    @staticmethod
    def present(sim, request, time, destination, generation, key=(0, 0)):
        cand = GeneratedCandidate(
            len(sim.candidates),
            request,
            0,
            0,
            time,
            sim.snapshot(time),
            {"c": destination},
            set(),
            generation_key=key,
            generation=generation,
        )
        sim.candidates.append(cand)
        sim.counts["candidates"] += 1
        return cand, sim._admit(cand, time)

    def test_superseded_revision_is_discarded_without_retracting_admitted_work(self):
        sim, request = self.simulator()
        sim.publish_generation((0, 0), 0, 0)
        first, status = self.present(sim, request, 0, "h1", 0)
        self.assertEqual(status, ADMITTED)
        sim.publish_generation((0, 0), 1, 1)
        stale, status = self.present(sim, request, 1, "h2", 0)
        self.assertEqual(status, REJECTED_SUPERSEDED)
        self.assertEqual(stale.actions, [])
        self.assertEqual(sim._admit(first, 1), ADMITTED)
        current, status = self.present(sim, request, 1, "h2", 1)
        self.assertEqual(status, ADMITTED)
        self.assertTrue(current.serialized)
        result = sim.run()
        self.assertEqual(result["rejected_superseded"], 1)
        self.assertEqual((sim.m_act["c"], result["final_consistent"]), ("h2", 1))

    def test_candidate_ahead_of_the_published_frontier_is_an_error(self):
        sim, request = self.simulator()
        with self.assertRaises(ValueError):
            self.present(sim, request, 0, "h1", 0)
        sim.publish_generation((0, 0), 0, 0)
        with self.assertRaises(ValueError):
            self.present(sim, request, 0, "h1", 1)
        with self.assertRaises(ValueError):
            sim.publish_generation((0,), 0, 0)


def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    require(suite.countTestCases() > 0, "No behavioral tests found")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    require(result.wasSuccessful(), "Behavioral tests failed")
    return dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("results", "tests", "replay", "all"))
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction/validation")
    args = parser.parse_args()
    started = time.monotonic()
    result = dict(status="running", command=args.command)
    try:
        if args.command in ("results", "all"):
            result["integrity"] = data_integrity()
            result["timing"] = timing()
            result["costs"] = costs()
            result["placement"] = placement()
            result["duration_sensitivity"] = sensitivity()
        if args.command in ("tests", "all"):
            result["tests"] = run_tests()
        if args.command in ("replay", "all"):
            result["replay"] = replay(args.output / "replay")
        result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / (args.command + ".json")).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
