"""Run timing, resource-cost and placement experiments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import statistics
import time

from scipy.stats import t

from simulation import (
    SAFETY_METRICS,
    WORKLOAD,
    BatchAblationSimulator,
    BatchedNetworkDagSimulator,
    ExactTimingSimulator,
    completed_action_observations,
    failure_model,
    make_workload,
    stream,
)
from theory import build, pathwise

ROOT = Path(__file__).resolve().parent
N = 200
WINDOW = 100
REPLICATIONS = 30
# Replication r of every campaign uses the seed r.
FIXED_GRID = tuple(range(25, 501, 25))
VARIABLE_GRID = tuple(range(50, 901, 50))
SURFACE_GRID = tuple(range(50, 501, 50))
TIMING_CONTROLLERS = ("sequential", "macpe1", "sufficient")
VARIABLE_CONTROLLERS = ("sequential", "macpe1", "macpe2", "macpe3", "sufficient")
COST_CONTROLLERS = (("sequential", 1), ("macpe", 1), ("macpe", 3))
CAPACITY_SCALES = (0.5, 0.75, 1.0, 2.0, 4.0, 16.0)
PLACEMENT_CONTROLLERS = ((1, "sequential"), (1, "macpe"), (2, "macpe"), (3, "macpe"), (4, "macpe"))
# Campaign C duration profiles: inclusive bounds of the uniform Planning and action durations.
# The long values are those of the recorded placement workload.
DURATION_PROFILES = {
    "short_short": dict(planning=(2, 4), action_duration=(2, 6)),
    "long_short": dict(planning=(20, 40), action_duration=(2, 6)),
    "short_long": dict(planning=(2, 4), action_duration=(20, 60)),
    "long_long": dict(planning=(20, 40), action_duration=(20, 60)),
}
EXECUTORS = 16
FAILURE_PROBABILITY = 0.1
MAX_RECHECKS = 3
MAX_RECOMPUTATIONS = 5


def read_json(path):
    """Read a JSON file, gzip-compressed when its name ends with .gz."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def write_json(path, value):
    """Write compact JSON, gzip-compressed when the name ends with .gz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    if path.suffix == ".gz":
        with gzip.GzipFile(temporary, "wb", mtime=0) as stream:
            stream.write(text.encode())
    else:
        temporary.write_text(text)
    temporary.replace(path)


def rounded(value, digits=2):
    """Copy of a summary with every float rounded; per-run records keep full precision."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item, digits) for item in value]
    return value


# Timing inputs and exact event runs.
def timing_planner_count(spec):
    """Planners of a timing run.

    One for the baseline, the controller's k_P, or a sufficient count: one planner per
    request for Poisson arrivals, otherwise ceil(T_P / spacing) or ceil(T_P / window).
    """
    c = spec["controller"]
    if c == "sequential":
        return 1
    if c != "sufficient":
        return int(c[-1])
    if spec["window"]:
        return math.ceil(spec["planning"] / spec["window"])
    return N if spec["arrival"] == "poisson" else math.ceil(spec["planning"] / 2)


def timing_key(spec):
    """Checkpoint name of a timing run."""
    return "_".join(
        (
            str(spec[k])
            for k in (
                "family",
                "arrival",
                "window",
                "planning",
                "execution",
                "controller",
                "replication",
            )
        )
    )


def timing_inputs(spec):
    """Seed, arrivals, Planning durations and four component durations of a timing run.

    The seed is the replication number. Poisson arrivals have a mean gap of 120 slots and
    are binned into integer slots; variable durations are uniform on 1..planning and
    1..execution.
    """
    seed = spec["replication"]
    rng = stream(seed, "arrivals")
    if spec["arrival"] == "poisson":
        arrivals, clock = [], 0.0
        for _ in range(N):
            clock += rng.expovariate(1 / 120)
            arrivals.append(int(clock))
    else:
        arrivals = list(range(0, 2 * N, 2))
    rng = stream(seed, "durations")
    if spec["family"] == "variable":
        planning = [rng.randint(1, spec["planning"]) for _ in range(N)]
        execution = [[rng.randint(1, spec["execution"]) for _ in range(4)] for _ in range(N)]
    else:
        planning = [spec["planning"]] * N
        execution = [[spec["execution"]] * 4 for _ in range(N)]
    return seed, arrivals, planning, execution


def timing_interval(values):
    """Mean and 95% Student-t interval over run means."""
    values = [float(value) for value in values if value is not None]
    if not values:
        return dict(n=0, mean=None, ci_low=None, ci_high=None)
    mean = statistics.mean(values)
    if len(values) == 1:
        return dict(n=1, mean=mean, ci_low=None, ci_high=None)
    half = float(t.ppf(0.975, len(values) - 1)) * statistics.stdev(values) / len(values) ** 0.5
    return dict(n=len(values), mean=mean, ci_low=mean - half, ci_high=mean + half)


def timing_specifications():
    """All timing runs: fixed durations, variable durations, and the duration surface."""
    specs = []
    for window in (0, WINDOW):
        for p in FIXED_GRID:
            for arrival in ("periodic", "poisson"):
                for controller in TIMING_CONTROLLERS:
                    for rep in (1,) if arrival == "periodic" else range(1, REPLICATIONS + 1):
                        specs.append(
                            dict(
                                family="fixed",
                                arrival=arrival,
                                window=window,
                                planning=p,
                                execution=p,
                                controller=controller,
                                replication=rep,
                            )
                        )
        for p in VARIABLE_GRID:
            for controller in VARIABLE_CONTROLLERS:
                for rep in range(1, REPLICATIONS + 1):
                    specs.append(
                        dict(
                            family="variable",
                            arrival="periodic",
                            window=window,
                            planning=p,
                            execution=100,
                            controller=controller,
                            replication=rep,
                        )
                    )
        for p in SURFACE_GRID:
            for e in SURFACE_GRID:
                for controller in TIMING_CONTROLLERS:
                    specs.append(
                        dict(
                            family="surface",
                            arrival="periodic",
                            window=window,
                            planning=p,
                            execution=e,
                            controller=controller,
                            replication=1,
                        )
                    )
    return specs


def run_timing(spec):
    """Run one timing specification and check every request against the pathwise recurrence."""
    start = time.monotonic()
    seed, arrivals, planning, execution = timing_inputs(spec)
    w = make_workload(seed)
    w["hosts"] = {h: v * 1000 for h, v in w["hosts"].items()}
    w["link_capacities"] = {link: v * 1000 for link, v in w["link_capacities"].items()}
    total_demand = sum(c["demand"] for c in w["components"].values())
    assert min(w["hosts"].values()) > total_demand
    assert min(w["link_capacities"].values()) > N * 15
    for i, r in enumerate(w["requests"]):
        r.arrival, r.planning = arrivals[i], planning[i]
        r.durations = dict(zip(r.scope, execution[i]))
    k = timing_planner_count(spec)
    sim = ExactTimingSimulator(
        w["hosts"],
        w["components"],
        w["requests"],
        applications=w["applications"],
        distances=w["distances"],
        link_capacities=w["link_capacities"],
        edge_demands=w["edge_demands"],
        planners=k,
        policy="sequential" if spec["controller"] == "sequential" else "macpe",
        executors=N,
        window=spec["window"],
        seed=seed,
    )
    metrics = sim.run()
    assert metrics["completed_requests"] == N and metrics["lost_requests"] == 0
    assert metrics["final_consistent"] == 1
    for name in SAFETY_METRICS:
        assert metrics[name] == 0, (timing_key(spec), name, metrics[name])
    for name in (
        "executor_waited_waves",
        "rejected_stale",
        "rejected_infeasible",
        "recomputed",
        "serialized",
        "delayed",
    ):
        assert metrics[name] == 0, (timing_key(spec), name, metrics[name])
    reference = pathwise(
        arrivals, planning, execution, k, spec["window"], spec["controller"] == "sequential"
    )
    actual_waits = [r.started - r.arrival for r in w["requests"]]
    actual_deployments = [r.finished - r.arrival for r in w["requests"]]
    assert actual_waits == reference["waits"]
    assert actual_deployments == reference["deployment"]
    assert len(sim.batches) == len(reference["batches"])
    for observed, expected in zip(sim.batches, reference["batches"]):
        assert observed["start"] == expected["start"]
        assert observed["members"] == expected["members"]
    data = dict(
        spec=spec,
        planners=k,
        executors=N,
        mean_TW=statistics.mean(actual_waits),
        mean_TD=statistics.mean(actual_deployments),
        completed=N,
        batches=len(sim.batches),
        metrics=metrics,
        input=dict(arrivals=arrivals, planning=planning, execution=execution),
        observations=dict(waiting=actual_waits, deployment=actual_deployments),
        pathwise_reference=dict(
            mean_TW=reference["mean_TW"],
            mean_TD=reference["mean_TD"],
            all_individual_waits_match=True,
            all_individual_deployments_match=True,
            all_batch_memberships_and_starts_match=True,
        ),
        runtime_seconds=time.monotonic() - start,
    )
    return data


def summarize_timing(rows):
    """Group timing runs by configuration and summarize their means and extrema."""
    groups = {}
    names = ("family", "arrival", "window", "planning", "execution", "controller")
    for row in rows:
        groups.setdefault(tuple(row["spec"][k] for k in names), []).append(row)
    summary = []

    def run_interval(values):
        return dict(timing_interval(values), min=min(values), max=max(values))

    for identity, group in sorted(groups.items()):
        summary.append(
            dict(
                zip(names, identity),
                planners=group[0]["planners"],
                mean_TW=run_interval([r["mean_TW"] for r in group]),
                mean_TD=run_interval([r["mean_TD"] for r in group]),
                completed=N,
                request_TW_min=min((min(r["observations"]["waiting"]) for r in group)),
                request_TW_max=max((max(r["observations"]["waiting"]) for r in group)),
                request_TD_min=min((min(r["observations"]["deployment"]) for r in group)),
                request_TD_max=max((max(r["observations"]["deployment"]) for r in group)),
                reference_TW=statistics.mean(r["pathwise_reference"]["mean_TW"] for r in group),
                reference_TD=statistics.mean(r["pathwise_reference"]["mean_TD"] for r in group),
            )
        )
    result = dict(
        protocol=dict(
            hosts=100,
            applications=N,
            components_per_application=4,
            capacity_multiplier=1000,
            initial_state="empty",
            all_pending_batches=True,
            planning_duration="first queued request's independent draw; constant per batch",
            execution_duration="four independent component durations per application; parallel execution",
            periodic_arrivals="0,2,...,398; complete drainage",
            poisson_arrivals="first 200 rate-1/120 arrivals binned into integer slots; complete drainage",
            windows=[0, WINDOW],
            replications=REPLICATIONS,
            executors=N,
            failures=False,
            coordination_delay=0,
            fixed_planning_grid=list(FIXED_GRID),
            variable_planning_grid=list(VARIABLE_GRID),
            surface_planning_execution_grid=list(SURFACE_GRID),
            variability="confidence intervals over run means; minima and maxima over individual requests",
            finite_input_effect="200 applications; finite workload averages, not stationary values",
        ),
        runs=len(rows),
        summary=summary,
        checks=dict(
            all200completed=True,
            no_resource_or_executor_waits=True,
            all_individual_waits_deployments_and_batch_starts_equal_independent_recurrence=True,
            safety_metrics={name: max(r["metrics"][name] for r in rows) for name in SAFETY_METRICS},
        ),
    )
    return result


def iter_timing_runs(directory):
    """Read archived compressed records or fresh per-run checkpoints."""
    directory = Path(directory)
    compressed = directory / "runs.jsonl.gz"
    if compressed.exists():
        with gzip.open(compressed, "rt") as stream:
            for line in stream:
                yield json.loads(line)
    else:
        for path in sorted((directory / "runs").glob("*.json")):
            yield json.loads(path.read_text())


# Resource contention and incremental coordination delay.
def cost_interval(values):
    """Mean and 95% Student-t half-width; zero for a single value."""
    values = list(values)
    mean = statistics.mean(values)
    half = (
        float(t.ppf(0.975, len(values) - 1)) * statistics.stdev(values) / len(values) ** 0.5
        if len(values) > 1
        else 0.0
    )
    return dict(mean=mean, half_width=half, n=len(values))


def cost_timing_reference(arrivals, planners, policy, coordination):
    """Expected Planning start and completion of each request when resources are ample."""
    free = [0] * planners
    index, rows = 0, []
    while index < len(arrivals):
        planner = min(range(planners), key=lambda i: (free[i], i))
        start = max(arrivals[index], free[planner])
        end = index
        while end < len(arrivals) and arrivals[end] <= start:
            rows.append(
                dict(id=end, planning_started=start, completed=start + 30 + coordination + 60)
            )
            end += 1
        free[planner] = start + 30 + coordination + (60 if policy == "sequential" else 0)
        index = end
    return rows


def run_cost(spec):
    """Run one capacity or coordination-cost specification."""
    started = time.perf_counter()
    seed = spec["replication"]
    work = make_workload(seed, "frequent")
    scale = spec["capacity_scale"]
    work["hosts"] = {key: max(1, round(value * scale)) for key, value in work["hosts"].items()}
    work["link_capacities"] = {
        key: max(1, round(value * scale)) for key, value in work["link_capacities"].items()
    }
    for request in work["requests"]:
        request.arrival = 10 * (request.id + 1)
        request.planning = 30
        request.durations = dict.fromkeys(request.scope, 60)
    for component in work["components"].values():
        component["duration"] = 60
    sim = BatchedNetworkDagSimulator(
        **{
            name: work[name]
            for name in (
                "hosts",
                "components",
                "requests",
                "applications",
                "distances",
                "link_capacities",
                "edge_demands",
            )
        },
        planners=spec["planners"],
        policy=spec["policy"],
        executors=16,
        coordination=spec["coordination"],
        seed=seed,
        max_retries=3,
        max_recomputations=5,
    )
    metrics = sim.run()
    assert metrics["final_consistent"] == 1
    for name in SAFETY_METRICS:
        assert metrics[name] == 0, (spec, name, metrics[name])
    assert metrics["completed_requests"] + metrics["lost_requests"] == 200
    assert not sim.active and not sim.ledger and not sim.queue and not sim.waiting_waves
    outcomes = [
        dict(
            id=r.id,
            arrival=r.arrival,
            planning_started=r.started,
            completed=r.finished,
            TW=None if r.started is None else r.started - r.arrival,
            TD=None if r.finished is None else r.finished - r.arrival,
            exit=getattr(r, "outcome_reason", None) or ("lost" if r.lost else "completed"),
        )
        for r in work["requests"]
    ]
    metrics.update(
        mean_TW=statistics.mean(r["TW"] for r in outcomes if r["TW"] is not None),
        mean_TD=metrics["response_mean"],
        completion_percent=metrics["completed_requests"] / 2,
        planning_batches=len(sim.batches),
    )
    reference = None
    if spec["family"] == "coordination":
        reference = cost_timing_reference(
            [r["arrival"] for r in outcomes], spec["planners"], spec["policy"], spec["coordination"]
        )
        assert metrics["completed_requests"] == 200
        for actual, expected in zip(outcomes, reference):
            for key in ("id", "planning_started", "completed"):
                assert actual[key] == expected[key], (spec, key, actual, expected)
        assert metrics["executor_wait_mean"] == 0
    return dict(
        **spec,
        metrics=metrics,
        outcomes=outcomes,
        exit_counts=dict(Counter(row["exit"] for row in outcomes)),
        batches=sim.batches,
        finite_trace_reference=reference,
        elapsed_seconds=time.perf_counter() - started,
    )


def cost_summaries(runs):
    """Group cost runs and compute intervals and paired common-success savings."""
    groups = defaultdict(list)
    for row in runs:
        groups[
            row["family"],
            row["capacity_scale"],
            row["coordination"],
            row["policy"],
            row["planners"],
        ].append(row)
    summary = []
    for (family, scale, cost, policy, planners), rows in sorted(groups.items()):
        item = dict(
            family=family, capacity_scale=scale, coordination=cost, policy=policy, planners=planners
        )
        for metric in (
            "mean_TW",
            "mean_TD",
            "completion_percent",
            "completed_requests",
            "lost_requests",
            "planning_batches",
            "planning_infeasible",
            "planning_search_exhausted",
            "rejected_infeasible",
            "rejected_stale",
            "delayed",
            "executor_wait_mean",
        ):
            item[metric] = cost_interval((r["metrics"].get(metric, 0) for r in rows))
        categories = sorted({category for r in rows for category in r["exit_counts"]})
        item["exit_counts"] = {
            cat: cost_interval((r["exit_counts"].get(cat, 0) for r in rows)) for cat in categories
        }
        if policy == "macpe":
            baseline_cost = 0
            baseline = {
                r["replication"]: r for r in groups[family, scale, baseline_cost, "sequential", 1]
            }
            paired, intersections = [], []
            for row in rows:
                old = {
                    r["id"]: r
                    for r in baseline[row["replication"]]["outcomes"]
                    if r["completed"] is not None
                }
                overlap = [
                    r for r in row["outcomes"] if r["completed"] is not None and r["id"] in old
                ]
                intersections.append(len(overlap))
                if overlap:
                    before = statistics.mean(old[r["id"]]["TD"] for r in overlap)
                    after = statistics.mean(r["TD"] for r in overlap)
                    paired.append(100 * (before - after) / before)
            item["common_success_count"] = cost_interval(intersections)
            item["common_success_TD_saving_percent"] = cost_interval(paired)
        summary.append(item)
    return summary


def cost_specifications(replications=30, scales=CAPACITY_SCALES, costs=tuple(range(0, 61, 10))):
    """Capacity sweep over replications, then the deterministic coordination-cost sweep."""
    specs = [
        dict(
            family="capacity",
            capacity_scale=scale,
            coordination=0,
            replication=rep,
            policy=policy,
            planners=k,
        )
        for rep in range(1, replications + 1)
        for scale in scales
        for policy, k in COST_CONTROLLERS
    ]
    specs += [
        dict(
            family="coordination",
            capacity_scale=1000.0,
            coordination=cost,
            replication=1,
            policy=policy,
            planners=k,
        )
        for policy, k in COST_CONTROLLERS
        for cost in ([0] if policy == "sequential" else costs)
    ]
    return specs


def cost_identity(spec):
    """Checkpoint name of a cost run."""
    return "_".join(
        str(spec[k])
        for k in ("family", "replication", "capacity_scale", "coordination", "policy", "planners")
    )


# Physical completion and placement campaigns.
def expected_scopes(run):
    """Requested components of each original request."""
    result = {}
    for row in run["outcomes"]:
        if "components" in row:
            scope = tuple(row["components"])
        else:
            app = row.get("application")
            if not isinstance(app, str) or not app.startswith("app") or (not app[3:].isdigit()):
                raise ValueError("supply explicit requested components for nonstandard workloads")
            scope = tuple(f"{app}.c{index}" for index in range(4))
        if not scope or len(scope) != len(set(scope)):
            raise ValueError("invalid original application scope")
        result[row["id"]] = frozenset(scope)
    return result


def physical_completion_view(raw, *, scopes=None):
    """Recompute request completion from successful component actions.

    The controller's own record is kept alongside, under controller_* keys.
    """
    if raw.get("observation", {}).get("definition") == "effective_component_completion":
        return deepcopy(raw)
    if "completed_action_outcomes" not in raw:
        raise ValueError("effective completion requires successful action traces")
    scopes = (
        expected_scopes(raw) if scopes is None else {k: frozenset(v) for k, v in scopes.items()}
    )
    ids = {row["id"] for row in raw["outcomes"]}
    if set(scopes) != ids:
        raise ValueError("original request scopes do not match outcomes")
    if len(set().union(*scopes.values())) != sum(map(len, scopes.values())):
        raise ValueError("this observer requires disjoint deployment scopes")
    completed = {identity: {} for identity in ids}
    seen_actions = set()
    for action in raw["completed_action_outcomes"]:
        if action["kind"] != "deploy":
            raise ValueError("physical deployment observer does not handle relocation/deletion")
        identity, component = action["request"], action["component"]
        if identity not in ids or component not in scopes[identity]:
            raise ValueError("completed action does not belong to its original request")
        if action["id"] in seen_actions or component in completed[identity]:
            raise ValueError("duplicate successful deployment; cannot infer unchanged lifetime")
        seen_actions.add(action["id"])
        if (
            action["started"] is None
            or action["finished"] is None
            or action["finished"] <= action["started"]
        ):
            raise ValueError("invalid completed-action timestamps")
        completed[identity][component] = action["finished"]
    if raw.get("ablation", "full") == "full":
        metrics = raw["metrics"]
        if metrics["final_consistent"] != 1 or metrics.get("final_dag_violations", 0):
            raise ValueError("full-mechanism final deployment is not valid")
    result = deepcopy(raw)
    result["controller_metrics"] = deepcopy(raw["metrics"])
    result["controller_exit_counts"] = deepcopy(raw["exit_counts"])
    late = []
    for row in result["outcomes"]:
        finishes = completed[row["id"]]
        finish = max(finishes.values()) if set(finishes) == set(scopes[row["id"]]) else None
        if not row["lost"] and (finish is None or finish != row["completed"]):
            raise AssertionError("recorded successful request disagrees with component trace")
        row.update(
            controller_lost=row["lost"],
            controller_completed=row["completed"],
            controller_exit=row["exit"],
            components=sorted(scopes[row["id"]]),
        )
        if finish is not None:
            if row["lost"]:
                late.append(row["id"])
            row.update(
                lost=False, completed=finish, response=finish - row["arrival"], exit="completed"
            )
        else:
            row.update(lost=True, completed=None, response=None)
    successes = [row for row in result["outcomes"] if not row["lost"]]
    if "final_deployed_applications" in raw["metrics"]:
        if raw["metrics"]["final_deployed_applications"] != len(successes):
            raise AssertionError(
                "component traces disagree with the final deployed-application audit"
            )
    responses = [row["response"] for row in successes]
    ordered = sorted(responses)
    mean = sum(responses) / len(responses) if responses else None
    result["metrics"].update(
        completed_requests=len(successes),
        lost_requests=len(ids) - len(successes),
        response_mean=mean,
        mean_TD=mean,
        deployment_mean=mean,
        response_p95=ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else None,
        mean_TW_success=(
            sum(row["first_planner_wait"] for row in successes) / len(successes)
            if successes
            else None
        ),
        completed_after_controller_stop=len(late),
    )
    result["exit_counts"] = dict(Counter(row["exit"] for row in result["outcomes"]))
    result["observation"] = dict(
        definition="effective_component_completion",
        completed_after_controller_stop=late,
        raw_record_preserved=True,
        timing="max successful component completion minus original request arrival",
        controller_status="retained separately; stopping retries does not cancel running actions",
    )
    return result


def placement_identity(spec):
    """Checkpoint name of a placement run."""
    identity = "r{replication:02d}_w{window}_{arrival_mode}_{policy}_k{planners}_{ablation}".format(
        **spec
    )
    if "duration_profile" in spec:
        identity += "_" + spec["duration_profile"]
    return identity


def placement_workload(spec):
    """Placement workload of a replication, with the durations of its profile when one is given.

    Short durations come from generators keyed by the seed and the phase, so a request keeps
    the same short Planning duration, and a component the same short action duration, in
    every profile and under every controller. Long durations are the recorded workload's own.
    """
    profile = spec.get("duration_profile")
    if "duration_profile" in spec and profile not in DURATION_PROFILES:
        raise ValueError(f"Unknown duration profile: {profile}")
    seed = spec["replication"]
    work = make_workload(seed, spec["arrival_mode"])
    if profile is None:
        return work
    durations = DURATION_PROFILES[profile]
    if durations["planning"] != WORKLOAD["planning"]:
        rng = stream(seed, "duration_sensitivity", "planning")
        for request in work["requests"]:
            request.planning = rng.randint(*durations["planning"])
    if durations["action_duration"] != WORKLOAD["action_duration"]:
        rng = stream(seed, "duration_sensitivity", "action")
        for component in work["components"].values():
            component["duration"] = rng.randint(*durations["action_duration"])
        for request in work["requests"]:
            request.durations = {
                component: work["components"][component]["duration"] for component in request.scope
            }
    return work


def _run_full_placement(spec):
    """Run one placement specification with every mechanism enabled."""
    seed = spec["replication"]
    work = placement_workload(spec)
    sim = BatchedNetworkDagSimulator(
        **{
            name: work[name]
            for name in (
                "hosts",
                "components",
                "requests",
                "applications",
                "distances",
                "link_capacities",
                "edge_demands",
            )
        },
        planners=spec["planners"],
        policy=spec["policy"],
        executors=EXECUTORS,
        window=spec["window"],
        seed=seed,
        failure=failure_model(FAILURE_PROBABILITY),
        max_retries=MAX_RECHECKS,
        max_recomputations=MAX_RECOMPUTATIONS,
    )
    started = time.perf_counter()
    metrics = sim.run()
    for name in SAFETY_METRICS:
        assert metrics[name] == 0, (spec, name, metrics[name])
    assert metrics["final_consistent"] == 1, (spec, sim.final_audit())
    assert not sim.active and not sim.ledger and not sim.queue and not sim.waiting_waves
    assert not sim.delayed
    requests = work["requests"]
    assert all(((request.finished is None) == request.lost for request in requests))
    assert metrics["completed_requests"] + metrics["lost_requests"] == len(requests)
    assert len({r.id for r in requests}) == len(requests)
    migration_metrics, actions = completed_action_observations(sim)
    owner = {component: request.id for request in requests for component in request.scope}
    for action in actions:
        action["batch"] = action.pop("request")
        action["request"] = owner[action["component"]]
    metrics.update(migration_metrics)
    outcomes = [
        dict(
            id=r.id,
            application=r.application,
            operation=r.operation,
            arrival=r.arrival,
            planning_started=r.started,
            completed=r.finished,
            lost=r.lost,
            response=None if r.finished is None else r.finished - r.arrival,
            first_planner_wait=None if r.started is None else r.started - r.arrival,
            recomputations=r.recomputations,
            exit=getattr(r, "outcome_reason", None) or ("lost" if r.lost else "completed"),
        )
        for r in requests
    ]
    exits = dict(Counter(row["exit"] for row in outcomes))
    metrics.update(
        mean_TW=sum(r.started - r.arrival for r in requests) / len(requests),
        mean_TD=metrics["response_mean"],
        planning_batches=len(sim.batches),
        max_batch_size=max((len(b["members"]) for b in sim.batches)),
    )
    run = dict(
        **spec,
        executors=EXECUTORS,
        metrics=metrics,
        outcomes=outcomes,
        completed_action_outcomes=actions,
        batches=sim.batches,
        exit_counts=exits,
        runtime_seconds=time.perf_counter() - started,
    )

    successes = [r for r in run["outcomes"] if not r["lost"]]
    run["metrics"]["mean_TW_success"] = (
        sum(r["first_planner_wait"] for r in successes) / len(successes) if successes else None
    )
    return run


def run_placement(spec):
    """Run one placement specification, with every mechanism or with one disabled."""
    if spec["ablation"] == "full":
        raw = _run_full_placement(spec)
    else:
        seed = spec["replication"]
        work = make_workload(seed, spec["arrival_mode"])
        sim = BatchAblationSimulator(
            **{
                k: work[k]
                for k in (
                    "hosts",
                    "components",
                    "requests",
                    "applications",
                    "distances",
                    "link_capacities",
                    "edge_demands",
                )
            },
            planners=spec["planners"],
            policy=spec["policy"],
            executors=EXECUTORS,
            window=spec["window"],
            seed=seed,
            ablation=spec["ablation"],
            failure=failure_model(FAILURE_PROBABILITY),
            max_retries=MAX_RECHECKS,
            max_recomputations=MAX_RECOMPUTATIONS,
        )
        began = time.perf_counter()
        metrics = sim.run()
        assert not sim.active and not sim.queue and not sim.waiting_waves and not sim.delayed
        assert metrics["completed_requests"] + metrics["lost_requests"] == len(sim.requests)
        for key in (
            "invalid_transitions",
            "duplicate_placements",
            "distinct_host_violations",
            "executor_capacity_violations",
            "dag_latency_violations",
        ):
            assert metrics[key] == 0, (spec, key, metrics[key])
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
                first_planner_wait=None if r.started is None else r.started - r.arrival,
                recomputations=r.recomputations,
                exit=getattr(r, "outcome_reason", None) or ("lost" if r.lost else "completed"),
            )
            for r in sim.requests
        ]
        successes = [r for r in outcomes if not r["lost"]]
        metrics.update(
            mean_TW=sum(r.started - r.arrival for r in sim.requests) / len(sim.requests),
            mean_TW_success=(
                sum(r["first_planner_wait"] for r in successes) / len(successes)
                if successes
                else None
            ),
            mean_TD=metrics["response_mean"],
            planning_batches=len(sim.batches),
            max_batch_size=max((len(b["members"]) for b in sim.batches)),
        )
        raw = dict(
            **spec,
            executors=EXECUTORS,
            metrics=metrics,
            outcomes=outcomes,
            completed_action_outcomes=actions,
            batches=sim.batches,
            exit_counts=dict(Counter(r["exit"] for r in outcomes)),
            runtime_seconds=time.perf_counter() - began,
        )
    physical_completion_view(raw)
    return raw


def placement_specifications(replications=30, windows=(0,), ablations=False):
    """Placement runs for every controller, or the ablation variants at k_P = 3."""
    configs = (
        [
            (arrival, k, policy, "full")
            for arrival in ("frequent", "sparse")
            for k, policy in PLACEMENT_CONTROLLERS
        ]
        if not ablations
        else [
            ("frequent", 3, "macpe", variant)
            for variant in ("full", "no_aggregate", "no_fresh", "no_compensation")
        ]
    )
    return [
        dict(
            replication=r,
            window=w,
            arrival_mode=arrival,
            planners=k,
            policy=policy,
            ablation=variant,
        )
        for r in range(1, replications + 1)
        for w in windows
        for arrival, k, policy, variant in configs
    ]


def sensitivity_specifications(replications=30):
    """Four duration profiles with sequential control and one, two or four planners."""
    return [
        dict(
            replication=replication,
            window=0,
            arrival_mode="frequent",
            planners=planners,
            policy=policy,
            ablation="full",
            duration_profile=profile,
        )
        for replication in range(1, replications + 1)
        for profile in DURATION_PROFILES
        for planners, policy in ((1, "sequential"), (1, "macpe"), (2, "macpe"), (4, "macpe"))
    ]


# Checkpointed execution and command-line interface.
def _execute(specs, runner, identity, output, workers, resume=False, reused=None):
    """Run the missing specifications with one checkpoint file per run; resume if asked."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected = dict(specifications=specs)
    configuration = output / "configuration.json"
    if configuration.exists():
        if not resume or json.loads(configuration.read_text()) != expected:
            raise ValueError("Existing campaign requires --resume and unchanged inputs.")
    else:
        if any(output.iterdir()):
            raise ValueError("Choose an empty output directory.")
        write_json(configuration, expected)
    checkpoint_dir = output / "runs" if runner is run_timing else output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    records = {}
    reused = reused or {}
    for spec in specs:
        key = identity(spec)
        path = checkpoint_dir / (key + ".json")
        if path.exists():
            records[key] = json.loads(path.read_text())
            if key in reused and records[key] != reused[key]:
                raise ValueError("Reused baseline differs from its checkpoint.")
        elif key in reused:
            records[key] = reused[key]
            write_json(path, records[key])
        if key in records:
            actual = records[key].get("spec", records[key])
            if any(actual.get(name) != value for name, value in spec.items()):
                raise ValueError("Checkpoint input differs from the requested specification.")
    missing = [spec for spec in specs if identity(spec) not in records]
    print(f"{output.name}: {len(specs)} runs, {len(missing)} pending", flush=True)
    began = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(runner, spec): spec for spec in missing}
        for future in as_completed(jobs):
            row = future.result()
            key = identity(jobs[future])
            write_json(checkpoint_dir / (key + ".json"), row)
            records[key] = row
            if len(records) % 25 == 0 or len(records) == len(specs):
                print(
                    f"{len(records)}/{len(specs)} runs; {time.perf_counter()-began:.1f}s",
                    flush=True,
                )
    return [records[identity(spec)] for spec in specs]


def timing_campaign(output, workers=2, resume=False):
    """Campaign A: timing runs and their summary."""
    output = Path(output)
    runs = _execute(timing_specifications(), run_timing, timing_key, output, workers, resume)
    with gzip.GzipFile(output / "runs.jsonl.gz", "wb", mtime=0) as archive:
        for run in runs:
            archive.write((json.dumps(run, separators=(",", ":")) + "\n").encode())
    write_json(output / "summary.json", rounded(summarize_timing(runs)))
    return runs


def cost_campaign(output, workers=2, replications=30, resume=False):
    """Campaign B: capacity and coordination-cost runs, summary, and protocol."""
    output = Path(output)
    specs = cost_specifications(replications)
    began = time.perf_counter()
    runs = _execute(specs, run_cost, cost_identity, output, workers, resume)
    runs.sort(
        key=lambda r: (
            r["family"],
            r["replication"],
            r["capacity_scale"],
            r["coordination"],
            r["policy"],
            r["planners"],
        )
    )
    write_json(output / "runs.json.gz", runs)
    write_json(output / "summary.json", rounded(cost_summaries(runs)))
    protocol = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        host_count=100,
        physical_links=4950,
        applications=N,
        components_per_application=4,
        initial_state="empty hosts and links",
        admission="all pending applications",
        planning_duration=30,
        component_execution_duration=60,
        arrival_spacing=10,
        executors=16,
        failed_action_probability=0,
        capacity_scales=list(CAPACITY_SCALES),
        capacity_replications=replications,
        coordination_costs=list(range(0, 61, 10)),
        coordination_capacity_scale=1000,
        coordination_replications=1,
        coordination_reference="exact finite-stream all-pending recurrence",
        coordination_semantics="incremental simulated delay after planning; planner remains occupied until check",
        latency_population="successful requests; completion and paired common-success comparisons retained",
        maximum_rechecks=3,
        maximum_recomputations=5,
        run_count=len(runs),
        elapsed_seconds=time.perf_counter() - began,
    )
    write_json(output / "protocol.json", rounded(protocol))
    return runs


def placement_campaign(
    output, workers=2, replications=30, windows=(0,), ablations=False, reuse_full=None, resume=False
):
    """Campaign C: placement runs or mechanism ablations, effective completion, and protocol."""
    output = Path(output)
    specs = placement_specifications(replications, windows, ablations)
    reused = {}
    if ablations:
        if reuse_full is None:
            raise ValueError("Ablations require a completed full-placement campaign.")
        reference = Path(reuse_full)
        if not (reference / "runs.json.gz").exists():
            raise ValueError("The full-placement campaign has no runs.json.gz.")
        reused = {
            placement_identity(run): run
            for run in read_json(reference / "runs.json.gz")
            if run["arrival_mode"] == "frequent"
            and run["policy"] == "macpe"
            and run["planners"] == 3
            and run["window"] in windows
        }
        required = {placement_identity(spec) for spec in specs if spec["ablation"] == "full"}
        if not required.issubset(reused):
            raise ValueError("The full-placement campaign lacks required paired baselines.")
    began = time.perf_counter()
    runs = _execute(specs, run_placement, placement_identity, output, workers, resume, reused)
    effective = [physical_completion_view(run) for run in runs]
    write_json(output / "runs.json.gz", runs)
    write_json(output / "effective_runs.json.gz", effective)
    protocol = dict(
        campaign="ablations" if ablations else "full",
        replications=replications,
        windows=list(windows),
        specs=specs,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        run_count=len(runs),
        workload_parameters=WORKLOAD,
        executors=EXECUTORS,
        failure_probability=FAILURE_PROBABILITY,
        max_rechecks=MAX_RECHECKS,
        max_recomputations=MAX_RECOMPUTATIONS,
        instantaneous_checks=True,
        observation="physical completion of requested components; controller flags retained separately",
        wall_seconds=time.perf_counter() - began,
    )
    write_json(output / "protocol.json", rounded(protocol))
    return runs


def sensitivity_campaign(output, workers=2, replications=30, resume=False):
    """Campaign C: duration sensitivity with paired placement workloads."""
    output = Path(output)
    specs = sensitivity_specifications(replications)
    began = time.perf_counter()
    runs = _execute(specs, run_placement, placement_identity, output, workers, resume)
    write_json(output / "runs.json.gz", runs)
    protocol = dict(
        campaign="duration_sensitivity",
        replications=replications,
        seeds=list(range(1, replications + 1)),
        run_count=len(runs),
        duration_profiles=DURATION_PROFILES,
        duration_distribution="discrete uniform, inclusive bounds; simulated time slots",
        action_duration="duration of one component action, not an application deployment",
        arrival_mode="frequent",
        arrival_gap=WORKLOAD["arrival_gap"]["frequent"],
        window=0,
        controllers=[
            dict(policy=policy, planners=planners)
            for planners, policy in ((1, "sequential"), (1, "macpe"), (2, "macpe"), (4, "macpe"))
        ],
        workload_parameters=WORKLOAD,
        pairing=(
            "same infrastructure, application DAGs, demands and arrivals for every profile"
            " and controller of a seed"
        ),
        short_duration_sampling=(
            "generators keyed by seed and phase; shared across profiles and controllers"
        ),
        long_durations="recorded placement draws; long_long equals the frequent-arrival baseline",
        executors=EXECUTORS,
        failure_probability=FAILURE_PROBABILITY,
        max_rechecks=MAX_RECHECKS,
        max_recomputations=MAX_RECOMPUTATIONS,
        instantaneous_checks=True,
        observation="physical completion of requested components; controller flags retained",
        waiting_population="all original requests; arrival to first Planning start",
        response_population="completed requests; paired comparisons use common completions",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        wall_seconds=time.perf_counter() - began,
    )
    write_json(output / "protocol.json", rounded(protocol))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "campaign", choices=("timing", "costs", "placement", "ablations", "sensitivity", "full")
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction" / "full")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--replications",
        type=int,
        default=30,
        help="Independent repetitions for costs and placement.",
    )
    parser.add_argument("--windows", type=int, nargs="+", default=[0])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.replications < 2 or any(w < 0 for w in args.windows):
        parser.error("Use positive workers, at least two replications, and nonnegative windows.")
    if len(set(args.windows)) != len(args.windows):
        parser.error("Duplicate windows.")
    if args.campaign == "full" and 0 not in args.windows:
        parser.error("The full campaign requires window 0 for the placement tables.")
    if args.campaign in ("timing", "full"):
        timing_campaign(args.output / "timing", args.workers, args.resume)
        build(args.output / "timing" / "theory.json")
    if args.campaign in ("costs", "full"):
        cost_campaign(args.output / "costs", args.workers, args.replications, args.resume)
    if args.campaign in ("placement", "full"):
        placement_campaign(
            args.output / "placement" / "full",
            args.workers,
            args.replications,
            args.windows,
            resume=args.resume,
        )
    if args.campaign in ("ablations", "full"):
        placement_campaign(
            args.output / "placement" / "ablations",
            args.workers,
            args.replications,
            args.windows,
            ablations=True,
            reuse_full=args.output / "placement" / "full",
            resume=args.resume,
        )
    if args.campaign in ("sensitivity", "full"):
        sensitivity_campaign(
            args.output / "placement" / "sensitivity",
            args.workers,
            args.replications,
            resume=args.resume,
        )
    print(f"Results: {args.output}")
    if args.campaign == "full":
        print(f"Next: python report.py all --data {args.output} --output reproduction/outputs")


if __name__ == "__main__":
    main()
