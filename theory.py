"""Expected waiting and deployment times of the admission-cycle model.

Requests arrive at known instants, every `spacing` slots in the recorded campaigns. At each
gate a free planner takes every pending request as one batch and keeps it for the batch
Planning duration (MACPE-K) or until the batch's actions complete (sequential MAPE-K).
Window-based control opens gates only at multiples of `window`. The waiting time of a
request is the delay from its arrival to the start of its batch.

Two kinds of references are computed:

* `pathwise` replays one sequence of known durations exactly. It is the reference for the
  fixed-duration runs, and verify.py compares every recorded run with it.
* `reference` returns the expectation of the mean waiting time when batch Planning durations
  are uniform on 1..delta_P and component Execution durations uniform on 1..delta_E, for an
  initially empty system that serves exactly `requests` arrivals. Each controller has its
  own finite recurrence over the admission gates:
  - sequential MAPE-K: `sequential_waiting`, forward over gate times and batch sizes;
  - MACPE-K with one planner: `single_planner_waiting`, forward over gate times;
  - MACPE-K with two or three planners under event-driven control: `residual_waiting`,
    backward over the remaining horizon and the residual busy times of the other planners;
  - MACPE-K with several planners under window-based control: `window_waiting`, forward
    over window-aligned residuals with exact rational arithmetic;
  - sufficient Planning capacity: `sufficient_waiting`, requests only wait for the next
    window boundary.

`python theory.py` writes the reference table read by report.py; it takes a few minutes and
up to 2 GB of memory. `python theory.py --self-check` compares every recurrence with an
exhaustive enumeration of small cases.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import heapq
from itertools import product
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reproduction/full/timing/theory.json"
VARIABLE_GRID = tuple(range(50, 901, 50))
FIXED_GRID = tuple(range(25, 501, 25))
SURFACE_GRID = tuple(range(50, 501, 50))
CONTROLLERS = ("sequential", "macpe1", "macpe2", "macpe3", "sufficient")


# Replay of known durations.


def pathwise(arrivals, planning, execution, planners=1, window=0, sequential=False):
    """Replay one sequence of known durations; execution[i] lists the four action durations.

    Returns the waiting and deployment time of every request and the batches formed.
    """
    n = len(arrivals)
    if not (len(planning) == len(execution) == n):
        raise ValueError("one Planning duration and four action durations per request")
    if sequential and planners != 1:
        raise ValueError("the sequential baseline has one planner")
    available = [0] * planners  # heap of planner release times
    waits, deployment, batches = [0] * n, [0] * n, []
    i = 0
    while i < n:
        first = i
        start = max(arrivals[i], available[0])
        if window:
            start = ((start + window - 1) // window) * window
        while i < n and arrivals[i] <= start:
            i += 1
        duration = planning[first]
        for member in range(first, i):
            waits[member] = start - arrivals[member]
            deployment[member] = waits[member] + duration + max(execution[member])
        release = start + duration
        if sequential:
            release += max(max(e) for e in execution[first:i])
        heapq.heapreplace(available, release)
        batches.append(
            dict(
                start=start,
                members=list(range(first, i)),
                planning_duration=duration,
                release=release,
            )
        )
    return dict(
        mean_TW=sum(waits) / n,
        mean_TD=sum(deployment) / n,
        waits=waits,
        deployment=deployment,
        batches=batches,
    )


def expected_application_execution(execution_max):
    """Expected maximum of four independent uniform durations on 1..execution_max."""
    return sum(1 - (m / execution_max) ** 4 for m in range(execution_max))


# Shared helpers.


def _check_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _check_window(window, spacing):
    if isinstance(window, bool) or not isinstance(window, int) or window < 0:
        raise ValueError("window must be a nonnegative integer")
    if window and window % spacing:
        raise ValueError("window must be a multiple of the arrival spacing")


def _arrival_count(horizon, spacing):
    """Number of arrivals at 0, spacing, 2 * spacing, ... before the horizon."""
    return (horizon + spacing - 1) // spacing


def _batch_waiting(h, gap, d, spacing):
    """Total waiting of the requests admitted at a gate opening d slots from now.

    Arrivals occur gap, gap + spacing, ... slots from now; ceil(h / spacing) of them remain.
    """
    remaining = (h + spacing - 1) // spacing
    admitted = np.where(d >= gap, np.minimum(remaining, (d - gap) // spacing + 1), 0)
    return admitted * (d - gap) - spacing * admitted * (admitted - 1) // 2


# One planner.


def single_planner_waiting(durations, horizon, spacing=2, window=0):
    """Mean waiting time with one MACPE-K planner and equiprobable Planning durations.

    The state is the time of the current gate. The planner is busy for the batch's Planning
    duration; the next gate opens at its release, or at the next arrival if that comes later,
    rounded up to a window boundary under window-based control. The probability mass of the
    gate time moves forward and the waiting of every admitted request is accumulated.
    """
    _check_positive(horizon, "horizon")
    _check_positive(spacing, "spacing")
    _check_window(window, spacing)
    durations = tuple(durations)
    if not durations:
        raise ValueError("durations must contain at least one value")
    for duration in durations:
        _check_positive(duration, "duration")
    count = _arrival_count(horizon, spacing)
    last_arrival = (count - 1) * spacing
    if count == 1:
        return 0.0
    holds = np.asarray(durations, dtype=np.int64)
    if window:
        holds = np.maximum(window, ((holds + window - 1) // window) * window)
    holds, multiplicity = np.unique(holds, return_counts=True)
    weights = multiplicity / len(durations)
    mass = np.zeros(last_arrival + 1)
    mass[0] = 1.0
    total_waiting = terminal = 0.0
    for gate in range(last_arrival):
        probability = mass[gate]
        if not probability:
            continue
        served = gate // spacing  # index of the last arrival at or before this gate
        next_arrival = (served + 1) * spacing
        earliest = next_arrival
        if window:
            earliest = ((next_arrival + window - 1) // window) * window
        next_gates = np.maximum(gate + holds, earliest)
        last_served = np.minimum(next_gates, last_arrival) // spacing
        admitted = last_served - served
        # waiting of arrivals served + 1 .. last_served, all admitted at the next gate
        waiting = (
            admitted * next_gates
            - spacing * (last_served * (last_served + 1) - served * (served + 1)) / 2
        )
        total_waiting += float(probability * (weights @ waiting))
        continuing = next_gates < last_arrival
        np.add.at(mass, next_gates[continuing], probability * weights[continuing])
        terminal += float(probability * weights[~continuing].sum())
    if abs(terminal - 1.0) > 1e-8:
        raise ArithmeticError("the gate recurrence lost probability mass")
    return total_waiting / count


# Sequential baseline.


def sequential_waiting(planning_max, execution_max, horizon, spacing=2, window=0):
    """Mean waiting time of sequential MAPE-K with uniform Planning and Execution durations.

    The state is (gate time, batch size). The planner is held for the batch Planning duration,
    uniform on 1..planning_max, plus the batch makespan, the maximum of four Execution
    durations per admitted application, each uniform on 1..execution_max. The next gate opens
    at the release, or at the next arrival if that comes later, rounded up to a window
    boundary under window-based control; its batch holds every arrival since the last gate.
    """
    for value, name in ((planning_max, "planning_max"), (execution_max, "execution_max")):
        _check_positive(value, name)
    _check_positive(horizon, "horizon")
    _check_positive(spacing, "spacing")
    _check_window(window, spacing)
    count = _arrival_count(horizon, spacing)
    last_arrival = (count - 1) * spacing
    p, e = planning_max, execution_max
    longest_hold = p + e
    if window:
        longest_hold = ((longest_hold + window - 1) // window) * window
    largest_batch = min(count, (longest_hold + spacing - 1) // spacing + 1)
    tick = window or 1
    rows = (last_arrival + longest_hold + spacing + window) // tick + 2
    # makespan[n, m]: probability that the longest of 4n Execution durations equals m
    sizes = np.arange(largest_batch + 1)
    cdf = (np.arange(e + 1)[None, :] / e) ** (4 * sizes[:, None])
    makespan = np.diff(cdf, axis=1, prepend=0.0)
    makespan[0] = 0.0
    holds = np.arange(2, p + e + 1)  # Planning plus makespan
    upper = np.minimum(e, holds - 1)  # makespan range giving that hold: holds - p .. holds - 1
    lower = np.maximum(1, holds - p)
    mass = np.zeros((rows, largest_batch + 1))
    mass[0, 1] = 1.0  # the first arrival is admitted alone at time 0
    total_waiting = admitted = terminal = 0.0
    for row in range(rows):
        present = mass[row]
        if not present.any():
            continue
        gate = row * tick
        served = min(count - 1, gate // spacing)  # index of the last arrival at or before gate
        # waiting of the n arrivals served - n + 1 .. served, admitted at this gate
        waiting = sizes * gate - spacing * sizes * (2 * served - sizes + 1) / 2
        total_waiting += float(present @ waiting)
        admitted += float(present @ sizes)
        if served == count - 1:
            terminal += float(present.sum())
            continue
        mixed = present @ makespan
        prefix = np.cumsum(mixed)
        hold_mass = (prefix[upper] - prefix[lower - 1]) / p
        target = np.maximum(gate + holds, (served + 1) * spacing)
        if window:
            target = ((target + window - 1) // window) * window
        next_size = np.minimum(count - 1, target // spacing) - served
        np.add.at(mass, (target // tick, next_size), hold_mass)
    if abs(terminal - 1.0) > 1e-8 or abs(admitted - count) > 1e-6:
        raise ArithmeticError("the batch recurrence lost probability mass")
    return total_waiting / count


# Two or three planners, event-driven.


def residual_waiting(planning_max, planners, horizon, spacing=2):
    """Mean waiting time of event-driven MACPE-K with two or three planners.

    Backward recurrence over the remaining horizon h, the number of slots until the last
    arrival. The state at a gate is h and the residual busy times of the other planners
    (a for two planners, a <= b for three); the batch just admitted keeps its planner for a
    uniform duration P on 1..planning_max. The next gate opens when the first planner is
    released, min(P, a) slots later, or at the next arrival if that comes later; every request
    arriving in between waits until then. Averaging over P uses running sums of the value
    function: along the diagonal (h - P, a - P, ...) when the new planner is released first,
    and over the first residual when another planner is released first.
    """
    _check_positive(planning_max, "planning_max")
    _check_positive(horizon, "horizon")
    _check_positive(spacing, "spacing")
    if planners not in (2, 3):
        raise ValueError("this recurrence covers two or three planners")
    count = _arrival_count(horizon, spacing)
    last_arrival = (count - 1) * spacing
    if last_arrival == 0:
        return 0.0
    solve = _two_planners if planners == 2 else _three_planners
    return solve(planning_max, last_arrival, spacing) / count


def _two_planners(D, H, spacing):
    """Total expected waiting with two planners; see residual_waiting."""
    value = np.zeros((H + 1, D))  # value[h, a]: expected waiting still to come
    diagonal = np.zeros((H + 1, D))  # value summed along (h - i, a - i) for i >= 0
    prefix = np.zeros((H + 1, D))  # prefix[h, x]: value[h, 1] + ... + value[h, x], x < D
    a = np.arange(D)
    for h in range(1, H + 1):
        gap = (h - 1) % spacing + 1  # slots until the next arrival
        t = h - gap  # horizon left at that arrival
        waiting = _batch_waiting(h, gap, a, spacing)
        # waiting summed over P = 1..D, the gate opening min(P, a) slots from now
        total = (np.cumsum(waiting) + (D - a) * waiting).astype(float)
        if t > 0:
            free = a < gap
            # A planner is released before the next arrival, which opens the next gate
            # without waiting; the other planner then has max(P - gap, 0) slots left.
            total[free] += min(D, gap) * value[t, 0] + prefix[t, max(D - gap, 0)]
            # Otherwise P <= gap also opens the gate at the arrival, with a - gap slots left,
            # and gap < P <= a opens it at the release of the new planner, with a - P left.
            j = a[~free] - gap
            total[~free] += diagonal[t, j] + (gap - 1) * value[t, j]
        # P > a: the other planner is released first, a slots from now; the new one keeps P - a.
        late = (a >= gap) & (a < h)
        idx = a[late]
        total[late] += prefix[h - idx, D - idx]
        value[h] = total / D
        diagonal[h, 0] = value[h, 0]
        diagonal[h, 1:] = value[h, 1:] + diagonal[h - 1, :-1]
        prefix[h, 1:] = np.cumsum(value[h, 1:])
    return value[H, 0]


def _three_planners(D, H, spacing):
    """Total expected waiting with three planners; see residual_waiting.

    States (a, b) with a <= b form one flat triangle. The value function and its diagonal
    sums are kept for the last few horizons; the prefix sums over the first residual are
    needed up to D horizons back, about 1.3 GB for planning_max = 900.
    """
    a = np.repeat(np.arange(D), D - np.arange(D))
    b = np.concatenate([np.arange(start, D) for start in range(D)])
    offset = np.concatenate(([0], np.cumsum(D - np.arange(D))[:-1]))

    def pack(first, second):
        return offset[first] + (second - first)

    keep = spacing + 1
    depth = min(D, H + 1)
    value = np.zeros((keep, a.size))
    diagonal = np.zeros((keep, a.size))
    prefix = np.zeros(
        (depth, a.size)
    )  # prefix[., pack(z, m)]: sum over x = 1..m of value(sorted(x, z))
    square = np.zeros((D, D))
    first = np.arange(D)
    older = pack(
        np.maximum(a - 1, 0), np.maximum(b - 1, 0)
    )  # state (a - 1, b - 1), used when a >= 1
    late_index = pack(b - a, np.minimum(D - a, D - 1))  # state (b - a, D - a), used when a >= 1
    cases = {}
    for gap in range(1, spacing + 1):
        z = np.maximum(b - gap, 0)
        cases[gap] = dict(
            free=a < gap,
            free_value=pack(np.zeros_like(z), z),
            free_prefix=pack(z, np.full_like(z, max(D - gap, 0))),
            busy=pack(np.maximum(a - gap, 0), np.maximum(b - gap, 0)),
        )
    for h in range(1, H + 1):
        gap = (h - 1) % spacing + 1
        t = h - gap
        case = cases[gap]
        waiting = _batch_waiting(h, gap, first, spacing)
        total = (np.cumsum(waiting) + (D - first) * waiting).astype(float)[a]
        if t > 0:
            earlier, diag, pre = value[t % keep], diagonal[t % keep], prefix[t % depth]
            free = case["free"]
            total[free] += min(D, gap) * earlier[case["free_value"][free]]
            total[free] += pre[case["free_prefix"][free]]
            busy = ~free
            j = case["busy"][busy]
            total[busy] += diag[j] + (gap - 1) * earlier[j]
        late = (a >= gap) & (a < h)
        total[late] += prefix[(h - a[late]) % depth, late_index[late]]
        total /= D
        value[h % keep] = total
        running = total.copy()
        inner = a >= 1
        running[inner] += diagonal[(h - 1) % keep][older[inner]]
        diagonal[h % keep] = running
        square[a, b] = total
        square[b, a] = total
        square[:, 0] = 0.0
        prefix[h % depth] = np.cumsum(square, axis=1)[a, b]
    return value[H % keep][0]


# Several planners, window-based.


def window_waiting(planning_max, planners, horizon, spacing=2, window=100):
    """Mean waiting time of window-based MACPE-K with several planners, computed exactly.

    Gates open at window boundaries, so residual busy times are whole numbers of windows and
    the state space stays small. The state at a gate is the residual, in windows, of the
    other planners; the batch just admitted keeps its planner for ceil(P / window) windows.
    The next gate is the first boundary at which a planner is free, and every arrival in
    between waits until it. Probabilities are exact fractions.
    """
    for value, name in ((planning_max, "planning_max"), (planners, "planners")):
        _check_positive(value, name)
    _check_positive(horizon, "horizon")
    _check_positive(spacing, "spacing")
    _check_positive(window, "window")
    if window < spacing or window % spacing:
        raise ValueError("window must be a multiple of the arrival spacing")
    count = _arrival_count(horizon, spacing)
    last_arrival = (count - 1) * spacing
    # number of durations occupying q windows: (q - 1) * window + 1 .. min(q * window, D)
    occupancy = {
        q: min(window, planning_max - (q - 1) * window)
        for q in range(1, (planning_max + window - 1) // window + 1)
    }
    pending = defaultdict(lambda: defaultdict(Fraction))
    pending[0][(0,) * (planners - 1)] = Fraction(1)
    total_waiting, terminal = Fraction(), Fraction()
    for gate in range((last_arrival + window - 1) // window + max(occupancy) + 1):
        for residuals, probability in pending.pop(gate, {}).items():
            if gate * window >= last_arrival:
                terminal += probability
                continue
            for q, weight in occupancy.items():
                ordered = sorted((*residuals, q))
                elapsed = max(1, ordered[0])  # windows until a planner is free, at least one
                next_gate = gate + elapsed
                remaining = tuple(max(0, r - elapsed) for r in ordered[1:])
                first = gate * window // spacing + 1  # first arrival after this gate
                last = min(last_arrival, next_gate * window) // spacing  # last one admitted next
                admitted = max(0, last - first + 1)
                waiting = admitted * next_gate * window - spacing * (first + last) * admitted // 2
                mass = probability * Fraction(weight, planning_max)
                total_waiting += mass * waiting
                pending[next_gate][remaining] += mass
    if pending or terminal != 1:
        raise ArithmeticError("the window recurrence lost probability mass")
    return float(total_waiting / count)


def sufficient_waiting(requests=200, spacing=2, window=0):
    """Mean waiting time when a planner is always available: the next boundary only."""
    waits = [(window - (i * spacing) % window) % window if window else 0 for i in range(requests)]
    return sum(waits) / requests


# References for the variable-duration runs.


def reference(delta_p, controller, window=0, delta_e=100, requests=200, spacing=2):
    """Expected mean waiting and deployment times of one controller."""
    if controller not in CONTROLLERS:
        raise ValueError(f"unknown controller: {controller}")
    horizon = requests * spacing
    if controller == "sequential":
        planners = 1
        waiting = sequential_waiting(delta_p, delta_e, horizon, spacing, window)
        method = "forward recurrence over gate times and batch sizes"
    elif controller == "sufficient":
        planners = math.ceil(delta_p / (window or spacing))
        waiting = sufficient_waiting(requests, spacing, window)
        method = "next window boundary only"
    else:
        planners = int(controller[-1])
        if planners == 1:
            waiting = single_planner_waiting(range(1, delta_p + 1), horizon, spacing, window)
            method = "forward recurrence over gate times"
        elif window:
            waiting = window_waiting(delta_p, planners, horizon, spacing, window)
            method = "exact forward recurrence over window-aligned residuals"
        else:
            waiting = residual_waiting(delta_p, planners, horizon, spacing)
            method = "backward recurrence over the horizon and residual busy times"
    return dict(
        controller=controller,
        k_P=planners,
        window=window,
        delta_P=delta_p,
        delta_E=delta_e,
        requests=requests,
        spacing=spacing,
        mean_TW=waiting,
        mean_TD=waiting + (delta_p + 1) / 2 + expected_application_execution(delta_e),
        method=method,
    )


# Exhaustive checks on small cases.


def self_check():
    """Compare every recurrence with an exhaustive enumeration of small cases."""
    p = pathwise([0, 1, 2, 3, 4], [5] * 5, [[2] * 4] * 5, 1)
    assert p["waits"] == [0, 4, 3, 2, 1]
    p = pathwise([0, 1, 2, 3, 4], [5] * 5, [[2] * 4] * 5, 3)
    assert p["waits"] == [0, 0, 0, 2, 1]
    p = pathwise([0, 1, 2, 3, 4], [5] * 5, [[2] * 4] * 5, 1, 4)
    assert p["waits"] == [0, 7, 6, 5, 4]
    # Sequential baseline: two requests, Planning on {1, 2}, eight action durations on {1, 2}.
    waits, deployments = [], []
    for planning in product((1, 2), repeat=2):
        for execution in product((1, 2), repeat=8):
            row = pathwise([0, 2], planning, [execution[:4], execution[4:]], sequential=True)
            waits.append(row["mean_TW"])
            deployments.append(row["mean_TD"])
    expected = reference(2, "sequential", 0, 2, 2)
    assert abs(expected["mean_TW"] - statistics.mean(waits)) < 1e-12
    assert abs(expected["mean_TD"] - statistics.mean(deployments)) < 1e-12
    sequential = dict(paths=len(waits), mean_TW=statistics.mean(waits))
    # MACPE-K: four requests, every combination of Planning durations.
    macpe = []
    for d, k, w in ((7, 1, 0), (7, 2, 0), (9, 3, 0), (7, 1, 2), (7, 2, 2), (9, 3, 2)):
        arrivals = [0, 2, 4, 6]
        values = [
            pathwise(arrivals, planning, [[1] * 4] * 4, k, w)["mean_TW"]
            for planning in product(range(1, d + 1), repeat=4)
        ]
        mean = statistics.mean(values)
        assert abs(reference(d, f"macpe{k}", w, 2, 4)["mean_TW"] - mean) < 1e-12
        macpe.append(dict(delta_P=d, k_P=k, window=w, paths=len(values), mean_TW=mean))
    # Sufficient capacity: no planner contention on the full workload.
    sufficient = []
    for d, w in ((7, 0), (7, 4), (50, 0), (900, 100)):
        k = math.ceil(d / (w or 2))
        row = pathwise(list(range(0, 400, 2)), [d] * 200, [[1] * 4] * 200, k, w)
        assert row["mean_TW"] == sufficient_waiting(200, 2, w)
        sufficient.append(dict(delta_P=d, window=w, k_P=k, mean_TW=row["mean_TW"]))
    return dict(passed=True, sequential=sequential, macpe=macpe, sufficient=sufficient)


# Reference table.


def build(output=OUTPUT):
    """Write the reference table: variable-duration expectations and fixed-duration replays.

    Expectations are rounded to two decimals.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []
    for window in (0, 100):
        for delta_p in VARIABLE_GRID:
            for controller in CONTROLLERS:
                row = reference(delta_p, controller, window)
                row.update(mean_TW=round(row["mean_TW"], 2), mean_TD=round(row["mean_TD"], 2))
                rows.append(row)
            print(f"window {window}, delta_P {delta_p}: done", flush=True)
    arrivals = list(range(0, 400, 2))
    fixed, surfaces = [], []
    for window in (0, 100):
        for t_p in FIXED_GRID:
            for controller in ("sequential", "macpe1", "sufficient"):
                k = math.ceil(t_p / (window or 2)) if controller == "sufficient" else 1
                r = pathwise(
                    arrivals, [t_p] * 200, [[t_p] * 4] * 200, k, window, controller == "sequential"
                )
                fixed.append(
                    dict(
                        controller=controller,
                        window=window,
                        k_P=k,
                        T_P=t_p,
                        T_E=t_p,
                        T_G=2 * t_p,
                        mean_TW=round(r["mean_TW"], 2),
                        mean_TD=round(r["mean_TD"], 2),
                    )
                )
        for t_p in SURFACE_GRID:
            for t_e in SURFACE_GRID:
                for controller in ("sequential", "macpe1", "sufficient"):
                    k = math.ceil(t_p / (window or 2)) if controller == "sufficient" else 1
                    r = pathwise(
                        arrivals,
                        [t_p] * 200,
                        [[t_e] * 4] * 200,
                        k,
                        window,
                        controller == "sequential",
                    )
                    surfaces.append(
                        dict(
                            controller=controller,
                            window=window,
                            k_P=k,
                            T_P=t_p,
                            T_E=t_e,
                            mean_TW=round(r["mean_TW"], 2),
                            mean_TD=round(r["mean_TD"], 2),
                        )
                    )
    document = dict(
        references=rows,
        fixed_periodic_references=fixed,
        surface_references=surfaces,
        protocol=dict(
            requests=200,
            arrivals="0,2,...,398",
            spacing=2,
            windows=[0, 100],
            initial_state="empty",
            variable_delta_P=list(VARIABLE_GRID),
            delta_E=100,
            fixed_T_P=list(FIXED_GRID),
            fixed_T_E_equals_T_P=True,
            surface_T_P=list(SURFACE_GRID),
            surface_T_E=list(SURFACE_GRID),
            application_execution="maximum of four independent uniform component durations",
            sequential_release="maximum over all component actions of the admitted batch",
            sufficient_planners=(
                "ceil(delta_P/spacing) for events; ceil(delta_P/window) for windows"
            ),
            note=(
                "finite workload of 200 arrivals from an empty state, fully drained;"
                " not stationary averages"
            ),
            created_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(time.perf_counter() - started, 2),
        ),
    )
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    print(
        f"Saved {len(rows)} variable, {len(fixed)} fixed, {len(surfaces)} surface references"
        f" to {output}",
        flush=True,
    )
    return document


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--self-check", action="store_true", help="run the exhaustive small-case checks"
    )
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
    else:
        build(args.output)
