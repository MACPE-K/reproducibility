"""Discrete-time simulation of the admission contract and of the placement workload.

The simulator classes build on each other:

* ContractSimulator: knowledge base, atomic admission, execution waves and recovery;
* ScopedGenerationContract: read-only scopes and revision generations;
* DagContractSimulator: DAG applications, latency bounds, guards on retained components;
* NetworkDagContractSimulator: link capacities, anti-affinity, a bounded executor pool;
* BatchedNetworkDagSimulator: all pending requests planned as one batch (the controllers);
* ExactTimingSimulator and BatchAblationSimulator: the cached variant used by campaign A
  and the variant with one mechanism disabled used by campaign C.

`make_workload` builds the infrastructure, the applications and their requests from one
seed; `failure_model` draws action failures.
"""

from __future__ import annotations

from collections import Counter, deque
from copy import copy, deepcopy
from dataclasses import dataclass, field
from itertools import combinations
import math
import random
import statistics
from typing import Callable, Dict, List, Optional, Set, Tuple

# Core state, admission and recovery

UND, DPL, DEP, MIG, DEL = "UND", "DPL", "DEP", "MIG", "DEL"


TRANSIENT = {"deploy": DPL, "migrate": MIG, "delete": DEL}


STABLE_BEFORE = {DPL: UND, MIG: DEP, DEL: DEP}


PENDING, ADMITTED, DELAYED = "pending", "admitted", "delayed"


REJECTED_STALE, REJECTED_INFEASIBLE, CANCELLED = (
    "rejected_stale",
    "rejected_infeasible",
    "cancelled",
)


COUNTERS = (
    "requests",
    "candidates",
    "admitted",
    "admitted_actions",
    "empty_candidates",
    "empty_waiting",
    "empty_failed_dependency",
    "serialized",
    "abandoned_after_stale",
    "abandoned_after_failure",
    "abandoned_after_precondition",
    "split_waves",
    "delayed",
    "rejected_stale",
    "rejected_infeasible",
    "recomputed",
    "recomputed_after_stale",
    "recomputed_after_failure",
    "recomputed_after_precondition",
    "abandoned",
    "failures",
    "compensations",
    "precondition_failures",
    "completed_actions",
    "completed_requests",
    "lost_requests",
    "capacity_violations",
    "violation_time",
    "max_overcommit",
    "stale_admissions",
    "target_overwritten",
    "duplicate_placements",
    "invalid_transitions",
    "leaked_reservations",
    "leaked_amount",
    "predicate_evaluations",
    "comparisons_fresh",
    "comparisons_pairwise",
    "comparisons_aggregate",
)


def stream(*key):
    """Random generator for one purpose of one run, keyed by the run seed and identifiers.

    Keyed generators give a request the same planner tie-breaks and the same failures under
    every controller, whatever the order of events.
    """
    return random.Random("/".join(map(str, key)))


@dataclass
class Request:
    """One adaptation request: a scope and a planner function."""

    id: int
    arrival: int
    scope: Tuple[str, ...]
    planning: int
    plan: Callable
    attempt: int = 0
    recomputations: int = 0
    started: Optional[int] = None
    finished: Optional[int] = None
    lost: bool = False


@dataclass
class Snapshot:
    """Consistent read of K at a planner start."""

    time: int
    view: Dict[str, Optional[str]]
    lifecycle: Dict[str, str]
    target: Dict[str, Optional[str]]
    free: Dict[str, int]
    versions: Dict[tuple, int]


@dataclass
class Action:
    id: int
    kind: str
    component: str
    source: Optional[str]
    destination: Optional[str]
    amount: int
    candidate: int
    wave: int
    reads: frozenset
    writes: frozenset
    predecessors: Set[int] = field(default_factory=set)
    start: Optional[int] = None
    finish: Optional[int] = None
    fail_at: Optional[int] = None
    done: bool = False

    @property
    def pre_state(self):
        """(placement, lifecycle) the action expects when it starts."""
        return (self.source, UND if self.kind == "deploy" else DEP)

    @property
    def post_state(self):
        """(placement, lifecycle) the action leaves on completion."""
        return (self.destination, UND if self.kind == "delete" else DEP)


@dataclass
class Candidate:
    id: int
    request: Request
    planner: int
    attempt: int
    emitted: int
    snapshot: Snapshot
    mapping: Dict[str, Optional[str]]
    redeploy: Set[str]
    waves: List[List[Action]] = field(default_factory=list)
    status: str = PENDING
    first_check: Optional[int] = None
    admitted_at: Optional[int] = None
    retries: int = 0
    serialized: bool = False
    stale: bool = False
    counted_delayed: bool = False
    waits_on: Set[int] = field(
        default_factory=set
    )  # in-flight actions an empty candidate relies on

    @property
    def actions(self):
        return [action for wave in self.waves for action in wave]


def chi(a, b):
    """Actions conflict when their read/write footprints overlap."""
    return bool((a.writes & b.writes) or (a.writes & b.reads) or (a.reads & b.writes))


class ContractSimulator:
    """Discrete-time simulation of the admission contract on a small K."""

    def __init__(
        self,
        hosts,
        components,
        requests,
        planners=3,
        check_fresh=True,
        check_aggregate=True,
        compensate=True,
        failure=None,
        max_retries=3,
        max_recomputations=5,
        seed=0,
        log=False,
        policy="macpe",
    ):
        """Build the knowledge base and the planners.

        policy 'macpe' releases a planner after admission; 'sequential' holds it until the
        batch's actions complete.
        """
        if policy not in ("macpe", "sequential"):
            raise ValueError("policy must be 'macpe' or 'sequential'")
        if planners < 1 or max_retries < 0 or max_recomputations < 0:
            raise ValueError("planners must be >= 1 and bounds >= 0")
        self.cap = dict(hosts)
        self.hosts = sorted(self.cap)
        self.dem = {c: spec["demand"] for c, spec in components.items()}
        self.duration = {c: spec["duration"] for c, spec in components.items()}
        for c, spec in components.items():
            if spec["demand"] < 0 or spec["duration"] < 1:
                raise ValueError("demand must be >= 0 and duration >= 1")
            if spec.get("host") is not None and spec["host"] not in self.cap:
                raise ValueError("unknown initial host for %s" % c)
        self.m_act = {c: spec.get("host") for c, spec in components.items()}
        self.m_tgt = dict(self.m_act)
        self.lc = {c: DEP if host is not None else UND for c, host in self.m_act.items()}
        self.versions = {("comp", c): 0 for c in components}
        self.versions.update({("host", h): 0 for h in self.cap})
        self.ledger = {}  # action id -> [component, host, amount, leaked]
        self.active = {}  # O_active: action id -> Action
        self.inflight = {c: [] for c in components}  # admitted unfinished action ids per component
        self.requests = sorted(requests, key=lambda r: (r.arrival, r.id))
        for request in self.requests:
            if request.planning < 1 or not request.scope:
                raise ValueError("planning duration must be >= 1 and scope nonempty")
        self.queue = deque()  # pending requests (FIFO; recomputations re-enter at the front)
        self.planners = [
            dict(request=None, candidate=None, emit=None, held=None) for _ in range(planners)
        ]
        self.check_fresh, self.check_aggregate, self.compensate = (
            check_fresh,
            check_aggregate,
            compensate,
        )
        self.failure, self.policy = failure, policy
        self.max_retries, self.max_recomputations, self.seed = max_retries, max_recomputations, seed
        self.counts = {name: 0 for name in COUNTERS}
        self.counts["requests"] = len(self.requests)
        self.candidates, self.delayed, self.waiting_waves = [], [], []
        self.pending_noops = (
            []
        )  # admitted candidates without action, waiting for the actions their view relies on
        self.latencies, self.planner_waits, self.responses = [], [], []
        self.in_violation = {}
        self.events = [] if log else None
        self.next_action_id = 0
        self.end_time = 0
        self._arrival_cursor = 0

    # State helpers
    def usage(self, host):
        return sum(self.dem[c] for c, h in self.m_act.items() if h == host)

    def reserved(self, host):
        return sum(entry[2] for entry in self.ledger.values() if entry[1] == host)

    def loads(self):
        return {h: self.usage(h) + self.reserved(h) for h in self.hosts}

    def view(self, component):
        """Conservative placement: M_res if an admitted action is in flight, else M_act."""
        if self.inflight[component]:
            return self.active[self.inflight[component][-1]].post_state[0]
        return self.m_act[component]

    def touch(self, *elements):
        for element in elements:
            if element[1] is not None:
                self.versions[element] += 1

    def log(self, time, message):
        if self.events is not None:
            self.events.append((time, message))

    def snapshot(self, time):
        view = {c: self.view(c) for c in self.dem}
        free = {h: self.cap[h] - load for h, load in self.loads().items()}
        return Snapshot(time, view, dict(self.lc), dict(self.m_tgt), free, dict(self.versions))

    def pre_state_of(self, action, candidate):
        """State action will see at its start: after its predecessors complete."""
        c = action.component
        if action.wave > 0:
            earlier = [a for a in candidate.waves[action.wave - 1] if a.component == c]
            if earlier:
                return earlier[-1].post_state
        if self.inflight[c]:
            return self.active[self.inflight[c][-1]].post_state
        lifecycle = self.lc[c]
        return (self.m_act[c], lifecycle)

    # Event loop
    def run(self):
        time = 0
        while True:
            if self._execution_events(time):
                self._recheck_delayed(time)
            self._emissions(time)
            self._start_planners(time)
            self._observe(time)
            following = self._next_time(time)
            if following is None:
                break
            time = following
        self.end_time = time
        for cand in list(
            self.pending_noops
        ):  # dependencies never settled: not expected at quiescence
            self.pending_noops.remove(cand)
            cand.status = CANCELLED
            self._release_candidate(cand)
            self._lose(cand.request)
        for host in list(self.in_violation):
            self.counts["violation_time"] += time - self.in_violation.pop(host)
        return self.metrics()

    def _next_time(self, time):
        times = [
            a.fail_at if a.fail_at is not None else a.finish
            for a in self.active.values()
            if a.start is not None
        ]
        times += [p["emit"] for p in self.planners if p["emit"] is not None]
        if self._arrival_cursor < len(self.requests):
            times.append(self.requests[self._arrival_cursor].arrival)
        times = [t for t in times if t is not None and t > time]
        return min(times) if times else None

    def _observe(self, time):
        for host, load in self.loads().items():
            over = load > self.cap[host]
            if over and host not in self.in_violation:
                self.in_violation[host] = time
                self.counts["capacity_violations"] += 1
                self.log(time, "violation on %s: load %d > cap %d" % (host, load, self.cap[host]))
            elif not over and host in self.in_violation:
                self.counts["violation_time"] += time - self.in_violation.pop(host)
            self.counts["max_overcommit"] = max(
                self.counts["max_overcommit"], load - self.cap[host]
            )

    # Planning
    def _start_planners(self, time):
        while (
            self._arrival_cursor < len(self.requests)
            and self.requests[self._arrival_cursor].arrival <= time
        ):
            self.queue.append(self.requests[self._arrival_cursor])
            self._arrival_cursor += 1
        for index, planner in enumerate(self.planners):
            if planner["request"] is not None or not self.queue:
                continue
            request = self.queue.popleft()
            if request.started is None:
                request.started = time
                self.planner_waits.append(time - request.arrival)
            planner.update(
                request=request, emit=time + request.planning, snapshot=self.snapshot(time)
            )
            self.log(
                time,
                "planner %d starts request %d attempt %d" % (index, request.id, request.attempt),
            )

    def _emissions(self, time):
        for index, planner in enumerate(self.planners):
            if planner["emit"] != time:
                continue
            request = planner["request"]
            rng = stream(self.seed, "plan", request.id, request.attempt)
            mapping, redeploy = request.plan(
                planner["snapshot"], request.scope, request.attempt, rng
            )
            candidate = Candidate(
                len(self.candidates),
                request,
                index,
                request.attempt,
                time,
                planner["snapshot"],
                dict(mapping),
                set(redeploy),
            )
            self.candidates.append(candidate)
            self.counts["candidates"] += 1
            planner.update(emit=None, candidate=candidate)
            self._admit(candidate, time)

    def _release(self, planner_index):
        self.planners[planner_index].update(
            request=None, candidate=None, emit=None, held=None, snapshot=None
        )

    def _release_candidate(self, cand):
        """Release only a slot still owned by this candidate, never a reused slot."""
        if self.planners[cand.planner]["candidate"] is cand:
            self._release(cand.planner)

    def _hold_planner(self, cand):
        """The sequential baseline keeps its planner until the batch's actions complete."""
        return self.policy == "sequential"

    def _recompute(self, request, time, cause):
        """Flag a request for recomputation (stale, failure or precondition)."""
        request.attempt += 1
        if request.recomputations >= self.max_recomputations:
            self.counts["abandoned"] += 1
            self.counts["abandoned_after_" + cause] += 1
            self._lose(request)
            return
        request.recomputations += 1
        self.counts["recomputed"] += 1
        self.counts["recomputed_after_" + cause] += 1
        self.queue.appendleft(request)
        self.log(time, "request %d recomputed after %s" % (request.id, cause))

    def _lose(self, request):
        if not request.lost:
            request.lost = True
            self.counts["lost_requests"] += 1

    def _finish_request(self, cand, time):
        """Effective completion of a request: the placement of its scope is actual."""
        cand.request.finished = time
        self.responses.append(time - cand.request.arrival)
        self.counts["completed_requests"] += 1

    def _settle_dependents(self, action, time, failed):
        """Settle the empty candidates whose requested placement relied on this action."""
        for cand in [c for c in self.pending_noops if action.id in c.waits_on]:
            cand.waits_on.discard(action.id)
            if failed:
                self.pending_noops.remove(cand)
                cand.status = CANCELLED
                self._release_candidate(cand)
                self.counts["empty_failed_dependency"] += 1
                self.log(
                    time,
                    "candidate %d (request %d) cancelled: %s(%s) did not complete"
                    % (cand.id, cand.request.id, action.kind, action.component),
                )
                if self.compensate:
                    self._recompute(cand.request, time, "failure")
                else:
                    self._lose(cand.request)
            elif not cand.waits_on:
                self.pending_noops.remove(cand)
                if all(self.m_act[c] == cand.mapping[c] for c in cand.request.scope):
                    self._finish_request(cand, time)
                else:
                    cand.status = CANCELLED
                    self.counts["precondition_failures"] += 1
                    self.log(
                        time,
                        "candidate %d (request %d): placement not effective"
                        % (cand.id, cand.request.id),
                    )
                    if self.compensate:
                        self._recompute(cand.request, time, "precondition")
                    else:
                        self._lose(cand.request)
                if cand not in self.pending_noops:
                    self._release_candidate(cand)

    # Atomic admission
    def _admit(self, cand, time):
        """Evaluate Admissible(O_j, K_t) and commit on success."""
        counts = self.counts
        counts["predicate_evaluations"] += 1
        if cand.first_check is None:
            cand.first_check = time
        scope = cand.request.scope
        # (1) Fresh_j: every version read by the planner is unchanged.
        read_set = [("comp", c) for c in scope]
        counts["comparisons_fresh"] += len(read_set)
        changed = [e for e in read_set if self.versions[e] != cand.snapshot.versions[e]]
        cand.stale = bool(changed)
        if changed and self.check_fresh:
            if cand.status == DELAYED:
                self.delayed.remove(cand)
            cand.status = REJECTED_STALE
            counts["rejected_stale"] += 1
            self.log(
                time, "candidate %d (request %d) stale on %s" % (cand.id, cand.request.id, changed)
            )
            self._release_candidate(cand)
            self._recompute(cand.request, time, "stale")
            return REJECTED_STALE
        # (2) Derive actions and split intra-candidate conflicts into waves.
        if not cand.waves:
            self._derive_waves(cand)
        actions = cand.actions
        # (3) Pairwise conflict check chi against O_active.
        for action in actions:
            action.predecessors = (
                {a.id for a in cand.waves[action.wave - 1] if a.component == action.component}
                if action.wave > 0
                else set()
            )
            for other in self.active.values():
                counts["comparisons_pairwise"] += 1
                if chi(action, other):
                    action.predecessors.add(other.id)
        cand.serialized = any(action.predecessors & set(self.active) for action in actions)
        # (4) Aggregate invariant on K extended with the candidate reservations.
        if self.check_aggregate and not self._invariant(cand, actions):
            return self._delay(cand, time)
        # (5) Commit: target (scope only), reservations, registration, versions.
        self._audit_stale(cand, changed, actions)
        written = set()
        for c in scope:
            if self.m_tgt[c] != cand.mapping[c]:
                self.m_tgt[c] = cand.mapping[c]
                written.add(("comp", c))
        for action in actions:
            if action.amount:
                self.ledger[action.id] = [
                    action.component,
                    action.destination,
                    action.amount,
                    False,
                ]
                written.add(("host", action.destination))
            self.active[action.id] = action
            self.inflight[action.component].append(action.id)
            written.add(("comp", action.component))
        self.touch(*sorted(written))  # one admission write = one increment per element
        if cand.status == DELAYED:
            self.delayed.remove(cand)
        cand.status, cand.admitted_at = ADMITTED, time
        counts["admitted"] += 1
        counts["admitted_actions"] += len(actions)
        counts["serialized"] += cand.serialized
        counts["split_waves"] += len(cand.waves) > 1
        counts["empty_candidates"] += not actions
        self.latencies.append(time - cand.emitted)
        self.log(
            time,
            "candidate %d (request %d) admitted%s: %s"
            % (
                cand.id,
                cand.request.id,
                " serialized" if cand.serialized else "",
                ", ".join("%s(%s)" % (a.kind, a.component) for a in actions) or "no action",
            ),
        )
        if not actions:
            # No new action: the requested placement equals the conservative view. It is
            # effective only once the in-flight actions that view relies on complete.
            cand.waits_on = {self.inflight[c][-1] for c in scope if self.inflight[c]}
            if cand.waits_on:
                self.pending_noops.append(cand)
                counts["empty_waiting"] += 1
                self.log(
                    time,
                    "candidate %d (request %d) waits for action(s) %s"
                    % (cand.id, cand.request.id, ", ".join(str(i) for i in sorted(cand.waits_on))),
                )
            else:
                self._finish_request(cand, time)
            if cand in self.pending_noops and self.policy == "sequential":
                self.planners[cand.planner]["held"] = cand
            else:
                self._release_candidate(cand)
        elif self._hold_planner(cand):
            self.planners[cand.planner]["held"] = cand  # policy-dependent hold, see _hold_planner
        else:
            self._release_candidate(cand)
        for wave_index in range(len(cand.waves)):
            self.waiting_waves.append((cand, wave_index))
        self._start_ready_waves(time)
        return ADMITTED

    def _derive_waves(self, cand):
        """Actions of the merged target against the planner's conservative view."""
        first, second = [], []
        for c in cand.request.scope:
            before, after = cand.snapshot.view[c], cand.mapping[c]
            if before == after:
                continue
            footprint = frozenset({("comp", c)})
            if before is None:
                first.append(self._action("deploy", c, None, after, cand, 0, footprint))
            elif after is None:
                first.append(self._action("delete", c, before, None, cand, 0, footprint))
            elif c in cand.redeploy:
                first.append(self._action("delete", c, before, None, cand, 0, footprint))
                second.append(self._action("deploy", c, None, after, cand, 1, footprint))
            else:
                first.append(self._action("migrate", c, before, after, cand, 0, footprint))
        cand.waves = [wave for wave in (first, second) if wave]

    def _action(self, kind, component, source, destination, cand, wave, footprint):
        amount = self.dem[component] if destination is not None else 0
        action = Action(
            self.next_action_id,
            kind,
            component,
            source,
            destination,
            amount,
            cand.id,
            wave,
            footprint,
            footprint,
        )
        self.next_action_id += 1
        return action

    def _invariant(self, cand, actions):
        """Inv(K^{+j}): host capacities with new reservations, and the pre-state of each action."""
        counts = self.counts
        extra = {}
        for action in actions:
            if action.amount:
                extra[action.destination] = extra.get(action.destination, 0) + action.amount
        loads = self.loads()
        feasible = True
        for host, amount in extra.items():
            counts["comparisons_aggregate"] += 1
            if loads[host] + amount > self.cap[host]:
                feasible = False
        for action in actions:
            counts["comparisons_aggregate"] += 1
            if self.pre_state_of(action, cand) != action.pre_state:
                feasible = False
        return feasible

    def _delay(self, cand, time):
        counts = self.counts
        if cand.status != DELAYED:
            cand.status = DELAYED
            self.delayed.append(cand)
            counts["delayed"] += 1
            self.log(time, "candidate %d (request %d) delayed" % (cand.id, cand.request.id))
        else:
            cand.retries += 1
        if not self.active or cand.retries >= self.max_retries:
            self.delayed.remove(cand)
            cand.status = REJECTED_INFEASIBLE
            counts["rejected_infeasible"] += 1
            self.log(
                time,
                "candidate %d (request %d) rejected after %d re-checks"
                % (cand.id, cand.request.id, cand.retries),
            )
            self._release_candidate(cand)
            self._lose(cand.request)
            return REJECTED_INFEASIBLE
        if self.policy == "macpe":
            self._release_candidate(cand)
        return DELAYED

    def _recheck_delayed(self, time):
        for cand in sorted(self.delayed, key=lambda x: (x.first_check, x.id)):
            if cand.status == DELAYED:
                self._admit(cand, time)

    def _audit_stale(self, cand, changed, actions):
        """Audit of a stale candidate admitted without the Fresh check."""
        if not changed:
            return
        counts = self.counts
        counts["stale_admissions"] += 1
        for element in changed:
            c = element[1]
            if (
                element[0] == "comp"
                and cand.snapshot.view[c] != self.view(c)
                and cand.mapping[c] != self.m_tgt[c]
            ):
                counts["target_overwritten"] += 1  # written from an outdated placement
        for action in actions:
            placement, lifecycle = self.pre_state_of(action, cand)
            if (placement, lifecycle) == action.pre_state:
                continue
            if action.kind != "delete" and placement is not None and placement != action.source:
                counts["duplicate_placements"] += 1
            else:
                counts["invalid_transitions"] += 1

    # Execution
    def _start_ready_waves(self, time):
        """Start admitted waves whose blocking predecessors are all finished."""
        for cand, wave_index in sorted(
            self.waiting_waves, key=lambda x: (x[0].admitted_at, x[0].id, x[1])
        ):
            if (cand, wave_index) not in self.waiting_waves:
                continue
            if cand.status != ADMITTED:
                self.waiting_waves.remove((cand, wave_index))
                continue
            wave = cand.waves[wave_index]
            if any(
                predecessor in self.active for action in wave for predecessor in action.predecessors
            ):
                continue
            self.waiting_waves.remove((cand, wave_index))
            failed = [
                a for a in wave if (self.m_act[a.component], self.lc[a.component]) != a.pre_state
            ]
            if failed:
                self.counts["precondition_failures"] += 1
                self.log(
                    time,
                    "candidate %d wave %d: precondition failed for %s"
                    % (cand.id, wave_index, ", ".join(a.component for a in failed)),
                )
                self._abort(cand, time, "precondition")
                continue
            for action in wave:
                self._start_action(action, cand, time)

    def _start_action(self, action, cand, time):
        """Start one admitted action: lifecycle, finish time and, when modeled, its failure."""
        durations = getattr(cand.request, "durations", {})
        duration = durations.get(action.component, self.duration[action.component])
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 1:
            raise ValueError("action duration must be a positive integer")
        action.start, action.finish = time, time + duration
        self.lc[action.component] = TRANSIENT[action.kind]
        if self.failure is not None:
            rng = stream(
                self.seed, "failure", cand.request.id, cand.attempt, action.kind, action.component
            )
            offset = self.failure(action, cand.request, cand.attempt, rng)
            if offset is not None:
                action.fail_at = min(time + max(1, int(offset)), action.finish)
        self.log(
            time,
            "%s(%s) starts, ends %d%s"
            % (
                action.kind,
                action.component,
                action.finish,
                "" if action.fail_at is None else ", fails at %d" % action.fail_at,
            ),
        )

    def _execution_events(self, time):
        due = [
            a
            for a in self.active.values()
            if a.start is not None
            and (a.fail_at == time or (a.fail_at is None and a.finish == time))
        ]
        for action in sorted(due, key=lambda a: a.id):
            if action.fail_at == time:
                self._fail(action, time)
            else:
                self._complete(action, time)
        if due:
            self._start_ready_waves(time)
        return bool(due)

    def _finish_action(self, action):
        self.active.pop(action.id)
        self.inflight[action.component].remove(action.id)

    def _complete(self, action, time):
        """Completion write: M_act, LC and Res updated, versions bumped."""
        c = action.component
        self.ledger.pop(action.id, None)
        if action.kind == "delete":
            self.m_act[c], self.lc[c] = None, UND
        else:
            self.m_act[c], self.lc[c] = action.destination, DEP
        self.touch(("comp", c), ("host", action.source), ("host", action.destination))
        self._finish_action(action)
        action.done = True
        self.counts["completed_actions"] += 1
        self.log(time, "%s(%s) completed" % (action.kind, c))
        cand = self.candidates[action.candidate]
        if cand.status == ADMITTED and all(a.done for a in cand.actions):
            self._finish_request(cand, time)
            if self.planners[cand.planner]["held"] is cand:
                self._release_candidate(cand)
        self._settle_dependents(action, time, failed=False)

    def _fail(self, action, time):
        """Failure before completion: compensation or, in the ablation, leftovers."""
        c = action.component
        self.counts["failures"] += 1
        self.log(time, "%s(%s) failed" % (action.kind, c))
        self._finish_action(action)
        cand = self.candidates[action.candidate]
        if self.compensate:
            self.ledger.pop(action.id, None)
            self.lc[c] = STABLE_BEFORE.get(self.lc[c], self.lc[c])
            self.m_tgt[c] = self.view(c)  # placement the remaining admitted work leaves, else M_act
            self.touch(("comp", c), ("host", action.destination))
            self.counts["compensations"] += 1
        else:
            self._leak(action.id)
        self._settle_dependents(action, time, failed=True)
        self._abort(cand, time, "failure")

    def _leak(self, action_id):
        entry = self.ledger.get(action_id)
        if entry is not None and not entry[3]:
            entry[3] = True
            self.counts["leaked_reservations"] += 1
            self.counts["leaked_amount"] += entry[2]

    def _abort(self, cand, time, cause):
        """Cancel the unfinished part of a candidate after a failure or a failed guard."""
        for action in cand.actions:
            if action.id in self.active and action.start is None:
                self._finish_action(action)
                if self.compensate:
                    self.ledger.pop(action.id, None)
                    self.m_tgt[action.component] = self.view(action.component)
                    self.touch(("comp", action.component), ("host", action.destination))
                else:
                    self._leak(action.id)
                self._settle_dependents(action, time, failed=True)
        self.waiting_waves = [entry for entry in self.waiting_waves if entry[0] is not cand]
        already_cancelled = cand.status == CANCELLED
        cand.status = CANCELLED
        if self.planners[cand.planner]["held"] is cand:
            self._release_candidate(cand)
        if already_cancelled:
            return  # a second failed action of the same candidate: the request is already handled
        if self.compensate:
            self._recompute(cand.request, time, cause)
        else:
            self._lose(cand.request)

    # Results
    def final_audit(self):
        """End-state consistency of K after quiescence (observer only)."""
        loads = self.loads()
        audit = dict(
            final_target_actual_divergence=sum(self.m_tgt[c] != self.m_act[c] for c in self.dem),
            final_ledger_entries=len(self.ledger),
            final_transient_lifecycles=sum(state in STABLE_BEFORE for state in self.lc.values()),
            final_overcommitted_hosts=sum(loads[h] > self.cap[h] for h in self.hosts),
            final_active_actions=len(self.active),
        )
        audit["final_consistent"] = int(not any(audit.values()))
        return audit

    def metrics(self):
        result = dict(self.counts)
        result["element_comparisons"] = (
            result["comparisons_fresh"]
            + result["comparisons_pairwise"]
            + result["comparisons_aggregate"]
        )
        admitted = max(1, result["admitted"])
        result["evaluations_per_admission"] = result["predicate_evaluations"] / admitted
        result["comparisons_per_admission"] = result["element_comparisons"] / admitted
        for name, values in (
            ("admission_latency_mean", self.latencies),
            ("planner_wait_mean", self.planner_waits),
            ("response_mean", self.responses),
        ):
            result[name] = sum(values) / len(values) if values else 0.0
        ordered = sorted(self.responses)
        result["response_p95"] = (
            ordered[min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))]
            if ordered
            else 0.0
        )
        result["end_time"] = self.end_time
        result.update(self.final_audit())
        return result


# Scope versions

REJECTED_SUPERSEDED = "rejected_superseded"


@dataclass
class ScopedRequest(Request):
    read_scope: Tuple[str, ...] = ()


@dataclass
class GeneratedCandidate(Candidate):
    generation_key: Optional[Tuple[int, int]] = None
    generation: Optional[int] = None


def extra_reads(request):
    """Read-only component entries, excluding already checked write entries."""
    return set(getattr(request, "read_scope", ())) - set(request.scope)


class ScopedGenerationContract(ContractSimulator):
    """Admission contract with read-only scopes and revision generations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_frontier = {}
        self.superseded_candidates = []
        for request in self.requests:
            self._validate_scope(request)

    def _validate_scope(self, request):
        unknown = (set(request.scope) | set(getattr(request, "read_scope", ()))) - set(self.dem)
        if unknown:
            raise ValueError("unknown component scope: %s" % sorted(unknown))
        if extra_reads(request) and not self.check_fresh:
            raise ValueError("read-only extension is evaluated with Fresh enabled")
        if extra_reads(request) and len(request.scope) != 1:
            raise ValueError("read-only extension currently requires one written component")

    def publish_generation(self, key, generation, time):
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(not isinstance(part, int) for part in key)
        ):
            raise ValueError("generation key must be (adaptation_id, planning_session_id)")
        if not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        old = self.generation_frontier.get(key, -1)
        if generation > old:
            self.generation_frontier[key] = generation
            self.log(time, "generation frontier %s advances to %d" % (key, generation))
        return self.generation_frontier[key]

    def _discard_superseded(self, cand, time):
        if cand.status == DELAYED:
            self.delayed.remove(cand)
        cand.status = REJECTED_SUPERSEDED
        self.superseded_candidates.append(cand.id)
        self.counts["predicate_evaluations"] += 1
        if cand.first_check is None:
            cand.first_check = time
        self._release_owned_planner(cand)
        self.log(
            time,
            "candidate %d discarded: generation %d < frontier %d for %s"
            % (
                cand.id,
                cand.generation,
                self.generation_frontier[cand.generation_key],
                cand.generation_key,
            ),
        )
        return REJECTED_SUPERSEDED

    def _release_owned_planner(self, cand):
        # A controlled delayed revision can be superseded while its original
        # slot is serving a newer candidate. Never release the newer occupant.
        planner = self.planners[cand.planner]
        if planner["candidate"] is cand or planner["held"] is cand:
            self._release(cand.planner)

    def _admit(self, cand, time):
        self._validate_scope(cand.request)
        if cand.status in (ADMITTED, REJECTED_SUPERSEDED):
            return cand.status  # publication never retracts an existing commitment
        if isinstance(cand, GeneratedCandidate):
            if cand.generation_key is None or cand.generation is None:
                raise ValueError("generated candidate needs a key and generation")
            frontier = self.generation_frontier.get(cand.generation_key)
            if frontier is None or cand.generation > frontier:
                raise ValueError("publish the generation before presenting its candidate")
            if cand.generation < frontier:
                return self._discard_superseded(cand, time)

        read_only = extra_reads(cand.request)
        self.counts["comparisons_fresh"] += len(read_only)
        changed = [
            ("comp", c)
            for c in sorted(read_only)
            if cand.snapshot.versions[("comp", c)] != self.versions[("comp", c)]
        ]
        if changed:
            self.counts["predicate_evaluations"] += 1
            self.counts["rejected_stale"] += 1
            if cand.first_check is None:
                cand.first_check = time
            if cand.status == DELAYED:
                self.delayed.remove(cand)
            cand.status, cand.stale = REJECTED_STALE, True
            self.log(time, "candidate %d stale on read-only entries %s" % (cand.id, changed))
            self._release_owned_planner(cand)
            self._recompute(cand.request, time, "stale")
            return REJECTED_STALE

        # Capture the dependency even if the current placement equals its final
        # projection (e.g. queued x:h0->h1->h0). Equality does not make an
        # admitted, unfinished chain effective.
        cand._read_dependencies = {self.inflight[c][-1] for c in read_only if self.inflight[c]}
        result = super()._admit(cand, time)
        if result == ADMITTED and not cand.actions and cand in self.pending_noops:
            # The base may already be waiting for written placements. Include
            # read-only predecessor outcomes as well, without adding targets.
            cand.waits_on.update(cand._read_dependencies)
        return result

    def _action(self, kind, component, source, destination, cand, wave, footprint):
        action = super()._action(kind, component, source, destination, cand, wave, footprint)
        action.reads = action.reads | frozenset(("comp", c) for c in extra_reads(cand.request))
        return action

    def _read_guards_hold(self, cand):
        return all(
            (self.m_act[c], self.lc[c])
            == (cand.snapshot.view[c], DEP if cand.snapshot.view[c] is not None else UND)
            for c in extra_reads(cand.request)
        )

    def _start_ready_waves(self, time):
        for cand, index in list(self.waiting_waves):
            if cand.status != ADMITTED:
                continue
            wave = cand.waves[index]
            if any(pred in self.active for action in wave for pred in action.predecessors):
                continue
            if not self._read_guards_hold(cand):
                self.counts["precondition_failures"] += 1
                self.log(time, "candidate %d: read-only execution precondition failed" % cand.id)
                self._abort(cand, time, "precondition")
        super()._start_ready_waves(time)

    def _finish_request(self, cand, time):
        if not cand.actions:
            dependencies = getattr(cand, "_read_dependencies", set()) & set(self.active)
            if dependencies:
                cand.waits_on.update(dependencies)
                if cand not in self.pending_noops:
                    self.pending_noops.append(cand)
                    self.counts["empty_waiting"] += 1
                self.log(
                    time,
                    "empty candidate %d waits for read-only predecessor(s) %s"
                    % (cand.id, sorted(dependencies)),
                )
                return
            if not self._read_guards_hold(cand):
                cand.status = CANCELLED
                self.counts["precondition_failures"] += 1
                self._recompute(cand.request, time, "precondition")
                return
        super()._finish_request(cand, time)

    def metrics(self):
        result = super().metrics()
        result["rejected_superseded"] = len(self.superseded_candidates)
        return result


# Application dependencies


@dataclass
class DagRequest(ScopedRequest):
    application: str = ""
    operation: str = "deploy"


class PlacementUnavailable(Exception):
    """The placement search produced no candidate, with an explicit reason."""

    def __init__(self, reason="infeasible"):
        self.reason = str(reason)
        super().__init__(self.reason)


class DagContractSimulator(ScopedGenerationContract):
    """DAG applications: latency bounds, guards on retained components, effective completion."""

    def __init__(self, hosts, components, requests, *, applications, distances, **kwargs):
        if (
            kwargs.get("check_fresh", True) is not True
            or kwargs.get("check_aggregate", True) is not True
        ):
            raise ValueError("evaluation C requires freshness and aggregate checks")
        if kwargs.get("compensate", True) is not True:
            raise ValueError("evaluation C requires compensation")
        self.applications = {
            app: {"components": tuple(spec["components"]), "edges": tuple(spec["edges"])}
            for app, spec in applications.items()
        }
        self.distances = {h: dict(row) for h, row in distances.items()}
        self._immutable = {c for c, spec in components.items() if spec.get("immutable", False)}
        self._background = {c for c, spec in components.items() if spec.get("background", False)}
        self._initial_immutable = {c: components[c].get("host") for c in self._immutable}
        self._validate_applications(hosts, components)
        super().__init__(hosts, components, requests, **kwargs)
        for request in self.requests:
            self._validate_dag_request(request)
        self.counts.update(
            planning_infeasible=0,
            unavailable_applications=0,
            planning_search_exhausted=0,
            planning_no_change=0,
            planning_unavailable=0,
            dag_latency_violations=0,
            dag_admission_rejections=0,
            dag_precondition_failures=0,
        )
        self.planning_failures = []
        self._dag_violation_active = set()
        self._total_capacity = sum(self.cap.values())
        self._occupancy_initial_actual = self._actual_occupancy()
        self._occupancy_initial_committed = self._committed_occupancy()
        self._occupancy_last_actual = self._occupancy_initial_actual
        self._occupancy_last_committed = self._occupancy_initial_committed
        self._occupancy_last_time = 0
        self._occupancy_actual_area = 0.0
        self._occupancy_committed_area = 0.0
        self._occupancy_actual_peak = self._occupancy_initial_actual
        self._occupancy_committed_peak = self._occupancy_initial_committed

    def _validate_applications(self, hosts, components):
        if not hosts or any(cap <= 0 for cap in hosts.values()):
            raise ValueError("host capacities must be positive")
        seen = set()
        for app, spec in self.applications.items():
            members = set(spec["components"])
            if not members or len(members) != len(spec["components"]):
                raise ValueError("application components must be nonempty and unique")
            if members - set(components) or members & seen or members & self._immutable:
                raise ValueError("applications need known, disjoint, mutable components")
            seen.update(members)
            successors = {c: [] for c in members}
            indegree = {c: 0 for c in members}
            for source, target, limit in spec["edges"]:
                if source not in members or target not in members or limit < 0:
                    raise ValueError("invalid application edge")
                successors[source].append(target)
                indegree[target] += 1
            ready = [c for c in members if indegree[c] == 0]
            visited = 0
            while ready:
                source = ready.pop()
                visited += 1
                for target in successors[source]:
                    indegree[target] -= 1
                    if indegree[target] == 0:
                        ready.append(target)
            if visited != len(members):
                raise ValueError("application graph must be acyclic")
        for source in hosts:
            if source not in self.distances:
                raise ValueError("missing distance row")
            for target in hosts:
                if target not in self.distances[source] or self.distances[source][target] < 0:
                    raise ValueError("distances must cover all host pairs")
            if self.distances[source][source] != 0:
                raise ValueError("distance from a host to itself must be zero")

    def _validate_dag_request(self, request):
        if not isinstance(request, DagRequest) or request.application not in self.applications:
            raise ValueError("evaluation C requests need a known application")
        members = set(self.applications[request.application]["components"])
        writes, reads = set(request.scope), set(request.read_scope)
        if len(writes) != len(request.scope) or writes & self._immutable:
            raise ValueError("invalid write scope")
        if request.operation == "deploy":
            if writes != members or reads != members:
                raise ValueError("deployment must write and read the full application scope")
        elif request.operation == "migrate":
            if len(writes) != 1 or not writes <= members or reads != members:
                raise ValueError("migration must write one component and read its application")
        else:
            raise ValueError("operation must be deploy or migrate")

    def _emissions(self, time):
        # Distinguish a failed search from a completed deployment.
        for index, planner in enumerate(self.planners):
            if planner["emit"] != time:
                continue
            request = planner["request"]
            rng = stream(self.seed, "plan", request.id, request.attempt)
            candidate = None
            try:
                mapping, redeploy = request.plan(
                    planner["snapshot"], request.scope, request.attempt, rng
                )
                candidate = Candidate(
                    len(self.candidates),
                    request,
                    index,
                    request.attempt,
                    time,
                    planner["snapshot"],
                    dict(mapping),
                    set(redeploy),
                )
                self._prepare_candidate(candidate)
            except PlacementUnavailable as error:
                reason = error.reason
                self.counts["planning_unavailable"] += 1
                if reason == "incomplete_application":
                    self.counts["unavailable_applications"] += 1
                elif reason == "search_exhausted":
                    self.counts["planning_search_exhausted"] += 1
                elif reason == "no_migration":
                    self.counts["planning_no_change"] += 1
                else:
                    self.counts["planning_infeasible"] += 1
                self.planning_failures.append(
                    dict(request=request.id, attempt=request.attempt, time=time, reason=reason)
                )
                request.outcome_reason = reason
                self.log(time, "request %d placement unavailable: %s" % (request.id, reason))
                self._lose(request)
                self._release(index)
                continue
            self.candidates.append(candidate)
            self.counts["candidates"] += 1
            planner.update(emit=None, candidate=candidate)
            self._admit(candidate, time)

    def _prepare_candidate(self, cand):
        self._validate_dag_request(cand.request)
        members = self.applications[cand.request.application]["components"]
        if set(cand.mapping) != set(cand.request.scope):
            raise ValueError("a DAG candidate must map exactly its write scope")
        if cand.redeploy:
            raise ValueError("evaluation C uses direct deployment or migration, not redeployment")
        if any(host is not None and host not in self.cap for host in cand.mapping.values()):
            raise ValueError("unknown candidate host")
        if cand.request.operation == "deploy":
            for c in members:
                if cand.snapshot.view[c] is not None and cand.mapping[c] != cand.snapshot.view[c]:
                    raise ValueError("deployment recovery must retain existing placements")
            if any(cand.mapping[c] is None for c in members):
                raise PlacementUnavailable("infeasible")
        elif any(cand.snapshot.view[c] is None for c in members):
            raise PlacementUnavailable("incomplete_application")
        elif any(cand.mapping[c] is None for c in cand.request.scope):
            raise ValueError("migration cannot remove a component")
        written = {c for c in cand.request.scope if cand.snapshot.view[c] != cand.mapping[c]}
        if cand.request.operation == "migrate" and not written:
            raise PlacementUnavailable("no_migration")
        cand._dag_retained = set(members) - written
        cand._dag_read_dependencies = {
            self.inflight[c][-1] for c in cand._dag_retained if self.inflight[c]
        }
        cand._dag_failed_reads = set()

    def _admit(self, cand, time):
        if not hasattr(cand, "_dag_retained"):
            self._prepare_candidate(cand)
        # Refresh predecessor identities on a resource recheck, without changing
        # what was read in the snapshot. Fresh rejects changed component entries.
        cand._dag_read_dependencies.update(
            self.inflight[c][-1] for c in cand._dag_retained if self.inflight[c]
        )
        result = super()._admit(cand, time)
        if result == ADMITTED:
            self._occupancy_committed_peak = max(
                self._occupancy_committed_peak, self._committed_occupancy()
            )
        return result

    def _derive_waves(self, cand):
        super()._derive_waves(cand)
        retained_reads = frozenset(("comp", c) for c in cand._dag_retained)
        for action in cand.actions:
            action.reads |= retained_reads

    def _projected_mapping(self, cand):
        members = self.applications[cand.request.application]["components"]
        mapping = {c: cand.snapshot.view[c] for c in members}
        mapping.update(cand.mapping)
        return mapping

    def _network_holds(self, application, mapping, require_complete=True):
        spec = self.applications[application]
        if require_complete and any(mapping[c] is None for c in spec["components"]):
            return False
        return all(
            source_host is None
            or target_host is None
            or self.distances[source_host][target_host] <= limit
            for source, target, limit in spec["edges"]
            for source_host, target_host in [(mapping[source], mapping[target])]
        )

    def _invariant(self, cand, actions):
        feasible = super()._invariant(cand, actions)
        edges = self.applications[cand.request.application]["edges"]
        self.counts["comparisons_aggregate"] += len(edges)
        mapping = self._projected_mapping(cand)
        network = self._network_holds(cand.request.application, mapping)
        if not network:
            self.counts["dag_admission_rejections"] += 1
        return feasible and network

    def _read_guards_hold(self, cand):
        retained = getattr(cand, "_dag_retained", set())
        return not getattr(cand, "_dag_failed_reads", set()) and all(
            (self.m_act[c], self.lc[c])
            == (cand.snapshot.view[c], DEP if cand.snapshot.view[c] is not None else UND)
            for c in retained
        )

    def _start_ready_waves(self, time):
        # Preserve the base scheduler order and guards; add retained-component
        # reads, network validation, and optional per-request action durations.
        for cand, wave_index in sorted(
            self.waiting_waves, key=lambda entry: (entry[0].admitted_at, entry[0].id, entry[1])
        ):
            if (cand, wave_index) not in self.waiting_waves:
                continue
            if cand.status != ADMITTED:
                self.waiting_waves.remove((cand, wave_index))
                continue
            wave = cand.waves[wave_index]
            if any(
                predecessor in self.active for action in wave for predecessor in action.predecessors
            ):
                continue
            self.waiting_waves.remove((cand, wave_index))
            ordinary_guard = all(
                (self.m_act[a.component], self.lc[a.component]) == a.pre_state for a in wave
            )
            dag_guard = self._read_guards_hold(cand) and self._network_holds(
                cand.request.application, self._projected_mapping(cand)
            )
            if not ordinary_guard or not dag_guard:
                self.counts["precondition_failures"] += 1
                self.counts["dag_precondition_failures"] += not dag_guard
                self.log(
                    time,
                    "candidate %d wave %d: DAG execution precondition failed"
                    % (cand.id, wave_index),
                )
                self._abort(cand, time, "precondition")
                continue
            for action in wave:
                self._start_action(action, cand, time)

    def _settle_dependents(self, action, time, failed):
        if failed:
            for cand in self.candidates:
                if cand.status == ADMITTED and action.id in getattr(
                    cand, "_dag_read_dependencies", set()
                ):
                    cand._dag_failed_reads.add(action.id)
        super()._settle_dependents(action, time, failed)

    def _finish_request(self, cand, time):
        # Reservations are never effective deployment. This check also covers
        # retained siblings when a partial deployment has been retried.
        dependencies = getattr(cand, "_dag_read_dependencies", set()) & set(self.active)
        if dependencies:
            cand.waits_on.update(dependencies)
            if cand not in self.pending_noops:
                self.pending_noops.append(cand)
                self.counts["empty_waiting"] += 1
            return
        members = self.applications[cand.request.application]["components"]
        effective = (
            all(self.m_act[c] is not None and self.lc[c] == DEP for c in members)
            and all(self.m_act[c] == cand.mapping[c] for c in cand.request.scope)
            and self._read_guards_hold(cand)
            and self._network_holds(cand.request.application, self.m_act)
        )
        if not effective:
            cand.status = CANCELLED
            self.counts["precondition_failures"] += 1
            self.counts["dag_precondition_failures"] += 1
            self._release_candidate(cand)
            self._recompute(cand.request, time, "precondition")
            return
        if cand.request.finished is not None:
            raise AssertionError("a DAG request cannot complete twice")
        cand.request.outcome_reason = "completed"
        # Skip ScopedGenerationContract's no-op-only checks: the stronger guard
        # above applies to every DAG candidate, whether empty or nonempty.
        super(ScopedGenerationContract, self)._finish_request(cand, time)

    def _actual_occupancy(self):
        return (
            sum(self.dem[c] for c, h in self.m_act.items() if h is not None) / self._total_capacity
        )

    def _committed_occupancy(self):
        return (
            sum(self.dem[c] for c, h in self.m_act.items() if h is not None)
            + sum(entry[2] for entry in self.ledger.values())
        ) / self._total_capacity

    def _actual_edge_violations(self):
        violations = set()
        for app, spec in self.applications.items():
            for index, (source, target, limit) in enumerate(spec["edges"]):
                hs, ht = self.m_act[source], self.m_act[target]
                if hs is not None and ht is not None and self.distances[hs][ht] > limit:
                    violations.add((app, index))
        return violations

    def _observe(self, time):
        elapsed = time - self._occupancy_last_time
        self._occupancy_actual_area += elapsed * self._occupancy_last_actual
        self._occupancy_committed_area += elapsed * self._occupancy_last_committed
        self._occupancy_last_time = time
        self._occupancy_last_actual = self._actual_occupancy()
        self._occupancy_last_committed = self._committed_occupancy()
        self._occupancy_actual_peak = max(self._occupancy_actual_peak, self._occupancy_last_actual)
        self._occupancy_committed_peak = max(
            self._occupancy_committed_peak, self._occupancy_last_committed
        )
        violations = self._actual_edge_violations()
        self.counts["dag_latency_violations"] += len(violations - self._dag_violation_active)
        self._dag_violation_active = violations
        super()._observe(time)

    def final_audit(self):
        result = super().final_audit()
        result["final_dag_violations"] = len(self._actual_edge_violations())
        result["final_immutable_changes"] = sum(
            self.m_act[c] != h for c, h in self._initial_immutable.items()
        )
        result["final_deployed_applications"] = sum(
            all(self.m_act[c] is not None and self.lc[c] == DEP for c in spec["components"])
            for spec in self.applications.values()
        )
        result["final_consistent"] = int(
            bool(result["final_consistent"])
            and not result["final_dag_violations"]
            and not result["final_immutable_changes"]
        )
        return result

    def metrics(self):
        result = super().metrics()
        duration = self.end_time
        result.update(
            occupancy_actual_initial=self._occupancy_initial_actual,
            occupancy_committed_initial=self._occupancy_initial_committed,
            occupancy_actual_mean=(
                self._occupancy_actual_area / duration if duration else self._occupancy_last_actual
            ),
            occupancy_committed_mean=(
                self._occupancy_committed_area / duration
                if duration
                else self._occupancy_last_committed
            ),
            occupancy_actual_final=self._actual_occupancy(),
            occupancy_committed_final=self._committed_occupancy(),
            occupancy_actual_peak=self._occupancy_actual_peak,
            occupancy_committed_peak=self._occupancy_committed_peak,
            background_occupancy=sum(
                self.dem[c] for c in self._background if self.m_act[c] is not None
            )
            / self._total_capacity,
        )
        return result


# Host and network feasibility


def link_key(left, right):
    """Canonical direct-link identifier; colocated endpoints use no link."""
    return tuple(sorted((left, right)))


def network_application_loads(applications, edge_demands, endpoint_options):
    """Return each application's conservative bandwidth on each direct link."""
    result = {}
    for app, spec in applications.items():
        loads = Counter()
        for left, right, _ in spec["edges"]:
            demand = edge_demands[(app, left, right)]
            links = {
                link_key(hs, ht)
                for hs in endpoint_options.get(left, ())
                for ht in endpoint_options.get(right, ())
                if hs is not None and ht is not None and hs != ht
            }
            for link in links:
                loads[link] += demand
        result[app] = dict(loads)
    return result


def network_envelope_loads(applications, edge_demands, endpoint_options):
    """Aggregate reservations across all distinct application edges."""
    total = Counter()
    for loads in network_application_loads(applications, edge_demands, endpoint_options).values():
        total.update(loads)
    return dict(total)


def candidate_network_feasible(snapshot, mapping, applications):
    """Check trial destinations without reclaiming existing reservations."""
    affected = {
        app: spec for app, spec in applications.items() if set(spec["components"]) & set(mapping)
    }
    options = dict(snapshot.network_options)
    for component, host in mapping.items():
        if host is not None:
            options[component] = frozenset(options.get(component, ())) | {host}
    loads = Counter(snapshot.network_loads)
    for app in affected:
        loads.subtract(snapshot.network_application_loads.get(app, {}))
    for values in network_application_loads(
        affected, snapshot.network_edge_demands, options
    ).values():
        loads.update(values)
    return all(
        load <= snapshot.network_link_capacities.get(link, -1)
        for link, load in loads.items()
        if load > 0
    )


class NetworkDagContractSimulator(DagContractSimulator):
    """Application placement with aggregate link capacity and finite k_E."""

    def __init__(
        self,
        hosts,
        components,
        requests,
        *,
        applications,
        distances,
        link_capacities,
        edge_demands,
        executors=16,
        **kwargs,
    ):
        if kwargs.get("policy", "macpe") == "sequential":
            kwargs.setdefault("planners", 1)
            if kwargs["planners"] != 1:
                raise ValueError("the sequential baseline requires exactly one planner")
        if isinstance(executors, bool) or not isinstance(executors, int) or executors < 1:
            raise ValueError("executors must be a positive integer")
        self.executors = executors
        self.link_capacities = {}
        for pair, capacity in link_capacities.items():
            if (
                len(pair) != 2
                or pair[0] == pair[1]
                or set(pair) - set(hosts)
                or isinstance(capacity, bool)
                or not math.isfinite(capacity)
                or capacity <= 0
            ):
                raise ValueError("invalid direct-link capacity")
            key = link_key(*pair)
            if key in self.link_capacities:
                raise ValueError("duplicate undirected link")
            self.link_capacities[key] = capacity
        if set(self.link_capacities) != set(combinations(sorted(hosts), 2)):
            raise ValueError("the infrastructure must contain every undirected host pair")
        expected = {
            (app, left, right)
            for app, spec in applications.items()
            for left, right, _ in spec["edges"]
        }
        if set(edge_demands) != expected:
            raise ValueError("bandwidth demands must cover exactly the application edges")
        if any(
            isinstance(d, bool) or not math.isfinite(d) or d <= 0 for d in edge_demands.values()
        ):
            raise ValueError("application-edge bandwidth demands must be positive")
        self.edge_demands = dict(edge_demands)
        self.distinct_host_applications = {
            app for app, spec in applications.items() if spec.get("distinct_hosts", False)
        }
        self._executor_ready = {}
        self._executor_waits = []
        self.executor_starts = []
        self._network_actual_violations = set()
        self._network_reserved_violations = set()
        self._distinct_violations = set()
        super().__init__(
            hosts, components, requests, applications=applications, distances=distances, **kwargs
        )
        self.counts.update(
            network_capacity_violations=0,
            network_reservation_violations=0,
            network_admission_rejections=0,
            distinct_host_violations=0,
            executor_peak_running_waves=0,
            executor_capacity_violations=0,
        )
        if self._overloaded(self.network_loads()):
            raise ValueError("initial placement exceeds direct-link capacity")
        if self._actual_distinct_violations():
            raise ValueError("initial placement violates application anti-affinity")

    def _start_planners(self, time):
        # Arrivals keep their FIFO order while the single staged controller
        # finishes Execution. A failed candidate may release its planner slot
        # before its already-started siblings finish; those actions still belong
        # to the Execution phase and must complete before the next snapshot.
        while (
            self._arrival_cursor < len(self.requests)
            and self.requests[self._arrival_cursor].arrival <= time
        ):
            self.queue.append(self.requests[self._arrival_cursor])
            self._arrival_cursor += 1
        if self.policy == "sequential" and self.active:
            return
        super()._start_planners(time)

    def network_options(self):
        """Hosts a component may occupy: placement, in-flight destinations, leaked reservations."""
        options = {c: set() if host is None else {host} for c, host in self.m_act.items()}
        for action in self.active.values():
            if action.destination is not None:
                options[action.component].add(action.destination)
        for component, host, _, leaked in self.ledger.values():
            if leaked and host is not None:
                options[component].add(host)
        return {c: frozenset(hosts) for c, hosts in options.items()}

    def network_loads(self, actual_only=False):
        options = (
            {c: () if h is None else (h,) for c, h in self.m_act.items()}
            if actual_only
            else self.network_options()
        )
        return network_envelope_loads(self.applications, self.edge_demands, options)

    def _overloaded(self, loads):
        return {link for link, load in loads.items() if load > self.link_capacities.get(link, -1)}

    def snapshot(self, time):
        result = super().snapshot(time)
        result.network_options = self.network_options()
        result.network_application_loads = network_application_loads(
            self.applications, self.edge_demands, result.network_options
        )
        total = Counter()
        for values in result.network_application_loads.values():
            total.update(values)
        result.network_loads = dict(total)
        result.network_link_capacities = dict(self.link_capacities)
        result.network_edge_demands = dict(self.edge_demands)
        return result

    def _network_holds(self, application, mapping, require_complete=True):
        if not super()._network_holds(application, mapping, require_complete):
            return False
        if application in self.distinct_host_applications:
            hosts = [
                mapping[c]
                for c in self.applications[application]["components"]
                if mapping[c] is not None
            ]
            return len(hosts) == len(set(hosts))
        return True

    def _invariant(self, cand, actions):
        feasible = super()._invariant(cand, actions)
        # Recheck the aggregate link envelope with the candidate's destinations added.
        options = self.network_options()
        for action in actions:
            if action.destination is not None:
                options[action.component] = options[action.component] | {action.destination}
        loads = network_envelope_loads(self.applications, self.edge_demands, options)
        network_ok = not self._overloaded(loads)
        if not network_ok:
            self.counts["network_admission_rejections"] += 1
        return feasible and network_ok

    def _running_waves(self):
        return {
            (action.candidate, action.wave)
            for action in self.active.values()
            if action.start is not None
        }

    def _start_ready_waves(self, time):
        # These are the inherited DAG guards and action-start semantics, with a
        # bounded executor pool inserted only after dependency readiness.
        # This hook also observes newly committed reservations before another
        # admission or completion at the same timestamp can hide an overload.
        self._record_network_state()
        for cand, wave_index in sorted(
            self.waiting_waves, key=lambda entry: (entry[0].admitted_at, entry[0].id, entry[1])
        ):
            if (cand, wave_index) not in self.waiting_waves:
                continue
            if cand.status != ADMITTED:
                self.waiting_waves.remove((cand, wave_index))
                continue
            wave = cand.waves[wave_index]
            if any(pred in self.active for action in wave for pred in action.predecessors):
                continue
            key = (cand.id, wave_index)
            ready = self._executor_ready.setdefault(key, time)
            if len(self._running_waves()) >= self.executors:
                continue
            self.waiting_waves.remove((cand, wave_index))
            ordinary_guard = all(
                (self.m_act[a.component], self.lc[a.component]) == a.pre_state for a in wave
            )
            dag_guard = self._read_guards_hold(cand) and self._network_holds(
                cand.request.application, self._projected_mapping(cand)
            )
            if not ordinary_guard or not dag_guard:
                self.counts["precondition_failures"] += 1
                self.counts["dag_precondition_failures"] += not dag_guard
                self._abort(cand, time, "precondition")
                continue
            self._executor_waits.append(time - ready)
            self.executor_starts.append(
                dict(candidate=cand.id, wave=wave_index, ready=ready, start=time, wait=time - ready)
            )
            for action in wave:
                self._start_action(action, cand, time)
            self.counts["executor_peak_running_waves"] = max(
                self.counts["executor_peak_running_waves"], len(self._running_waves())
            )

    def _actual_distinct_violations(self):
        violations = set()
        for app in self.distinct_host_applications:
            hosts = [
                self.m_act[c]
                for c in self.applications[app]["components"]
                if self.m_act[c] is not None
            ]
            if len(hosts) != len(set(hosts)):
                violations.add(app)
        return violations

    def _record_network_state(self):
        actual = self._overloaded(self.network_loads(actual_only=True))
        reserved = self._overloaded(self.network_loads())
        distinct = self._actual_distinct_violations()
        self.counts["network_capacity_violations"] += len(actual - self._network_actual_violations)
        self.counts["network_reservation_violations"] += len(
            reserved - self._network_reserved_violations
        )
        self.counts["distinct_host_violations"] += len(distinct - self._distinct_violations)
        self.counts["executor_capacity_violations"] += len(self._running_waves()) > self.executors
        self._network_actual_violations, self._network_reserved_violations = actual, reserved
        self._distinct_violations = distinct

    def _finish_action(self, action):
        super()._finish_action(action)
        # Completion updates actual placement before reaching this hook.
        self._record_network_state()

    def _observe(self, time):
        self._record_network_state()
        super()._observe(time)

    def final_audit(self):
        result = super().final_audit()
        actual, reserved = self.network_loads(True), self.network_loads()
        result.update(
            final_overcommitted_links=len(self._overloaded(actual)),
            final_overreserved_links=len(self._overloaded(reserved)),
            final_network_reservation_excess=sum(
                max(0, reserved.get(k, 0) - actual.get(k, 0)) for k in reserved
            ),
            final_distinct_host_violations=len(self._actual_distinct_violations()),
            final_running_waves=len(self._running_waves()),
        )
        result["final_consistent"] = int(
            bool(result["final_consistent"])
            and not any(
                result[k]
                for k in (
                    "final_overcommitted_links",
                    "final_overreserved_links",
                    "final_network_reservation_excess",
                    "final_distinct_host_violations",
                    "final_running_waves",
                )
            )
        )
        return result

    def metrics(self):
        result = super().metrics()
        result.update(
            executor_capacity=self.executors,
            executor_started_waves=len(self._executor_waits),
            executor_waited_waves=sum(wait > 0 for wait in self._executor_waits),
            executor_wait_mean=statistics.mean(self._executor_waits) if self._executor_waits else 0,
            executor_wait_max=max(self._executor_waits, default=0),
        )
        return result


# All-pending batch execution


@dataclass
class BatchRequest(DagRequest):
    members: tuple = ()
    member_attempts: dict = field(default_factory=dict)


class BatchedNetworkDagSimulator(NetworkDagContractSimulator):
    """Deployment simulator with whole-queue batching and original-request metrics."""

    def __init__(
        self,
        hosts,
        components,
        requests,
        *,
        window=0,
        coordination=0,
        batch_duration=None,
        **kwargs,
    ):
        for name, value in (("window", window), ("coordination", coordination)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if batch_duration is not None and not callable(batch_duration):
            raise ValueError("batch_duration must be callable(members, snapshot)")
        requests = list(requests)
        if len({r.id for r in requests}) != len(requests):
            raise ValueError("original request IDs must be unique")
        if any(not isinstance(r, DagRequest) or r.operation != "deploy" for r in requests):
            raise ValueError("batched placement currently supports deployment requests only")
        if len({r.application for r in requests}) != len(requests):
            raise ValueError("overlapping application revisions are not supported")
        scopes = [c for r in requests for c in r.scope]
        if len(scopes) != len(set(scopes)):
            raise ValueError("original deployment write scopes must be disjoint")
        self.window, self.coordination = window, coordination
        self.batch_duration = batch_duration
        self.batches = []
        self.terminal_outcomes = {}
        self._member_by_component = {c: r for r in requests for c in r.scope}
        self._time = 0
        self._last_gate = None
        self._original_failure = kwargs.pop("failure", None)
        super().__init__(
            hosts,
            components,
            requests,
            failure=self._member_failure if self._original_failure else None,
            **kwargs,
        )
        if self.policy not in ("macpe", "sequential"):
            raise ValueError("batch admission supports macpe or sequential policies")

    def _member_failure(self, action, batch, attempt, rng):
        """Preserve failure draws by original request, not batch membership."""
        member = self._member_by_component[action.component]
        attempt = batch.member_attempts[member.id]
        rng = stream(self.seed, "failure", member.id, attempt, action.kind, action.component)
        return self._original_failure(action, member, attempt, rng)

    def _next_time(self, time):
        following = super()._next_time(time)
        if self.window and self.queue:
            available = any(p["request"] is None for p in self.planners)
            if available and not (self.policy == "sequential" and self.active):
                boundary = (time // self.window + 1) * self.window
                following = boundary if following is None else min(following, boundary)
        return following

    def _start_planners(self, time):
        self._time = time
        while (
            self._arrival_cursor < len(self.requests)
            and self.requests[self._arrival_cursor].arrival <= time
        ):
            self.queue.append(self.requests[self._arrival_cursor])
            self._arrival_cursor += 1
        if self.window and (time % self.window or self._last_gate == time):
            return
        if self.policy == "sequential" and self.active:
            return
        index = next((i for i, p in enumerate(self.planners) if p["request"] is None), None)
        if index is None:
            return
        members = tuple(r for r in self.queue if not r.lost and r.finished is None)
        self.queue.clear()
        if not members:
            return
        if len({r.id for r in members}) != len(members):
            raise AssertionError("a pending request cannot occur twice in a batch")
        snapshot = self.snapshot(time)
        duration = (
            members[0].planning
            if self.batch_duration is None
            else self.batch_duration(members, snapshot)
        )
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 1:
            raise ValueError("batch Planning duration must be a positive integer")
        scope = tuple(c for r in members for c in r.scope)
        batch = BatchRequest(
            id=len(self.batches),
            arrival=min(r.arrival for r in members),
            scope=scope,
            planning=duration,
            plan=None,
            read_scope=scope,
            application=tuple(r.application for r in members),
            members=members,
            member_attempts={r.id: r.attempt for r in members},
        )
        batch.durations = {
            c: getattr(r, "durations", {}).get(c, self.duration[c])
            for r in members
            for c in r.scope
        }
        batch.started = time
        for member in members:
            if member.started is None:
                member.started = time
                self.planner_waits.append(time - member.arrival)
        self.batches.append(
            dict(
                id=batch.id,
                members=[r.id for r in members],
                planner=index,
                start=time,
                planning_duration=duration,
                planning_end=time + duration,
                check_time=time + duration + self.coordination,
                candidate_created=False,
            )
        )
        self.planners[index].update(
            request=batch, emit=time + duration, snapshot=snapshot, candidate=None, checking=False
        )
        self._last_gate = time

    def _batch_plan(self, batch, snapshot, time):
        """Greedily combine feasible DAGs using private cumulative commitments."""
        projected = deepcopy(snapshot)
        mapping, accepted = {}, []
        for member in batch.members:
            if member.lost or member.finished is not None:
                continue
            attempt = batch.member_attempts[member.id]
            rng = stream(self.seed, "plan", member.id, attempt)
            try:
                local, redeploy = member.plan(projected, member.scope, attempt, rng)
                if redeploy or set(local) != set(member.scope):
                    raise ValueError("a deployment planner must map its complete scope")
                if any(h is None or h not in self.cap for h in local.values()):
                    raise PlacementUnavailable("infeasible")
                if any(
                    projected.view[c] is not None and projected.view[c] != h
                    for c, h in local.items()
                ):
                    raise ValueError("deployment recovery must retain committed placements")
            except PlacementUnavailable as error:
                reason = error.reason
                self.counts["planning_unavailable"] += 1
                self.counts[
                    (
                        "planning_search_exhausted"
                        if reason == "search_exhausted"
                        else "planning_infeasible"
                    )
                ] += 1
                self.planning_failures.append(
                    dict(request=member.id, attempt=attempt, time=time, reason=reason)
                )
                member.outcome_reason = "planning_" + reason
                self._lose(member)
                continue
            accepted.append(member)
            mapping.update(local)
            for component, host in local.items():
                if projected.view[component] is None:
                    projected.free[host] -= self.dem[component]
                projected.view[component] = host
                projected.target[component] = host
                projected.network_options[component] = frozenset(
                    projected.network_options.get(component, ())
                ) | {host}
            projected.network_application_loads = network_application_loads(
                self.applications, self.edge_demands, projected.network_options
            )
            loads = Counter()
            for values in projected.network_application_loads.values():
                loads.update(values)
            projected.network_loads = dict(loads)
        batch.members = tuple(accepted)
        batch.scope = batch.read_scope = tuple(c for r in accepted for c in r.scope)
        batch.application = tuple(r.application for r in accepted)
        self.batches[batch.id]["planned_members"] = [r.id for r in accepted]
        return mapping

    def _emissions(self, time):
        self._time = time
        for index, planner in enumerate(self.planners):
            if planner["emit"] != time:
                continue
            if planner.get("checking", False):
                candidate = planner["candidate"]
                planner.update(emit=None, checking=False)
                self._admit(candidate, time)
                continue
            batch = planner["request"]
            mapping = self._batch_plan(batch, planner["snapshot"], time)
            if not batch.members:
                self.batches[batch.id]["check_time"] = None
                self._release(index)
                continue
            candidate = Candidate(
                len(self.candidates),
                batch,
                index,
                batch.attempt,
                time,
                planner["snapshot"],
                mapping,
                set(),
            )
            self._prepare_candidate(candidate)
            self.candidates.append(candidate)
            self.batches[batch.id]["candidate_created"] = True
            self.counts["candidates"] += 1
            planner.update(candidate=candidate, emit=None)
            if self.coordination:
                planner.update(checking=True, emit=time + self.coordination)
            else:
                self._admit(candidate, time)

    def _validate_dag_request(self, request):
        if not isinstance(request, BatchRequest):
            return super()._validate_dag_request(request)
        expected = tuple(c for r in request.members for c in r.scope)
        if (
            len(expected) != len(set(expected))
            or tuple(request.scope) != expected
            or tuple(request.read_scope) != expected
        ):
            raise ValueError("a batch must contain disjoint, complete deployment scopes")
        for member in request.members:
            super()._validate_dag_request(member)

    def _prepare_candidate(self, cand):
        self._validate_dag_request(cand.request)
        if cand.redeploy or set(cand.mapping) != set(cand.request.scope):
            raise ValueError("batch candidates must map their exact deployment scope")
        if any(h is None or h not in self.cap for h in cand.mapping.values()):
            raise PlacementUnavailable("infeasible")
        if any(
            cand.snapshot.view[c] is not None and cand.snapshot.view[c] != h
            for c, h in cand.mapping.items()
        ):
            raise ValueError("batch recovery must retain committed placements")
        cand._dag_retained = {
            c for c in cand.request.scope if cand.snapshot.view[c] == cand.mapping[c]
        }
        cand._dag_read_dependencies = {
            self.inflight[c][-1] for c in cand._dag_retained if self.inflight[c]
        }
        cand._dag_failed_reads = set()

    def _projected_mapping(self, cand):
        return dict(cand.mapping)

    def _network_holds(self, application, mapping, require_complete=True):
        if isinstance(application, tuple):
            return all(
                super(BatchedNetworkDagSimulator, self)._network_holds(
                    app, mapping, require_complete
                )
                for app in application
            )
        return super()._network_holds(application, mapping, require_complete)

    def _invariant(self, cand, actions):
        feasible = ContractSimulator._invariant(self, cand, actions)
        self.counts["comparisons_aggregate"] += sum(
            len(self.applications[a]["edges"]) for a in cand.request.application
        )
        valid = self._network_holds(cand.request.application, cand.mapping)
        if not valid:
            self.counts["dag_admission_rejections"] += 1
        options = self.network_options()
        for action in actions:
            if action.destination is not None:
                options[action.component] = options[action.component] | {action.destination}
        network_ok = not self._overloaded(
            network_envelope_loads(self.applications, self.edge_demands, options)
        )
        if not network_ok:
            self.counts["network_admission_rejections"] += 1
        return feasible and valid and network_ok

    def _derive_waves(self, cand):
        ContractSimulator._derive_waves(self, cand)
        # Start-time guards cover the batch's complete retained placement.
        # Its whole wave must therefore wait for those placements to complete,
        # including retained components of other applications in the batch.
        retained = frozenset(("comp", c) for c in cand._dag_retained)
        for action in cand.actions:
            action.reads |= retained

    def _member_complete(self, member):
        return all(
            self.m_act[c] is not None and self.lc[c] == DEP for c in member.scope
        ) and self._network_holds(member.application, self.m_act)

    def _record_completion(self, member, time):
        if member.finished is None and not member.lost and self._member_complete(member):
            member.finished = time
            member.outcome_reason = "completed"
            self.responses.append(time - member.arrival)
            self.counts["completed_requests"] += 1

    def _complete(self, action, time):
        self._time = time
        super()._complete(action, time)
        self._record_completion(self._member_by_component[action.component], time)

    def _execution_events(self, time):
        self._time = time
        return super()._execution_events(time)

    def _finish_request(self, cand, time):
        batch = cand.request
        for member in batch.members:
            self._record_completion(member, time)
        dependencies = cand._dag_read_dependencies & set(self.active)
        if dependencies:
            cand.waits_on.update(dependencies)
            if cand not in self.pending_noops:
                self.pending_noops.append(cand)
                self.counts["empty_waiting"] += 1
            return
        if any(r.finished is None and not r.lost for r in batch.members):
            cand.status = CANCELLED
            self.counts["precondition_failures"] += 1
            self._release_candidate(cand)
            self._recompute(batch, time, "precondition")
        else:
            batch.finished = time

    def _recompute(self, request, time, cause):
        self._time = time
        members = request.members if isinstance(request, BatchRequest) else (request,)
        # Reverse appendleft keeps the members' prior queue order.
        for member in reversed(members):
            self._record_completion(member, time)
            if member.finished is not None or member.lost:
                continue
            member.attempt += 1
            if member.recomputations >= self.max_recomputations:
                self.counts["abandoned"] += 1
                self.counts["abandoned_after_" + cause] += 1
                member.outcome_reason = "recomputation_limit_" + cause
                self._lose(member)
                continue
            member.recomputations += 1
            self.counts["recomputed"] += 1
            self.counts["recomputed_after_" + cause] += 1
            if any(r.id == member.id for r in self.queue):
                raise AssertionError("duplicate queued recomputation")
            self.queue.appendleft(member)

    def _lose(self, request):
        members = request.members if isinstance(request, BatchRequest) else (request,)
        for member in members:
            self._record_completion(member, self._time)
            if member.finished is not None or member.lost:
                continue
            reason = getattr(member, "outcome_reason", None)
            if reason is None:
                reason = (
                    "admission_no_active_release" if not self.active else "admission_recheck_limit"
                )
            member.outcome_reason = reason
            ContractSimulator._lose(self, member)
            self.terminal_outcomes[member.id] = dict(
                request=member.id, time=self._time, exit=reason, attempt=member.attempt
            )

    def metrics(self):
        result = super().metrics()
        result.update(
            batches=len(self.batches),
            window=self.window,
            coordination=self.coordination,
            first_planner_wait_mean=(
                sum(self.planner_waits) / len(self.planner_waits) if self.planner_waits else 0.0
            ),
            deployment_mean=result["response_mean"],
        )
        return result


# Exact timing optimizations


class DictionarySnapshot(Snapshot):
    """Snapshot whose flat dictionaries are copied without recursion."""

    def __deepcopy__(self, memo):
        result = copy(self)
        memo[id(self)] = result
        flat = {
            "view",
            "lifecycle",
            "target",
            "free",
            "versions",
            "network_options",
            "network_loads",
            "network_link_capacities",
            "network_edge_demands",
        }
        for name, value in self.__dict__.items():
            if name in flat:
                setattr(result, name, dict(value))
            elif name == "network_application_loads":
                setattr(result, name, {key: dict(loads) for key, loads in value.items()})
            else:
                setattr(result, name, deepcopy(value, memo))
        return result


class ExactTimingSimulator(BatchedNetworkDagSimulator):
    """Batched simulator with cached host loads and link envelopes; same results, faster."""

    def __init__(self, *args, **kwargs):
        self._envelope_cache = {}
        self._application_envelope_cache = {}
        super().__init__(*args, **kwargs)

    def loads(self):
        result = dict.fromkeys(self.hosts, 0)
        for component, host in self.m_act.items():
            if host is not None:
                result[host] += self.dem[component]
        for _, host, amount, _ in self.ledger.values():
            if host is not None:
                result[host] += amount
        return result

    def snapshot(self, time):
        result = super().snapshot(time)
        result.__class__ = DictionarySnapshot
        return result

    def network_options(self):
        # Host identifiers are immutable and option sets are never mutated.
        singleton = {h: frozenset((h,)) for h in self.hosts}
        singleton[None] = frozenset()
        options = {c: singleton[h] for c, h in self.m_act.items()}
        for action in self.active.values():
            if action.destination is not None:
                options[action.component] = (
                    options[action.component] | singleton[action.destination]
                )
        for component, host, _, leaked in self.ledger.values():
            if leaked and host is not None:
                options[component] = options[component] | singleton[host]
        return options

    def network_loads(self, actual_only=False):
        if actual_only:
            singleton = {h: frozenset((h,)) for h in self.hosts}
            singleton[None] = frozenset()
            options = {c: singleton[h] for c, h in self.m_act.items()}
        else:
            options = self.network_options()
        identity = tuple(options.items())
        cached = self._envelope_cache.get(actual_only)
        if cached is not None and cached[0] == identity:
            return dict(cached[1])
        total = Counter()
        for app, spec in self.applications.items():
            endpoints = tuple(options[c] for c in spec["components"])
            key = (actual_only, app)
            previous = self._application_envelope_cache.get(key)
            if previous is None or previous[0] != endpoints:
                loads = network_application_loads({app: spec}, self.edge_demands, options)[app]
                self._application_envelope_cache[key] = (endpoints, loads)
            else:
                loads = previous[1]
            total.update(loads)
        result = dict(total)
        self._envelope_cache[actual_only] = (identity, result)
        return dict(result)


# Mechanism ablations

ABLATIONS = ("full", "no_aggregate", "no_fresh", "no_compensation")


class BatchAblationSimulator(BatchedNetworkDagSimulator):
    """Batched simulator with one admission mechanism disabled."""

    def __init__(self, *args, ablation="full", **kwargs):
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation: {ablation}")
        self.ablation = ablation
        super().__init__(*args, **kwargs)
        self.check_fresh = ablation != "no_fresh"
        self.compensate = ablation != "no_compensation"
        # Turning this switch off would also remove lifecycle checks.
        self.check_aggregate = True

    def _invariant(self, cand, actions):
        """Without the aggregate check: action pre-states and application constraints only."""
        if self.ablation != "no_aggregate":
            return super()._invariant(cand, actions)
        feasible = True
        for action in actions:
            self.counts["comparisons_aggregate"] += 1
            if self.pre_state_of(action, cand) != action.pre_state:
                feasible = False
        self.counts["comparisons_aggregate"] += sum(
            len(self.applications[a]["edges"]) for a in cand.request.application
        )
        valid = self._network_holds(cand.request.application, cand.mapping)
        if not valid:
            self.counts["dag_admission_rejections"] += 1
        return feasible and valid

    def _recompute(self, request, time, cause):
        if self.ablation == "no_compensation" and cause in ("failure", "precondition"):
            self._time = time
            self._lose(request)
            return
        return super()._recompute(request, time, cause)


# Workload generation

WORKLOAD = dict(
    hosts=100,
    host_capacities=(80, 90, 100, 110, 120),
    link_bandwidths=(10, 20, 30, 40),
    link_latency=(1, 8),
    applications=200,
    components_per_application=4,
    component_demand=(5, 15),
    edge_bandwidth=(5, 15),
    edge_latency_bound=(2, 6),
    planning=(20, 40),
    action_duration=(20, 60),
    arrival_gap=dict(frequent=(10, 30), sparse=(100, 300)),
    extra_edge_probability=0.35,
    search_budget=10000,
)


def make_network_dag_plan(application_id, applications, demands, distances, search_budget=10000):
    """Bounded backtracking placement of one application's missing components.

    Hosts are tried by increasing latency to the neighbours already placed, then by
    decreasing free capacity; ties are broken with the generator given to `plan`. The
    returned `plan(snapshot, scope, attempt, rng)` maps the scope or raises
    PlacementUnavailable with the reason.
    """
    if isinstance(search_budget, bool) or not isinstance(search_budget, int) or search_budget < 1:
        raise ValueError("search_budget must be a positive integer")
    application = applications[application_id]
    members = tuple(application["components"])
    neighbours = {component: [] for component in members}
    for source, target, limit in application["edges"]:
        neighbours[source].append((target, limit))
        neighbours[target].append((source, limit))
    distinct = application.get("distinct_hosts", False)

    def plan(snapshot, scope, attempt, rng):
        scope = tuple(scope)
        if set(scope) != set(members):
            raise ValueError("deployment scope must contain the whole application")
        current = {component: snapshot.view.get(component) for component in members}
        missing = [component for component in members if current[component] is None]
        placement = {c: host for c, host in current.items() if host is not None}
        free = dict(snapshot.free)
        if any(host not in free for host in placement.values()) or (
            distinct and len(set(placement.values())) != len(placement)
        ):
            raise PlacementUnavailable("infeasible")
        if any(
            source in placement
            and target in placement
            and distances[placement[source]][placement[target]] > limit
            for source, target, limit in application["edges"]
        ):
            raise PlacementUnavailable("infeasible")
        host_ties = {host: rng.random() for host in sorted(free)}
        component_ties = {component: rng.random() for component in members}
        visited, exhausted = 0, False

        def available(component):
            occupied = set(placement.values()) if distinct else set()
            hosts = [
                host
                for host in sorted(free)
                if free[host] >= demands[component]
                and host not in occupied
                and all(
                    other not in placement or distances[host][placement[other]] <= limit
                    for other, limit in neighbours[component]
                )
            ]
            hosts.sort(
                key=lambda host: (
                    sum(
                        distances[host][placement[other]]
                        for other, _ in neighbours[component]
                        if other in placement
                    ),
                    -free[host],
                    host_ties[host],
                )
            )
            return hosts

        def search(remaining):
            nonlocal visited, exhausted
            if not remaining:
                return candidate_network_feasible(snapshot, placement, applications)
            choices = [(component, available(component)) for component in remaining]
            component, hosts = min(
                choices, key=lambda item: (len(item[1]), -demands[item[0]], component_ties[item[0]])
            )
            later = tuple(other for other in remaining if other != component)
            for host in hosts:
                if visited >= search_budget:
                    exhausted = True
                    return False
                visited += 1
                placement[component] = host
                free[host] -= demands[component]
                if candidate_network_feasible(snapshot, placement, applications) and search(later):
                    return True
                free[host] += demands[component]
                del placement[component]
                if exhausted:
                    return False
            return False

        if not search(tuple(missing)):
            raise PlacementUnavailable("search_exhausted" if exhausted else "infeasible")
        return {component: placement[component] for component in scope}, set()

    return plan


def make_workload(seed, arrival_mode="frequent"):
    """Infrastructure, 200 four-component applications and their deployment requests.

    One generator seeded with `seed` draws everything in a fixed order, so a replication
    has the same workload under every controller. The arrival mode only scales the
    inter-arrival gaps: the same uniform draw gives the frequent and the sparse arrivals.
    """
    if arrival_mode not in WORKLOAD["arrival_gap"]:
        raise ValueError("arrival_mode must be frequent or sparse")
    rng = random.Random(seed)
    hosts = {
        f"h{index:02d}": rng.choice(WORKLOAD["host_capacities"])
        for index in range(WORKLOAD["hosts"])
    }
    names = sorted(hosts)
    distances = {host: {host: 0} for host in hosts}
    link_capacities = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            latency = rng.randint(*WORKLOAD["link_latency"])
            distances[left][right] = distances[right][left] = latency
            link_capacities[(left, right)] = rng.choice(WORKLOAD["link_bandwidths"])
    applications, components, edge_demands = {}, {}, {}
    for index in range(WORKLOAD["applications"]):
        app = f"app{index:03d}"
        members = tuple(f"{app}.c{item}" for item in range(WORKLOAD["components_per_application"]))
        for component in members:
            components[component] = dict(
                host=None,
                application=app,
                demand=rng.randint(*WORKLOAD["component_demand"]),
                duration=rng.randint(*WORKLOAD["action_duration"]),
            )
        # A random DAG: a tree over a shuffled order, plus optional forward edges.
        order = list(members)
        rng.shuffle(order)
        pairs = {
            (rng.choice(order[:position]), order[position]) for position in range(1, len(order))
        }
        for left in range(len(order)):
            for right in range(left + 1, len(order)):
                pair = (order[left], order[right])
                if pair not in pairs and rng.random() < WORKLOAD["extra_edge_probability"]:
                    pairs.add(pair)
        edges = []
        for left, right in sorted(pairs):
            edges.append((left, right, rng.randint(*WORKLOAD["edge_latency_bound"])))
            edge_demands[(app, left, right)] = rng.randint(*WORKLOAD["edge_bandwidth"])
        applications[app] = dict(components=members, edges=tuple(edges), distinct_hosts=True)
    demands = {component: spec["demand"] for component, spec in components.items()}
    low, high = WORKLOAD["arrival_gap"][arrival_mode]
    requests, arrival = [], 0
    for request_id, (app, application) in enumerate(applications.items()):
        arrival += low + int((high - low + 1) * rng.random())
        scope = application["components"]
        request = DagRequest(
            id=request_id,
            arrival=arrival,
            scope=scope,
            planning=rng.randint(*WORKLOAD["planning"]),
            plan=make_network_dag_plan(
                app, applications, demands, distances, WORKLOAD["search_budget"]
            ),
            application=app,
            operation="deploy",
            read_scope=scope,
        )
        request.durations = {component: components[component]["duration"] for component in scope}
        requests.append(request)
    return dict(
        hosts=hosts,
        components=components,
        requests=requests,
        applications=applications,
        distances=distances,
        link_capacities=link_capacities,
        edge_demands=edge_demands,
    )


# Failures and completed-action observations


def failure_model(probability):
    """Each action fails with the given probability, at a uniform instant before completion."""

    def failure(action, request, attempt, rng):
        if rng.random() < probability:
            return rng.randint(1, request.durations[action.component])
        return None

    return failure


SAFETY_METRICS = (
    "capacity_violations",
    "stale_admissions",
    "leaked_reservations",
    "duplicate_placements",
    "invalid_transitions",
    "dag_latency_violations",
    "final_dag_violations",
    "network_capacity_violations",
    "network_reservation_violations",
    "distinct_host_violations",
    "executor_capacity_violations",
)


def completed_action_observations(sim):
    """Observe completed actions only; never infer migration from a plan target."""
    actions = [
        action for candidate in sim.candidates for action in candidate.actions if action.done
    ]
    assert len({action.id for action in actions}) == len(actions)
    deployed = {action.component for action in actions if action.kind == "deploy"}
    migrated = {action.component for action in actions if action.kind == "migrate"}
    records = [
        dict(
            id=action.id,
            candidate=action.candidate,
            request=sim.candidates[action.candidate].request.id,
            component=action.component,
            kind=action.kind,
            started=action.start,
            finished=action.finish,
        )
        for action in sorted(actions, key=lambda action: action.id)
    ]
    metrics = dict(
        successful_deploy_actions=sum(action.kind == "deploy" for action in actions),
        completed_migration_actions=sum(action.kind == "migrate" for action in actions),
        components_ever_deployed=len(deployed),
        distinct_migrated_components=len(migrated),
        component_migration_rate_percent=100 * len(migrated) / len(deployed) if deployed else None,
    )
    return metrics, records
