# MACPE-K

Simulation code, recorded results, and verification scripts for MACPE-K, a MAPE-K variant in which Planning and Execution instances overlap under conflict-aware admission. The code compares MACPE-K with a sequential MAPE-K baseline on a service-placement workload and checks the simulated timings against a discrete-time admission-cycle model.

## Code

| File | Role |
| --- | --- |
| `simulation.py` | Knowledge-base state, admission checks, resource reservations, recovery, and DAG placement workloads |
| `theory.py` | Timing recurrences and numerical expectations of waiting and deployment times |
| `experiments.py` | Inputs, seeds, campaigns A/B/C, and completion observations |
| `report.py` | Figures, confidence intervals, and tables |
| `verify.py` | Consistency checks of the recorded results, behavioral tests, and selected replays |

`data/` contains the recorded campaigns and `reference/` the figures and tables built from them. Per-run records are gzip-compressed JSON (`runs.json.gz`, `runs.jsonl.gz`); summaries and protocols are plain JSON.

## Campaigns

| Campaign | Question | Data | Outputs |
| --- | --- | --- | --- |
| A, timing | Waiting and deployment times with fixed and variable Planning and Execution durations, under event-driven and window-based control, against the admission-cycle model | `data/timing/` | `s1_metrics.pdf`, `s2_conf.pdf` |
| B, capacity and coordination cost | Deployment time and completion under scaled host and link capacities, and under an additional admission-check delay | `data/costs/` | `conflicts_coordination.pdf` |
| C, placement | Admission and recovery under base capacities and action failures, with mechanism ablations and four Planning/action-duration profiles | `data/placement/` | `placement_performance_table.tex`, `ablation_table.tex` |

Times are model slots, not measured CPU times. Deployment-time means are conditional on success and must be read alongside completion counts. In campaign A, sufficient Planning capacity means ceil(T_P / Pi) planners for periodic event-driven arrivals, ceil(T_P / w) planners under window-based control, and one planner per request for Poisson arrivals. Besides the `macpe` and `sequential` policies used in the campaigns, `simulation.py` accepts `blocking` and `throttled` planner-hold rules, which are not evaluated.

Each stochastic configuration is replicated 30 times, using seeds 1 to 30. Controllers share the same workload within each replication. Planner tie-breaks and action failures use generators keyed by seed, request, and attempt, so corresponding requests and attempts share the same random draws across controllers. Summary files are rounded to two decimals; per-run records keep full precision.

## Run

Use Python 3.11. Run commands from this directory.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py all
python theory.py --self-check
python report.py all
```

Outputs go to `reproduction/`. The checks cover 9,720 timing runs, 555 capacity/cost runs, and 900 placement/ablation/sensitivity records, representing 750 distinct runs. `requirements.txt` pins the versions used for verification; the checks also pass with Python 3.13, NumPy 2.5, SciPy 1.18, and Matplotlib 3.11, in about three minutes.

To rerun the complete campaigns and regenerate outputs:

```sh
python experiments.py full --workers 2
python report.py all --data reproduction/full
```

Individual campaigns are `timing`, `costs`, `placement`, `ablations`, and `sensitivity`; use `--help` for options. Each run is saved as a checkpoint, and `--resume` continues an interrupted campaign with unchanged inputs. The protocols record execution times for each campaign. `python theory.py` rebuilds `data/timing/theory.json` from the recurrences alone in about a minute, using up to 2 GB of memory.

## Duration sensitivity

The four profiles combine Planning durations of U[2,4] or U[20,40] with component-action durations of U[2,6] or U[20,60]. Arrival gaps remain U[10,30]. Each profile compares sequential control and MACPE-K with one, two or four planners over seeds 1--30, keeping infrastructure, applications, demands, and arrivals identical within a seed. The long/long profile reproduces the original frequent-arrival configuration.

The table reports the first Planning wait over all requests, the deployment time over completed requests, the completion count, and the paired response-time saving over requests completed by both controllers within a replication. Intervals are computed across replications; the sparse-arrival rows come from the main placement campaign.

```sh
python experiments.py sensitivity --workers 2
python report.py tables --data reproduction/full
```

## Checks

`verify.py all` recomputes the statistics of every recorded run from the stored records, replays selected runs and compares them with the records, and runs the behavioral tests. `theory.py --self-check` compares the recurrences with exhaustive enumerations of small cases. The checks do not rerun the complete campaigns; the `protocol.json` files record each campaign as executed. Migration, successive revisions, and revision generations have separate behavioral tests.
