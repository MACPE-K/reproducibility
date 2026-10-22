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
| C, placement | Admission and recovery under base capacities and action failures, with mechanism ablations | `data/placement/` | `integrated_table.tex`, `ablation_table.tex` |

Times are model slots, not measured CPU times. Deployment-time means are conditional on success and must be read alongside completion counts. In campaign A, sufficient Planning capacity means ceil(T_P / Pi) planners for periodic event-driven arrivals, ceil(T_P / w) planners under window-based control, and one planner per request for Poisson arrivals. Besides the `macpe` and `sequential` policies used in the campaigns, `simulation.py` accepts `blocking` and `throttled` planner-hold rules, which are not evaluated.

Each stochastic configuration is replicated 30 times, and replication r uses the seed r, from 1 to 30: the workload of a replication is drawn once from that seed and is identical for every controller, while planner tie-breaks and action failures are drawn from generators keyed by the seed, the request and the attempt, so that a request behaves the same under every controller whatever the order of events. Summary files are rounded to two decimals; per-run records keep full precision.

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

Outputs go to `reproduction/`. The checks cover 9,720 timing runs, 555 capacity/cost runs, and 360 placement/ablation records, including 30 shared baseline records, and take about three minutes. `requirements.txt` pins the versions used by the verification workflow; the checks also pass with Python 3.13, NumPy 2.5, SciPy 1.18, and Matplotlib 3.11.

To rerun the complete campaigns and regenerate outputs:

```sh
python experiments.py full --workers 2
python report.py all --data reproduction/full
```

Individual campaigns are `timing`, `costs`, `placement`, and `ablations`; use `--help` for options. Each run is saved as a checkpoint, and `--resume` continues an interrupted campaign with unchanged inputs. With 11 workers, the recorded campaigns took about 5 minutes for B, 8 minutes for C, and 25 minutes for A, about four CPU-hours in total. `python theory.py` rebuilds `data/timing/theory.json` from the recurrences alone in about a minute, using up to 2 GB of memory.

## Checks

`verify.py all` recomputes the statistics of every recorded run from the stored records, replays selected runs and compares them with the records, and runs the behavioral tests. `theory.py --self-check` compares the recurrences with exhaustive enumerations of small cases. The GitHub workflow runs these checks on every push. The checks do not rerun the complete campaigns; the `protocol.json` files record each campaign as executed. Migration, successive revisions, and revision generations have separate behavioral tests.
