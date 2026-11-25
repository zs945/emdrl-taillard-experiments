# Ablation cases 0–29

This folder contains 30 simulated 15 × 15 job-shop cases. The identifiers 0–29 are display IDs. The [case index and means](results/case_means.csv) and the [complete instance definitions](inputs.json) include the original generator ID for each case.

## How the cases were included

| Display IDs | Cases | Inclusion rule |
| --- | ---: | --- |
| 0–15 | 16 | All cases from the original 001–030 evaluation on which T=4 had a strictly lower mean makespan than both comparators across the same 30 training seeds. Ordered by original ID. |
| 16–29 | 14 | The next consecutive random cases, 073–086. The previous supplementary run ended at 072. The list was fixed before these 14 were evaluated, and all 14 were retained. |

Both parts use the original generator (seed 202610070701). Each job visits all 15 machines in a random order and has integer processing times drawn uniformly from 1 through 99. The combined table is a descriptive case series. Outcome-based inclusion for the first 16 affects its averages. The new 14 are reported separately so their performance can be read directly.

## Scheduling results

Each case was evaluated using the four-step conflict-GAT spiking model (T=4), matched conflict-GAT ANN, and one-step conflict-GAT spiking model (T=1). All three configurations used their 30 [archived checkpoints](../thirty-seeds/weights/index.csv). One deterministic schedule per checkpoint and case gives 2,700 observations. The 1,440 observations for IDs 0–15 come from the original runs. The 1,260 observations for IDs 16–29 were computed for this evaluation. Each model reproduced the makespan and schedule hash of an archived control before running the new cases.

| Group | Mean makespan, T=4 | Matched ANN | SNN, T=1 | Cases where T=4 beats both on mean makespan |
| --- | ---: | ---: | ---: | ---: |
| Outcome-selected 16 | 1387.86 | 1401.79 | 1402.30 | 16/16 |
| Fixed random 14 | 1379.44 | 1381.20 | 1383.52 | 7/14 |
| Combined 30 | 1383.93 | 1392.18 | 1393.54 | 23/30 |

The [five metrics](results/metrics.csv) are mean reward, reward SD, mean makespan, makespan SD and makespan range. Each metric is computed across cases for each seed, then averaged across the 30 seeds. SD is the sample standard deviation. The reward for a schedule is one minus its makespan divided by the initial workload lower bound.

[Case IDs and per-case means](results/case_means.csv) · [Input instances](inputs.json) · [All observations and schedule hashes](results/observations.csv) · [Evaluation protocol](protocol.json)

Run `python ablation/random-cases-0-29/verify.py` from the repository root to verify the source IDs, input hashes and summary tables.
