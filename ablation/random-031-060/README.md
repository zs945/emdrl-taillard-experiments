# Random 15 × 15 ablation cases 031–060

This follow-up compares the four-step spiking EMDRL policy (T=4), matched ANN and one-step spiking policy (T=1) on cases R15_test_031 through R15_test_060. Each configuration used its [30 archived seed models](../thirty-seeds/weights/index.csv) at training update 256. Each frozen model produced one deterministic schedule per case, giving 2,700 observations.

## Case selection

The original generator uses an independent machine permutation for each job and integer processing times sampled uniformly from 1 through 99. With generator seed 202610070701, the stream produced 100 training cases, five development cases and the [original 30 test cases](../ten-seeds/inputs.json). This evaluation continues the same stream. An exploratory run evaluated 42 additional cases (031–072) in three consecutive blocks of 14 while looking for examples favorable to T=4. After those results were available, cases 031–060 were taken in generation order for the current display. Every case in that interval is included. T=4 had a lower mean makespan than both comparators on 12 of the 30 cases.

## Results

| Model | Mean reward | Reward SD | Mean makespan | Makespan SD | Makespan range |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMDRL (T=4) | −0.4429 | 0.0930 | 1388.61 | 70.74 | 307.57 |
| Matched ANN | −0.4461 | 0.0963 | 1391.35 | 68.50 | 294.13 |
| SNN (T=1) | −0.4508 | 0.0938 | 1396.12 | 70.43 | 305.47 |

Each metric was calculated across the 30 cases for an individual training seed and then averaged over the 30 seeds. SD is the sample standard deviation across cases. For each case, reward equals one minus the ratio of makespan to the initial lower bound. The bound is the larger of the maximum job workload and maximum machine workload.

| Comparator | T=4 minus comparator, mean makespan | Reduction | 95% paired interval | Two-sided p | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched ANN | −2.73 | 0.20% | [−11.86, 6.14] | 0.4237 | 0.4237 |
| SNN (T=1) | −7.51 | 0.54% | [−16.04, 0.78] | 0.0251 | 0.0502 |

These exploratory contrasts reuse the [original paired analysis](../thirty-seeds/analyse.py). It uses 50,000 crossed bootstrap resamples of training seeds and instances for the intervals and a two-sided sign-flip test on the 30 paired seed means. Holm adjustment covers the two comparisons.

## Data and verification

- [30 case definitions and content hashes](inputs.json) and [mean makespans by case](results/case_means.csv).
- [All 2,700 observations with schedule hashes](results/observations.csv), [five aggregate metrics](results/metrics.csv) and [paired comparisons at full precision](results/contrasts.csv).
- [Original 001–030 evaluation](../thirty-seeds) and [full component and time-step ablation protocol](../full-re1-30).

Run `python ablation/random-031-060/verify.py` from the repository root to verify case membership and recompute the five metrics. To reproduce the paired intervals and tests, run the same analysis on the current observations:

```sh
python ablation/thirty-seeds/analyse.py --observations ablation/random-031-060/results/observations.csv --out reproduced-031-060
```
