# Full first-revision ablation with thirty training seeds

Sections 5.2.1 and 5.2.2 of the first revision specify five component configurations and five internal simulation horizons. The full method E and the four-step horizon use the same policy. This gives nine distinct configurations and 270 models. For each seed all policies use the same ordered sequence of 1,024 training instances sampled from the original pool of 100 across 256 PPO updates. All final weights are evaluated by deterministic dispatch on the same [30 random 15 × 15 instances](../ten-seeds/inputs.json).

| Study | Configurations |
| --- | --- |
| Components | A global MHA + ANN, B global MHA + four-step SNN, C fully connected GAT + four-step SNN, D conflict GAT + ANN, E conflict GAT + four-step SNN |
| Horizons of the conflict GAT + SNN | 1, 4, 8, 16 and 32 |

The [existing D, E and one-step runs](../thirty-seeds) supply 90 archived models. The other six configurations train 30 models each. Seed pairing covers the actor, critic, residual projection and training instances. The initial attention weights are shared within encoder families. The configuration audit checks the graphs, parameter counts and executed neuronal steps against the first revision, including four steps for B and C.

The [protocol](protocol.json) defines eight paired comparisons against E. The [analysis](analyse.py) uses crossed 95% bootstrap intervals and exact two-sided seed-level sign-flip tests, with Holm adjustment separately within the four contrasts in each study. The earlier two planned contrasts retain their original analysis in the thirty-seed release.

The component summary gives seed means and sample standard deviations for each of the five manuscript metrics. Inference time is measured on one CPU thread on five common development states, with five warmups per model/state and 1,500 timed calls per model. Process CPU time over updates 65–256 is recorded during the six additional configurations. Empty CPU-time fields for reused models indicate that these timings were not collected in the earlier CUDA training runs. All current runs use CPU rollout and CUDA optimization.
