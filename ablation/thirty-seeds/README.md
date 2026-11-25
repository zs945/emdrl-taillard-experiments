# Thirty-seed comparison of the conflict-GAT policies

This follow-up compares the four-step spiking policy with the matched ANN and the one-step spiking policy. The decision to extend training to 30 seed IDs followed inspection of the [ten-seed results](../ten-seeds/results/contrasts.csv). Seeds 1–10 use the previously published weights and observations. Seeds 11–30 use the same three configurations, training settings and [input instances](../ten-seeds/inputs.json). The full cohort has 90 models. Each receives 256 PPO updates, with four training episodes per update. All final policies use single deterministic inference on the same 30 random 15 × 15 test instances.

The comparisons use the mean paired makespan difference over the 30 training seeds and 30 test instances. A negative difference favors the four-step SNN. Marginal 95% intervals use 50,000 crossed bootstrap resamples of seeds and instances. The two-sided sign-flip tests use all 1,073,741,824 possible sign assignments to the 30 seed-level differences averaged over the fixed test set. The tests assume sign exchangeability under the null. Holm adjustment covers the two contrasts. The pooled results describe the extended cohort and include the original ten seeds.

[Experiment settings](protocol.json) and [statistical analysis](analyse.py).
