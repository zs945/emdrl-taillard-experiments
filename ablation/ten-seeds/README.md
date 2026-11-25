# Ten-seed comparison of the conflict-GAT policies

This experiment addresses the paired comparison of the matched ANN, SNN with one internal step and SNN with four internal steps on the paper's 30 shared random 15 × 15 test instances. It is separate from the Taillard fine-tuning experiment in the repository root.

Each configuration uses the same ten training seed IDs (1–10). All 30 policies are fitted on the same RTX PRO 6000 with the same training pool, seed-specific sequence of training instances, PPO settings and fixed final update (256). Each fit uses 1,024 training episodes. Model optimization runs on CUDA and rollout and deterministic evaluation use the CPU. A spiking policy resets its neuronal state between decisions. Its internal steps integrate the representation of the current decision.

The [analysis plan](protocol.json) defines two comparisons, T4 minus the matched ANN and T4 minus T1. A negative difference favors T4. Each cell is one deterministic makespan for a seed and an instance. A 95% percentile interval uses 50,000 crossed bootstrap draws of seeds and instances, preserving the method pairing. The two-sided sign-flip test enumerates 1,024 assignments of signs to the ten paired seed means, conditional on these 30 instances. It assumes sign exchangeability of the seed differences under the null. The two p values receive the Holm correction. The bootstrap intervals are marginal 95% intervals for each contrast.

[Historical observations](historical_observations.csv) are from the paper's original three seeds. The nine archived checkpoints were replayed on the same test instances. All 270 makespans and complete schedule hashes matched the original evaluation records. For the expanded experiment, seeds 1–3 are retrained alongside seven new seed IDs on the same backend. The historical rows are reported separately.

[Inputs](inputs.json) give the 100 training instances and 30 evaluation instances, including their original identifiers, routes, durations and specification hashes. [Analysis code](analyse.py) produces both planned tests from the completed seed-by-instance matrix.
