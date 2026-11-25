# OreSat 0 satellite assembly after 50 fine-tuning updates

The experiment starts from the [shared Taillard update-50 weights](../../weights/update-050.pt) and trains the same EMDRL architecture on simulated OreSat 0 assembly. The reported satellite checkpoint is the end of 50 additional PPO updates. Each realization schedules ten satellites on eight machines. Makespans are measured in minutes.

## Training and evaluation

Seed 3 and a fresh Adam optimizer were used for satellite fine-tuning. Each update sampled 32 distinct assembly realizations from a fixed pool of 100, for 1,600 training trajectories. The learning rate was 0.001, with 10 PPO epochs per update, minibatches of 64, a clipping threshold of 0.2 and an entropy coefficient of 0.01. The 24 validation cases were evaluated after each update. The endpoint was fixed in advance. The four policies are compared on the same 40 processing-time realizations previously used for the manuscript's assembly evaluation. These 40 realizations were excluded from training and checkpoint selection in this run. Each model produced one deterministic schedule per realization.

## Results

| Policy | CSV column | Mean makespan | Sample SD | Minimum | Manuscript Fig. 15 case (trial 18) |
| --- | --- | ---: | ---: | ---: | ---: |
| Shared update 50 before satellite fine-tuning | `pretrained` | 418.70 | 13.22 | 396 | 428 |
| Shared update 50 followed by 50 satellite updates | `satellite50` | 360.80 | 12.54 | 336 | 344 |
| Earlier satellite fine-tuning from the same shared weights | `satellite96` | 360.95 | 12.66 | 337 | 337 |
| Satellite policy used in the manuscript | `paper` | 360.45 | 12.22 | 336 | 336 |

The fixed-endpoint policy shortened the makespan on all 40 realizations relative to the starting weights. Its mean fell by 57.90 minutes (13.83%). Compared with the manuscript policy, it finished sooner on 15 realizations, tied on eight and finished later on 17. Its 336-minute schedule occurred on trial 19 (duration seed 230020019). The manuscript's 336-minute example occurred on trial 18 (duration seed 230020018).

The earlier `satellite96` comparison used the assembly-specific training procedure, with demonstrations and 96 PPO updates per seed. Its checkpoint was selected on the 24 validation cases. Trials in the results table are paired by the duration seed.

## Files

- [Training protocol and case seeds](protocol.json), [initial validation results](training/initial_validation.json) and [fixed endpoint](fixed_endpoint.json).
- [Training update summaries](training/updates.csv) and [all 1,600 sampled trajectories](training/trajectories.csv). The [50 original round records](training/rounds.zip) include individual validation makespans and checkpoint hashes.
- [Final satellite weights](weights/satellite-update-050.pt), stored as a PyTorch state dictionary, and [starting network definition](../../model.py).
- [40 paired makespans](results/paired_40.csv), [dispatch schedules of the fixed-endpoint policy](results/schedules_40.csv) and [aggregate results](results/summary.json). Each schedule has 190 nodes. Blank resource entries identify nodes that do not occupy a machine.

Run `python oresat/fixed50/verify.py` from the repository root to recompute the paired summaries and verify the training records, weights, duration hashes and schedule hashes.
