# EMDRL training records and Taillard benchmark evaluation

This repository archives the single shared EMDRL policy used in the fixed-update experiment. The model starts from seed 3 of the paper's full spiking ablation model (*T* = 4), pretrained on random 15 × 15 instances. One set of weights serves all seven problem sizes. The reported checkpoint is the end of PPO fine-tuning update 50. At each dispatch event, the policy chooses the highest-scoring eligible operation for each idle machine. Each reported makespan comes from one deterministic schedule.

## Training and model selection

The starting policy had completed 256 updates on 1,024 training episodes. Fine-tuning used a fixed pool of 100 generated instances at each of the seven sizes. Each update sampled 32 distinct instances per size, giving 224 trajectories per update and 11,200 in total. Only the generated training instances contributed gradients. The learning rate was 0.001. The remaining PPO settings and software versions are in [the run configuration](training/config.json).

The earlier manuscript procedure selected a separate checkpoint for each size using the two designated Taillard instances at that size. During development of the shared policy, the same 14 instances informed the choice of learning rate and initialization seed. At a given update, learning rates were compared across three seeds using the worst and then mean ratio of makespan to the previously fixed reference. The seed was then selected within the chosen rate. A2 results were also inspected during development. For the final continuation, update 50 was fixed before the last four updates were run. The saved training state after update 46 included the optimizer and random-number states. The final evaluation used one frozen checkpoint for all instances.

## Final deterministic results

| Set | Instances | Strictly first | Tied for first | In the top two |
| --- | ---: | ---: | ---: | ---: |
| Model-selection cases, manuscript Table 8 | 14 | 7 | 0 | 13 |
| Additional Taillard cases, manuscript Table A2 | 56 | 19 | 0 | 53 |

[Table 8 results](results/round50_validation14.csv) compare the final policy with ten reported comparator configurations. NR marks an unreported value. [Table A2 results](results/round50_A2.csv) use LRPT, L2D, SLIM-GM, UG-single and UG-comp. A strict first requires a lower makespan than every reported comparator for that instance. Model validation and repeated monitoring should be taken into account when interpreting the benchmark results.

## Satellite assembly and ablation studies

The [OreSat 0 satellite assembly experiment](oresat/fixed50) provides the fixed-endpoint policy, 50 training updates and all 40 paired test schedules. The [30-seed comparison](ablation/thirty-seeds) and [full ablation protocol](ablation/full-re1-30) document the component and simulation-horizon studies.

## Archive contents

- [Per-update records](training/updates.csv) give the number of transitions, Adam steps, validation summaries and original record hashes. [Training trajectories](training/trajectories.csv) identify the sampled instance, makespan and schedule hash for each episode in chronological order.
- [Weight index](weights/index.csv) lists the paper initialization (update 000) and all 50 fine-tuning checkpoints. Each public PT file contains only the named CPU tensors. The tensor digest matches the archived training record. Separate columns identify the exported file and the original training checkpoint, which also contained optimizer and random-number state.
- [Inputs](inputs/instances.zip) contain the 700 generated training instances and the 70 Taillard benchmark instances in job–machine order format. [Benchmark membership](inputs/benchmark.json) records the role and original input hash for every Taillard instance.
- [Model](model.py), [dispatch environment](runtime_env.py) and [replay](replay.py) reproduce the fixed model's deterministic decisions on CPU. The model implementation is an inference implementation of the recorded network. The per-trajectory hashes and archived weights document training. The replay script evaluates trained weights.

## Recheck the records

Use Python with the PyTorch and PyTorch Geometric versions in [requirements.txt](requirements.txt). Run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python verify.py
python replay.py
```

The first script checks the [release checksums](SHA256SUMS), tensor hashes, input hashes, trajectory counts and rankings. The second loads [the update-50 weights](weights/update-050.pt), reconstructs all 70 makespans, and checks the complete schedule hash for each of the 56 A2 cases.
