# EMDRL network parameters and trained weights

This directory contains the full conflict-aware graph attention and spiking actor–critic model used in the revised experiments. The spiking network runs for 16 internal steps within each dispatch decision.

| Experiment | Trained weights | Use |
| --- | --- | --- |
| Full-model ablation | [Seeds 1–10](ablation) | Each file contains the final model after 256 updates. [Seed 3](ablation/seed-03.pt) initialized the shared policy. |
| Shared policy across seven problem sizes | [Update 45](taillard/update-045.pt) | Selected by the lowest mean makespan on the 14 selection instances over 50 updates. All 50 model states are in [taillard](taillard). |
| OreSat 0 assembly | [Update 50](oresat/update-050.pt) | Initialized from shared update 45. The assembly policy was fine-tuned for 50 further updates and evaluated at its final update. All 50 model states are in [oresat](oresat). |

[Network dimensions, tensor shapes and training settings](configuration.json) · [Deterministic inference implementation](policy.py) · [File and tensor checksums](weights.json)

Each PT file is a CPU state dictionary containing all 15 named tensors. They can be loaded with PyTorch's weights-only loader. For a prepared scheduling state, the network returns node priorities and per-node critic outputs. At dispatch, take the highest-priority candidate within each eligible machine group. The state value is the mean critic output over candidate operations.

From this directory, using the versions in the repository's [requirements](../../requirements.txt):

```python
import torch
from policy import EMDRL

policy = EMDRL().eval()
state_dict = torch.load("taillard/update-045.pt", map_location="cpu", weights_only=True)
policy.load_state_dict(state_dict, strict=True)
```
