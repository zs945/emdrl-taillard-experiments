"""Verify the published payload, model tensors, PPO records and final rankings."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import torch

ROOT = Path(__file__).resolve().parent


def rows(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    for line in (ROOT / "SHA256SUMS").read_text("utf8").splitlines():
        expected, path = line.split("  ", 1)
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    config = json.loads((ROOT / "training/config.json").read_text("utf8"))
    models = rows("weights/index.csv")
    assert [int(r["update"]) for r in models] == list(range(51))
    for model in models:
        state = torch.load(ROOT / model["file"], map_location="cpu", weights_only=True)
        h = hashlib.sha256()
        for name, tensor in sorted(state.items()):
            h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode())
            h.update(tensor.contiguous().numpy().tobytes())
        assert h.hexdigest() == model["model_state_sha256"], model["file"]
    trajectories = rows("training/trajectories.csv")
    updates = rows("training/updates.csv")
    assert len(trajectories) == 11200 and len(updates) == 50
    train = json.loads((ROOT / "inputs/training_sha256.json").read_text("utf8"))
    benchmark = json.loads((ROOT / "inputs/benchmark.json").read_text("utf8"))
    assert len(train) == 700 and len(benchmark) == len({b["instance"] for b in benchmark}) == 70
    with ZipFile(ROOT / "inputs/instances.zip") as z:
        for path, sha in list(train.items()) + [(b["path"], b["sha256"]) for b in benchmark]:
            assert hashlib.sha256(z.read(path)).hexdigest() == sha, path
    steps = 0
    for idx, u in enumerate(updates, 1):
        traced = [r for r in trajectories if int(r["update"]) == idx]
        assert len(traced) == len({r["input"] for r in traced}) == int(u["trajectories"]) == 224
        assert all(r["input"] in train and int(r["makespan"]) > 0 and len(bytes.fromhex(r["schedule_sha256"])) == 32 for r in traced)
        assert Counter(r["input"].split("/")[1] for r in traced) == dict.fromkeys(config["sizes"], 32)
        steps += int(u["optimizer_steps"])
        assert int(u["update"]) == idx and int(u["cumulative_optimizer_steps"]) == steps
    for name, methods, expected in [
        ("round50_validation14.csv", ["DGERD","L2D","DRL_Liu","PetriRL","SLIM_GM","UG_single","UG_comp","SeEvo_DeepSeek_V3","SeEvo_GPT_4_1_mini","SeEvo_Qwen_Turbo"], (14,7,0,13)),
        ("round50_A2.csv", ["LRPT","L2D","SLIM_GM","UG_single","UG_comp"], (56,19,0,53))]:
        result = rows("results/" + name)
        wins = ties = top2 = 0
        for r in result:
            v = int(r["EMDRL_1"])
            peers = [int(r[k]) for k in methods if r[k] != "NR"]
            rank = 1 + sum(p < v for p in peers)
            assert (v < min(peers), v == min(peers), rank) == (int(r["strict_first"]), int(r["tied_first"]), int(r["rank"]))
            wins += v < min(peers)
            ties += v == min(peers)
            top2 += rank <= 2
        assert (len(result),wins,ties,top2) == expected
    print("Verified 51 model states, 50 updates, 11,200 trajectories, 770 input files and both final ranking tables.")


if __name__ == "__main__":
    main()
