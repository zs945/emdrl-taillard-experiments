"""Regenerate the deterministic schedules of the reported round-50 model."""
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import torch
from model import EMDRL
from runtime_env import Shop, check_schedule

ROOT = Path(__file__).resolve().parent


def parse_instance(raw):
    rows = [list(map(int, line.split())) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    jobs, machines = rows[0]
    assert len(rows) == jobs + 1 and all(len(row) == 2 * machines for row in rows[1:])
    routes = [row[::2] for row in rows[1:]]
    if any(0 in row for row in routes):
        routes = [[m + 1 for m in row] for row in routes]
    return dict(jobs=jobs, machines=machines, routes=routes, durations=[row[1::2] for row in rows[1:]])


def evaluate(policy, spec):
    shop = Shop(spec)
    with torch.no_grad():
        while True:
            state = shop.state()
            scores, _ = policy(state)
            choices = [int(torch.argmax(scores[group["indices"]])) for group in state["groups"]]
            _, done, _ = shop.advance(state, choices)
            if done:
                break
    return check_schedule(spec, shop.schedule), shop.schedule


def main():
    torch.set_num_threads(1)
    net = EMDRL().eval()
    net.load_state_dict(torch.load(ROOT / "weights/update-050.pt", map_location="cpu", weights_only=True), strict=True)
    tasks = json.loads((ROOT / "inputs/benchmark.json").read_text("utf8"))
    with (ROOT / "results/round50_A2.csv").open(encoding="utf-8-sig", newline="") as f:
        results = {r["instance"]: r for r in csv.DictReader(f)}
    with (ROOT / "results/round50_validation14.csv").open(encoding="utf-8-sig", newline="") as f:
        results.update({r["instance"]: r for r in csv.DictReader(f)})
    assert len(tasks) == len(results) == 70
    with ZipFile(ROOT / "inputs/instances.zip") as archive:
        for case in tasks:
            name = case["instance"]
            raw = archive.read(case["path"])
            assert hashlib.sha256(raw).hexdigest() == case["sha256"], name
            ms, schedule = evaluate(net, parse_instance(raw))
            row = results[name]
            assert ms == int(row["EMDRL_1"]), (name, ms, row["EMDRL_1"])
            if case["set"] == "A2":
                sha = hashlib.sha256(json.dumps(sorted(schedule, key=lambda op: op["node"]), sort_keys=True).encode()).hexdigest()
                assert sha == row["schedule_sha256"], name
            print(name, ms, flush=True)
    print("Matched all 70 recorded makespans and all 56 A2 schedule hashes.")


if __name__ == "__main__":
    main()
