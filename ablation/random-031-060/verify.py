"""Verify the generated case sequence and the published case and cohort summaries."""
import csv
import hashlib
import itertools
import json
import math
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("t4", "m4", "t1")
SEEDS = range(1, 31)


def table(name):
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check():
    for line in (ROOT / "SHA256SUMS").read_text("utf8").splitlines():
        h, name = line.split("  ", 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == h, name
    current = json.loads((ROOT / "inputs.json").read_bytes())["test"]
    original = json.loads((ROOT.parent / "ten-seeds/inputs.json").read_bytes())
    assert [c["key"] for c in current] == [f"R15_test_{n:03}" for n in range(31, 61)]
    rng = random.Random(202610070701)
    previous_hashes = set()
    for record in [*original["train"], *([None]*5), *original["test"], *current]:
        routes, durations = [], []
        for _ in range(15):
            m = list(range(1, 16))
            rng.shuffle(m)
            routes.append(m)
            durations.append([rng.randint(1, 99) for _ in m])
        spec = dict(jobs=15, machines=15, routes=routes, durations=durations)
        h = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert h not in previous_hashes
        previous_hashes.add(h)
        if record is not None:
            assert h == record["spec_sha256"] and spec == record["spec"]
    index = {r["key"]: r for r in current}
    rows = table("results/observations.csv")
    data = {(r["instance"], r["arm"], int(r["seed"])): int(r["makespan"]) for r in rows}
    assert len(rows) == len(data) == 2700 and set(data) == set(itertools.product(index, ARMS, SEEDS))
    assert all(r["spec_sha256"] == index[r["instance"]]["spec_sha256"] and len(bytes.fromhex(r["schedule_sha256"])) == 32 for r in rows)
    by_case = {r["instance"]: r for r in table("results/case_means.csv")}
    assert set(by_case) == set(index)
    lb = {}
    for n, case in index.items():
        s = case["spec"]
        loads = [0] * 15
        for route, times in zip(s["routes"], s["durations"]):
            for machine, duration in zip(route, times):
                loads[machine-1] += duration
        lb[n] = max(max(map(sum, s["durations"])), max(loads))
        t, a, o = [st.mean(data[n,k,i] for i in SEEDS) for k in ARMS]
        b = by_case[n]
        for key, val in (("T4_mean",t), ("ANN_mean",a), ("T1_mean",o),
                         ("reduction_vs_ANN_pct",100*(a-t)/a), ("reduction_vs_T1_pct",100*(o-t)/o)):
            assert math.isclose(float(b[key]), val, abs_tol=1e-9)
        assert int(b["beats_both"]) == (t < min(a,o))
        assert int(b["paired_seeds_beating_both"]) == sum(data[n,"t4",i] < min(data[n,"m4",i],data[n,"t1",i]) for i in SEEDS)
    summary = {r["arm"]:r for r in table("results/metrics.csv")}
    for arm in ARMS:
        metrics = []
        for seed in SEEDS:
            ms = [data[n,arm,seed] for n in index]
            rewards = [1-data[n,arm,seed]/lb[n] for n in index]
            metrics.append((st.mean(rewards), st.stdev(rewards), st.mean(ms), st.stdev(ms), max(ms)-min(ms)))
        for key, vals in zip(("mean_reward","reward_sd","mean_makespan","makespan_sd","makespan_range"), zip(*metrics)):
            assert math.isclose(st.mean(vals), float(summary[arm][key]), abs_tol=1e-9)
    print("Verified the 30 consecutive cases, 2,700 observations, case means and five cohort metrics.")


if __name__ == "__main__":
    check()
