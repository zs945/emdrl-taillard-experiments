"""Verify the case identities, scheduling records and descriptive metrics using Python 3."""
import csv
import hashlib
import itertools
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
ARMS = ("t4", "m4", "t1")
SEEDS = range(1, 31)
SEGMENTS = {"selected_16": range(16), "random_14": range(16, 30), "combined_30": range(30)}
FIELDS = ("mean_reward", "reward_sd", "mean_makespan", "makespan_sd", "makespan_range")


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(inputs, rows):
    data = {(int(r["case_id"]),r["arm"],int(r["seed"])): int(r["makespan"]) for r in rows}
    assert len(rows) == len(data) == 2700 and set(data) == set(itertools.product(range(30), ARMS, SEEDS))
    lb, cases, metrics = {}, [], []
    for c in inputs:
        i = c["case_id"]
        s = c["spec"]
        loads = defaultdict(int)
        for machines, times in zip(s["routes"], s["durations"]):
            for m,t in zip(machines,times):
                loads[m] += t
        lb[i] = max(max(loads.values()), max(map(sum, s["durations"])))
        t,a,o = [st.mean(data[i,k,z] for z in SEEDS) for k in ARMS]
        cases.append(dict(case_id=i, source_instance=c["source_instance"], selection=c["selection"],
                          initial_lower_bound=lb[i], T4_mean=t, ANN_mean=a, T1_mean=o,
                          T4_beats_both=int(t < min(a,o))))
    for g, ids in SEGMENTS.items():
        for arm in ARMS:
            seed_metrics = []
            for seed in SEEDS:
                ms = [data[i,arm,seed] for i in ids]
                rewards = [1-data[i,arm,seed]/lb[i] for i in ids]
                seed_metrics.append((st.mean(rewards),st.stdev(rewards),st.mean(ms),st.stdev(ms),max(ms)-min(ms)))
            metrics.append(dict(group=g,arm=arm,cases=len(ids),seeds=30,
                                **dict(zip(FIELDS,map(st.mean,zip(*seed_metrics))))))
    return cases,metrics


def check():
    for line in (BASE/"SHA256SUMS").read_text("utf8").splitlines():
        h,name=line.split("  ",1)
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==h,name
    cases=json.loads((BASE/"inputs.json").read_bytes())["cases"]
    assert [c["case_id"] for c in cases]==list(range(30)) and len({c["source_instance"] for c in cases})==30
    original=json.loads((BASE.parent/"ten-seeds/inputs.json").read_bytes())
    old={o["key"]:o for o in original["test"]}
    rng=random.Random(202610070701)
    series={}
    for i in range(1,192):
        routes,durations=[],[]
        for _ in range(15):
            route=list(range(1,16))
            rng.shuffle(route)
            routes.append(route)
            durations.append([rng.randint(1,99) for _ in route])
        if i>105:
            series[f"R15_test_{i-105:03}"]=dict(jobs=15,machines=15,routes=routes,durations=durations)
    for i,c in enumerate(cases):
        n=c["source_instance"]
        assert c["spec"]==series[n]
        assert c["spec_sha256"]==hashlib.sha256(json.dumps(c["spec"],sort_keys=True,separators=(",",":")).encode()).hexdigest()
        if i<16: assert c["selection"]=="outcome_selected" and old[n]["spec_sha256"]==c["spec_sha256"]
        else: assert c["selection"]=="fixed_random" and n==f"R15_test_{73+i-16:03}"
    rows=read_table(BASE/"results/observations.csv")
    for r in rows:
        c=cases[int(r["case_id"])]
        assert all(r[k]==c[k] for k in ("source_instance","selection","spec_sha256"))
        assert len(bytes.fromhex(r["schedule_sha256"]))==32
    summaries, metrics=summarize(cases,rows)
    assert all(c["T4_beats_both"] for c in summaries[:16])
    for expected, name in [(summaries,"case_means.csv"),(metrics,"metrics.csv")]:
        got=read_table(BASE/"results"/name)
        assert len(got)==len(expected)
        for a,b in zip(expected,got):
            assert set(a)==set(b)
            for k,v in a.items():
                if isinstance(v,(int,float)): assert math.isclose(v,float(b[k]),abs_tol=1e-9),k
                else: assert b[k]==v,k
    print("Verified cases 0-29, two inclusion rules, 2,700 schedules and all five metrics.")


if __name__=="__main__":
    check()
