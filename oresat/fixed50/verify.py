"""Check the OreSat run and its result tables with Python's standard library."""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

BASE = Path(__file__).resolve().parent


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(name):
    with (BASE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def document(name):
    return json.loads((BASE / name).read_bytes())


def check():
    for line in (BASE / "SHA256SUMS").read_text("utf8").splitlines():
        expected, name = line.split("  ", 1)
        assert checksum(BASE / name) == expected, name
    plan = document("protocol.json")
    train, validation, test = map(set, (plan["training_cases"], plan["validation_cases"], plan["test_cases"]))
    assert (len(train), len(validation), len(test), len(train | validation | test)) == (100, 24, 40, 164)
    root = BASE.parents[1]
    assert checksum(root / "weights/update-050.pt") == plan["source_weights"]
    assert checksum(root / "training/config.json") == plan["source_parameters"]
    endpoint = document("fixed_endpoint.json")
    assert endpoint["update"] == plan["satellite_updates"] == 50
    assert endpoint["episodes"] == 1600 and endpoint["seed"] == plan["seed"] == 3
    updates = table("training/updates.csv")
    episodes = table("training/trajectories.csv")
    assert len(updates) == 50 and len(episodes) == 1600
    with ZipFile(BASE / "training/rounds.zip") as archive:
        assert set(archive.namelist()) == {f"round_{i:02}.json" for i in range(1, 51)}
        for i, row in enumerate(updates, 1):
            raw = archive.read(f"round_{i:02}.json")
            run = json.loads(raw)
            assert hashlib.sha256(raw).hexdigest() == row["record_sha256"]
            assert run["update"] == int(row["update"]) == i
            sampled = run["episodes"]
            assert len(sampled) == len({e["seed"] for e in sampled}) == plan["episodes_per_update"] == 32
            assert all(e["seed"] in train for e in sampled)
            assert sum(e["decisions"] for e in sampled) == run["transitions"] == int(row["transitions"])
            assert run["adam_steps"] == int(row["adam_steps"])
            assert run["adam_steps"] == plan["ppo"]["epochs"] * math.ceil(run["transitions"] / plan["ppo"]["minibatch"])
            assert run["weight_sha256"] == row["weight_sha256"]
            cases = run["validation"]["cases"]
            assert {int(k) for k in cases} == validation
            assert math.isclose(statistics.mean(cases.values()), float(row["validation_mean_min"]), abs_tol=1e-9)
            for j, (e, public) in enumerate(zip(sampled, episodes[(i-1)*32:i*32]), 1):
                assert (int(public["update"]), int(public["trajectory"])) == (i, j)
                assert (int(public["seed"]), float(public["makespan_min"]), int(public["decisions"]), public["duration_sha256"]) == (
                    e["seed"], e["makespan"], e["decisions"], e["duration_sha256"])
    assert checksum(BASE / "weights/satellite-update-050.pt") == updates[-1]["weight_sha256"] == endpoint["sha256"]
    rows = table("results/paired_40.csv")
    schedules = defaultdict(list)
    for task in table("results/schedules_40.csv"):
        trial, seed = int(task.pop("trial")), int(task.pop("seed"))
        parsed = {k: float(v) if k in ("start", "end") else int(v) if v else None for k, v in task.items()}
        assert parsed["end"] - parsed["start"] == parsed["duration"]
        schedules[trial,seed].append(parsed)
    assert len(rows) == len(schedules) == 40 and {int(r["seed"]) for r in rows} == test
    for i, row in enumerate(rows, 1):
        assert int(row["trial"]) == i
        tasks = schedules[i,int(row["seed"])]
        assert [r["node"] for r in tasks] == list(range(1,191))
        assert max(r["end"] for r in tasks) == float(row["satellite50"])
        packed = json.dumps(tasks, sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(packed).hexdigest() == row["schedule_sha256"]
        durations = {r["node"]:r["duration"] for r in tasks}
        assert hashlib.sha256(json.dumps(durations, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == row["duration_sha256"]
    summary = document("results/summary.json")
    for name, stats in summary["results"].items():
        values = [float(row[name]) for row in rows]
        assert math.isclose(statistics.mean(values), stats["mean"])
        assert math.isclose(statistics.stdev(values), stats["sd"])
        assert (min(values), max(values), values[17]) == (stats["min"], stats["max"], stats["figure15"])
        assert [i for i,v in enumerate(values, 1) if v == min(values)] == stats["min_trials"]
        differences = Counter("lower" if float(row["satellite50"]) < v else "higher" if float(row["satellite50"]) > v else "equal" for row,v in zip(rows,values))
        assert all(differences[k] == v for k,v in summary["paired"][name].items())
    print("Verified 50 updates, 1,600 training trajectories, final weights and 40 test schedules.")


if __name__ == "__main__":
    check()
