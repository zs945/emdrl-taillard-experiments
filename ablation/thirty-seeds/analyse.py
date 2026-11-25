"""Analyse the two prespecified makespan contrasts using the original paired design."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def load_runs(folder, plan):
    rows = []
    identifiers = {}
    initial = {}
    sampled = {}
    for seed in plan["seeds"]:
        for arm in plan["arms"]:
            path = folder / f"{arm}_seed{seed}"
            receipt = read_json(path / "evaluation.json")
            assert (receipt["arm"], receipt["seed"], receipt["update"], receipt["episodes"]) == (arm, seed, 256, 1024)
            assert hashlib.sha256((path / "final.pt").read_bytes()).hexdigest() == receipt["file_sha256"]
            assert receipt["initial"] == initial.setdefault(seed, receipt["initial"])
            log = read_json(path / "training.json")
            assert [u["update"] for u in log] == list(range(1,257))
            sequence = [e["instance"] for u in log for e in u["episodes"]]
            assert len(sequence) == 1024 and sequence == sampled.setdefault(seed, sequence)
            observed = receipt["observations"]
            assert len(observed) == len({r["instance"] for r in observed}) == 30
            for r in observed:
                assert r["spec_sha256"] == identifiers.setdefault(r["instance"], r["spec_sha256"])
                rows.append(dict(seed=seed, arm=arm, **r))
    assert len(identifiers) == 30 and len(set(initial.values())) == len(plan["seeds"])
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_paired_p(seed_totals):
    """Count the two-sided exact sign-flip tail using integer half-sums."""
    def signed_sums(values):
        sums = np.array([0], dtype=np.int64)
        for value in values:
            sums = np.concatenate((sums - value, sums + value))
        return sums

    cutoff = abs(int(np.sum(seed_totals)))
    if cutoff == 0:
        return 1.0
    middle = len(seed_totals) // 2
    left = signed_sums(seed_totals[:middle])
    right = np.sort(signed_sums(seed_totals[middle:]))
    # For a positive cutoff the two tails are disjoint. Include exact ties.
    low = np.searchsorted(right, -cutoff - left, side="right")
    high = len(right) - np.searchsorted(right, cutoff - left, side="left")
    return int(np.sum(low + high)) / (len(left) * len(right))


def analyse(rows, plan):
    arms, seeds = plan["arms"], plan["seeds"]
    cases = sorted({r["instance"] for r in rows})
    values = {(r["arm"], int(r["seed"]), r["instance"]): int(r["makespan"]) for r in rows}
    assert len(cases) == 30 and len(rows) == len(values) == len(seeds) * len(arms) * len(cases)
    matrices = {arm: np.array([[values[arm, s, c] for c in cases] for s in seeds]) for arm in arms}
    assert all(np.all(m > 0) for m in matrices.values())
    settings = plan["statistics"]
    rng = np.random.default_rng(settings["bootstrap_seed"])
    ns, nc = len(seeds), len(cases)
    nboot = settings["resamples"]
    contrasts = settings["contrasts"]
    boot_samples = np.empty((len(contrasts), nboot))
    differences = [matrices[a] - matrices[b] for a, b in contrasts]
    for start in range(0, nboot, 500):
        stop = min(start + 500, nboot)
        b = stop - start
        seed_counts = rng.multinomial(ns, np.full(ns, 1 / ns), size=b)
        case_counts = rng.multinomial(nc, np.full(nc, 1 / nc), size=b)
        for j, difference in enumerate(differences):
            boot_samples[j, start:stop] = np.sum((seed_counts @ difference) * case_counts, axis=1) / (ns * nc)
    summary, paired = [], []
    for j, (a, b) in enumerate(contrasts):
        d = differences[j]
        pval = exact_paired_p(d.sum(axis=1))
        low, high = np.quantile(boot_samples[j], (.025, .975))
        summary.append(dict(first=a, comparator=b, first_mean=float(matrices[a].mean()),
                            comparator_mean=float(matrices[b].mean()), paired_difference=float(d.mean()),
                            reduction_percent=float(-100*d.mean()/matrices[b].mean()),
                            ci_95_low=float(low), ci_95_high=float(high),
                            two_sided_p=pval, holm_p=None))
        for i, seed in enumerate(seeds):
            paired.append(dict(seed=seed, first=a, comparator=b, first_mean=float(matrices[a][i].mean()),
                               comparator_mean=float(matrices[b][i].mean()), paired_difference=float(d[i].mean())))
    ordered = sorted(range(len(summary)), key=lambda j: summary[j]["two_sided_p"])
    floor = 0.0
    for rank, j in enumerate(ordered):
        floor = max(floor, (len(summary)-rank)*summary[j]["two_sided_p"])
        summary[j]["holm_p"] = min(1.0, floor)
    return summary, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--runs", type=Path)
    data.add_argument("--observations", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.protocol)
    if args.runs:
        rows = load_runs(args.runs, plan)
    else:
        with args.observations.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    summary, paired = analyse(rows, plan)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.runs:
        write_csv(args.out / "observations.csv", rows)
    write_csv(args.out / "paired_seed_means.csv", paired)
    write_csv(args.out / "contrasts.csv", summary)
    for r in summary:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
