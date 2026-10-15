"""Sanity checks on benchmark output, beyond "the script finished".

    python experiments/check_results.py results/dna_benchmark [results/lineage_benchmark ...]

For each results folder:
  * every (axis, value, replicate) in the config must have a row for every
    method; anything missing is listed as a failed dataset
  * metrics must be finite, RF distances within [0, max]
  * search methods are checked on their own criterion: the parsimony
    search should not lose to another method's tree on parsimony score, and
    the ML search should not lose on log-likelihood. These are heuristics, so
    a loss is possible; every loss is printed rather than hidden.

Exits with status 1 if rows are missing or values are invalid.
"""

import json
import math
import os
import sys
from collections import defaultdict

from phylo.benchmark import _jobs, read_csv


def check(folder: str) -> bool:
    with open(os.path.join(folder, "config_used.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    rows = read_csv(os.path.join(folder, "raw.csv"))
    ok = True
    print(f"== {folder}")

    expected = {(j["axis"], float(j["value"]), j["replicate"]) for j in _jobs(config)}
    by_dataset = defaultdict(dict)
    for r in rows:
        by_dataset[(r["axis"], float(r["value"]), int(r["replicate"]))][r["method"]] = r
    methods = set(config["methods"])
    complete = [k for k in expected if set(by_dataset.get(k, {})) == methods]
    missing = sorted(expected - set(complete))
    print(f"datasets expected: {len(expected)}, complete: {len(complete)}, failed/missing: {len(missing)}")
    for k in missing:
        print(f"  missing: {k} methods present: {sorted(by_dataset.get(k, {}))}")
        ok = False
    print(f"rows: {len(rows)} (expected {len(expected) * len(methods)})")

    bad = 0
    for r in rows:
        for key in ("rf_norm", "runtime_s", "parsimony_score"):
            if not math.isfinite(r[key]):
                bad += 1
        if not (0 <= r["rf"] <= r["rf_max"]) or not (0 <= r["rf_norm"] <= 1):
            bad += 1
        if "loglik" in r and not (math.isfinite(r["loglik"]) and r["loglik"] < 0):
            bad += 1
    print(f"rows with invalid values: {bad}")
    ok &= bad == 0

    def criterion(method, key, better):
        if method not in methods:
            return
        losses = []
        for k, by_m in by_dataset.items():
            if method not in by_m:
                continue
            own = by_m[method][key]
            for other, r in by_m.items():
                if other != method and better(r[key], own, 1e-6):
                    losses.append((k, other, r[key] - own))
        lost = {k for k, _, _ in losses}
        print(f"{method} search beaten on {key} by another method's tree: {len(lost)} of {len(by_dataset)} datasets"
              f" ({len(losses)} dataset/method pairs)")
        for k, other, diff in losses[:10]:
            print(f"  {k}: {other} better by {abs(diff):.4g}")

    criterion("parsimony", "parsimony_score", lambda other, own, tol: other < own - tol)
    if rows and "loglik" in rows[0]:
        criterion("ml", "loglik", lambda other, own, tol: other > own + 1e-3)
        better_than_true = sum(1 for k, m in by_dataset.items() if "ml" in m and m["ml"]["loglik"] > m["ml"]["true_loglik"] + 1e-3)
        print(f"ML tree has higher likelihood than the true tree in {better_than_true} of {len(by_dataset)} datasets "
              "(expected with finite data; not an error)")
    runtime = sum(r["runtime_s"] for r in rows)
    print(f"summed inference runtime over all rows: {runtime:.1f} s")
    return ok


def main() -> int:
    folders = sys.argv[1:] or ["results/dna_benchmark", "results/lineage_benchmark"]
    results = [check(f) for f in folders]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
