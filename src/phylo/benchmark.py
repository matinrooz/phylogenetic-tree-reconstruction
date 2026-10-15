"""Benchmark pipelines for the DNA methods and the lineage-tracing module.

Design
------
A config file gives a *base* condition and a list of values for each
factor ("sweeps"). Each sweep varies one factor while the others stay at
their base values, so every plot shows the effect of one thing at a time.
Every (factor, value, replicate) gets its own seed derived from the config
seed with ``numpy.random.SeedSequence``, so any single dataset can be
regenerated on its own.

Metrics
-------
Applies to every method:
  * normalised Robinson-Foulds distance to the true tree (topology error)
  * wall-clock runtime of the inference
  * parsimony score of the inferred topology
  * JC69 log-likelihood of the inferred topology after optimising its
    branch lengths (DNA benchmark only)

The last two are *criteria*: parsimony search optimises the first and ML
search the second, so those methods are expected to do well on "their"
criterion. The values are also reported for the true tree, so it is
visible when an inferred tree scores better than the truth (which happens
with short alignments and says something about the criterion, not about the
method being "better than perfect").
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Sequence

import numpy as np

DNA_FIELDS = ["axis", "value", "replicate", "seed", "method", "n_taxa", "n_sites", "tree_height",
              "clock_sigma", "rf", "rf_max", "rf_norm", "runtime_s", "parsimony_score",
              "true_parsimony_score", "loglik", "true_loglik", "n_saturated_pairs"]

LINEAGE_FIELDS = ["axis", "value", "replicate", "seed", "method", "n_generations", "n_cells",
                  "n_sites", "edit_rate", "n_outcomes", "outcome_skew", "dropout_rate",
                  "fraction_edited", "rf", "rf_max", "rf_norm", "runtime_s", "parsimony_score",
                  "true_parsimony_score"]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    for key in ("seed", "replicates", "base", "sweeps", "methods"):
        if key not in config:
            raise ValueError(f"config {path} is missing '{key}'")
    return config


def _jobs(config: dict) -> List[dict]:
    jobs = []
    for a, (axis, values) in enumerate(sorted(config["sweeps"].items())):
        if axis not in config["base"]:
            raise ValueError(f"sweep axis {axis!r} is not a base parameter")
        for v, value in enumerate(values):
            for rep in range(config["replicates"]):
                seed = int(np.random.SeedSequence([config["seed"], a, v, rep]).generate_state(1)[0])
                params = dict(config["base"])
                params[axis] = value
                jobs.append({"axis": axis, "value": value, "replicate": rep, "seed": seed,
                             "params": params, "methods": config["methods"],
                             "parsimony_starts": config.get("parsimony_starts", 5)})
    return jobs


def _run_dna_job(job: dict) -> List[dict]:
    from .distance import distance_matrix
    from .methods import CharacterData, fitch_score, infer_tree, optimize_branch_lengths
    from .metrics import robinson_foulds
    from .simulate import simulate_dataset

    p = job["params"]
    true_tree, aln = simulate_dataset(int(p["n_taxa"]), int(p["n_sites"]), float(p["tree_height"]),
                                      job["seed"], clock_sigma=float(p["clock_sigma"]))
    chars = CharacterData.from_alignment(aln)
    n_sat = distance_matrix(aln).n_saturated
    true_ps = fitch_score(true_tree, chars)
    true_ll = optimize_branch_lengths(true_tree.copy(), aln)
    rows = []
    for method in job["methods"]:
        t0 = time.perf_counter()
        result = infer_tree(aln, method, seed=job["seed"], parsimony_starts=job["parsimony_starts"])
        runtime = time.perf_counter() - t0
        rf = robinson_foulds(true_tree, result.tree)
        tree_for_ll = result.tree.copy()
        for node in tree_for_ll.branches():
            node.length = max(node.length, 0.01)
        rows.append({
            "axis": job["axis"], "value": job["value"], "replicate": job["replicate"],
            "seed": job["seed"], "method": method, "n_taxa": p["n_taxa"], "n_sites": p["n_sites"],
            "tree_height": p["tree_height"], "clock_sigma": p["clock_sigma"],
            "rf": rf.distance, "rf_max": rf.max_distance, "rf_norm": rf.normalized,
            "runtime_s": runtime, "parsimony_score": fitch_score(result.tree, chars),
            "true_parsimony_score": true_ps, "loglik": optimize_branch_lengths(tree_for_ll, aln),
            "true_loglik": true_ll, "n_saturated_pairs": n_sat,
        })
    return rows


def _run_lineage_job(job: dict) -> List[dict]:
    from .lineage import LineageConfig, reconstruct, simulate_lineage
    from .methods import CharacterData, fitch_score
    from .metrics import robinson_foulds

    cfg = LineageConfig(**job["params"])
    data = simulate_lineage(cfg, job["seed"])
    chars = CharacterData.from_states(data.names, data.states, n_states=cfg.n_outcomes + 1)
    true_ps = fitch_score(data.true_tree, chars)
    rows = []
    for method in job["methods"]:
        t0 = time.perf_counter()
        res = reconstruct(data, method, seed=job["seed"], parsimony_starts=job["parsimony_starts"])
        runtime = time.perf_counter() - t0
        rf = robinson_foulds(data.true_tree, res.tree)
        row = {"axis": job["axis"], "value": job["value"], "replicate": job["replicate"],
               "seed": job["seed"], "method": method, **cfg.as_dict(),
               "fraction_edited": data.fraction_edited, "rf": rf.distance, "rf_max": rf.max_distance,
               "rf_norm": rf.normalized, "runtime_s": runtime, "parsimony_score": res.parsimony_score,
               "true_parsimony_score": true_ps}
        rows.append(row)
    return rows


def _run(jobs: List[dict], worker, fields: Sequence[str], out_csv: str, workers: int,
         progress: bool = True) -> List[dict]:
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    rows: List[dict] = []
    t0 = time.perf_counter()
    if workers <= 1:
        results: Iterable = map(worker, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(worker, jobs, chunksize=1)
    for k, job_rows in enumerate(results, start=1):
        rows.extend(job_rows)
        if progress and (k % max(1, len(jobs) // 20) == 0 or k == len(jobs)):
            print(f"  {k}/{len(jobs)} datasets done ({time.perf_counter() - t0:.0f}s)", flush=True)
    if workers > 1:
        pool.shutdown()
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_dna_benchmark(config: dict, out_csv: str, workers: int = 1, progress: bool = True) -> List[dict]:
    return _run(_jobs(config), _run_dna_job, DNA_FIELDS, out_csv, workers, progress)


def run_lineage_benchmark(config: dict, out_csv: str, workers: int = 1, progress: bool = True) -> List[dict]:
    return _run(_jobs(config), _run_lineage_job, LINEAGE_FIELDS, out_csv, workers, progress)


# ---------------------------------------------------------------------- #
# summaries
# ---------------------------------------------------------------------- #
def read_csv(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, val in row.items():
            if key in ("axis", "method"):
                continue
            try:
                row[key] = float(val)
            except (TypeError, ValueError):
                pass
    return rows


def summarize(rows: List[dict], metric: str) -> Dict[str, Dict[str, tuple]]:
    """axis -> method -> (values, means, standard errors)."""
    out: Dict[str, Dict[str, tuple]] = {}
    axes = sorted({r["axis"] for r in rows})
    for axis in axes:
        sub = [r for r in rows if r["axis"] == axis]
        values = sorted({r["value"] for r in sub})
        out[axis] = {}
        for method in dict.fromkeys(r["method"] for r in sub):
            means, sems = [], []
            for v in values:
                x = np.array([r[metric] for r in sub if r["method"] == method and r["value"] == v], dtype=float)
                means.append(float(x.mean()))
                sems.append(float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0)
            out[axis][method] = (values, means, sems)
    return out


def write_summary(rows: List[dict], metrics: Sequence[str], path: str) -> None:
    """Long-format summary table: axis, value, method, metric, mean, sem, n."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["axis", "value", "method", "metric", "mean", "sem", "n"])
        for metric in metrics:
            for axis, by_method in summarize(rows, metric).items():
                for method, (values, means, sems) in by_method.items():
                    for v, m, s in zip(values, means, sems):
                        n = sum(1 for r in rows if r["axis"] == axis and r["value"] == v and r["method"] == method)
                        writer.writerow([axis, v, method, metric, f"{m:.6g}", f"{s:.6g}", n])
