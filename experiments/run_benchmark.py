"""Run the DNA benchmark: python experiments/run_benchmark.py [config] [--workers N]"""

import argparse
import os
import sys

from phylo.benchmark import load_config, run_dna_benchmark, write_summary

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=os.path.join(HERE, "configs", "dna_benchmark.json"))
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel processes (runtimes are measured per process either way)")
    parser.add_argument("--out", default=None, help="output directory (default: results/<config name>)")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = args.out or os.path.join(HERE, "..", "results", config["name"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"running {config['name']} -> {os.path.normpath(out_dir)}")
    rows = run_dna_benchmark(config, os.path.join(out_dir, "raw.csv"), workers=args.workers)
    write_summary(rows, ["rf_norm", "runtime_s", "parsimony_score", "loglik"],
                  os.path.join(out_dir, "summary.csv"))
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as handle:
        import json
        json.dump(config, handle, indent=2)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
