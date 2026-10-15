#!/usr/bin/env bash
# Rebuild every result and figure in the README from scratch.
#
#   ./reproduce.sh            full benchmarks (about 15-25 min on 4 cores)
#   ./reproduce.sh --quick    small configs, about a minute; writes to results/*_quick
set -euo pipefail
cd "$(dirname "$0")"

WORKERS="${WORKERS:-4}"

echo "== tests"
python -m pytest -q

if [[ "${1:-}" == "--quick" ]]; then
  python experiments/run_benchmark.py experiments/configs/dna_quick.json --workers "$WORKERS"
  python experiments/run_lineage.py experiments/configs/lineage_quick.json --workers "$WORKERS"
  python experiments/make_figures.py --dna results/dna_quick --lineage results/lineage_quick --figures results/figures_quick
else
  python experiments/run_benchmark.py --workers "$WORKERS"
  python experiments/run_lineage.py --workers "$WORKERS"
  python experiments/make_figures.py
fi
echo "done"
