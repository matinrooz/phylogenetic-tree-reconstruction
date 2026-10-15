"""Phylogenetic tree inference written from scratch.

The main pieces:

* ``phylo.tree``       - tree data structure, splits, NNI moves
* ``phylo.newick``     - Newick reading and writing
* ``phylo.alignment``  - FASTA input and alignment validation
* ``phylo.distance``   - p-distances and Jukes-Cantor distances
* ``phylo.methods``    - UPGMA, neighbor-joining, parsimony, likelihood
* ``phylo.simulate``   - random trees and JC69 sequence simulation
* ``phylo.metrics``    - Robinson-Foulds distance
* ``phylo.lineage``    - CRISPR lineage-tracing simulator (experimental)
"""

from .alignment import Alignment, AlignmentError, read_fasta
from .distance import DistanceMatrix, distance_matrix
from .metrics import robinson_foulds
from .newick import parse_newick, write_newick
from .tree import Node, Tree, TreeError

__version__ = "0.1.0"

__all__ = [
    "Alignment", "AlignmentError", "read_fasta", "DistanceMatrix", "distance_matrix",
    "robinson_foulds", "parse_newick", "write_newick", "Node", "Tree", "TreeError",
]
