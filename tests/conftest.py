import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "data", "examples")


@pytest.fixture
def examples_dir():
    return EXAMPLES


def write_fasta(path, records):
    with open(path, "w") as handle:
        for name, seq in records:
            handle.write(f">{name}\n{seq}\n")
    return str(path)
