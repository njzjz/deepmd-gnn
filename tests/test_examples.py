"""Test examples."""

import json
from pathlib import Path

import pytest
from dargs.check import check
from deepmd.utils.argcheck import gen_args

from deepmd_gnn.argcheck import mace_model_args  # noqa: F401

example_path = Path(__file__).parent.parent / "examples"

examples = (
    example_path / "water" / "mace" / "input.json",
    example_path / "dprc" / "mace" / "input.json",
    example_path / "water" / "nequip" / "input.json",
    example_path / "dprc" / "nequip" / "input.json",
)


@pytest.mark.parametrize("example", examples)
def test_examples(example: Path) -> None:
    """Check whether examples meet arguments."""
    with example.open("r") as f:
        data = json.load(f)
    check(
        gen_args(),
        data,
    )
