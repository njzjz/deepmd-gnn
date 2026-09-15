"""Test version."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import version

import pytest
import torch

from deepmd_gnn import __version__
from deepmd_gnn.torch_load_compat import trusted_e3nn_constants


def test_version() -> None:
    """Test version."""
    assert version("deepmd-gnn") == __version__


@pytest.mark.parametrize(
    "module",
    [
        "deepmd_gnn.mace",
        "deepmd_gnn.nequip",
        "deepmd.pt",
        "deepmd.pt_expt",
    ],
)
def test_import_does_not_disable_weights_only_loading(module: str) -> None:
    """Importing the plugin must not change process-wide torch.load policy."""
    env = os.environ.copy()
    env.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import os; import {module}; "
                "assert 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD' not in os.environ"
            ),
        ],
        check=True,
        env=env,
    )


def test_trusted_constants_preserve_caller_and_nested_allowlists() -> None:
    """A nested import must not revoke the caller's existing slice permission."""
    previous = torch.serialization.get_safe_globals()
    with torch.serialization.safe_globals([slice]):
        with trusted_e3nn_constants(), trusted_e3nn_constants():
            assert slice in torch.serialization.get_safe_globals()
        assert slice in torch.serialization.get_safe_globals()
    assert torch.serialization.get_safe_globals() == previous
