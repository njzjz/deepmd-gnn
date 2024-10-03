"""Test version."""

from __future__ import annotations

from importlib.metadata import version

from deepmd_gnn import __version__


def test_version() -> None:
    """Test version."""
    assert version("deepmd-gnn") == __version__
