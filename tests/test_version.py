"""Test version."""

from __future__ import annotations

from importlib.metadata import version

from deepmd_mace import __version__


def test_version() -> None:
    """Test version."""
    assert version("deepmd-mace") == __version__
