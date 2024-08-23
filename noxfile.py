"""Nox configuration file."""

from __future__ import annotations

import sys

import nox


@nox.session
def tests(session: nox.Session) -> None:
    """Run test suite with pytest."""
    session.install("torch -i https://download.pytorch.org/whl/cpu")
    cmake_prefix_path = session.run(
        sys.executable,
        "-c",
        "import torch;print(torch.utils.cmake_prefix_path)",
        silent=True,
    )
    session.env["CMAKE_PREFIX_PATH"] = cmake_prefix_path
    session.install("-e.[test]")
    session.run(
        "pytest",
        "--cov",
        "--cov-config",
        "pyproject.toml",
        "--cov-report",
        "term",
        "--cov-report",
        "xml",
    )
