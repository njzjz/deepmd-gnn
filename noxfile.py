"""Nox configuration file."""

from __future__ import annotations

import nox


@nox.session
def tests(session: nox.Session) -> None:
    """Run test suite with pytest."""
    session.install(
        "numpy",
        "deepmd-kit[torch]",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
    )
    cmake_prefix_path = session.run(
        "python",
        "-c",
        "import torch;print(torch.utils.cmake_prefix_path)",
        silent=True,
    ).strip()
    session.log(f"{cmake_prefix_path=}")
    session.install("-e.[test]", env={"CMAKE_PREFIX_PATH": cmake_prefix_path})
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
