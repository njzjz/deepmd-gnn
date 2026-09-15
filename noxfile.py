"""Nox configuration file."""

from __future__ import annotations

import os

import nox

nox.options.sessions = ["tests"]

UV_OVERRIDES = "requirements-overrides.txt"
SEVENNET_OVERRIDES = "requirements-sevennet-overrides.txt"
UV_TORCH_BACKEND = os.environ.get("UV_TORCH_BACKEND", "cpu")


def install(
    session: nox.Session,
    deepmd_requirement: str,
    *extra_requirements: str,
) -> None:
    """Install all dependencies with MACE's e3nn override in one resolution."""
    session.install(
        "numpy",
        deepmd_requirement,
        *extra_requirements,
        "-e.[test]",
        "--overrides",
        UV_OVERRIDES,
        "--torch-backend",
        UV_TORCH_BACKEND,
    )


@nox.session
def tests(session: nox.Session) -> None:
    """Run test suite with pytest."""
    install(session, "deepmd-kit[torch]>=3.2.0b0")
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


@nox.session
def sevennet(session: nox.Session) -> None:
    """Run SevenNet descriptor tests with e3nn>=0.5."""
    session.install(
        "numpy",
        "deepmd-kit[torch]>=3.2.0b0",
        "-e.[test,sevennet]",
        "--overrides",
        SEVENNET_OVERRIDES,
        "--torch-backend",
        UV_TORCH_BACKEND,
    )
    session.run(
        "pytest",
        "tests/test_sevennet_descriptor.py",
        "-m",
        "not slow",
        "--cov",
        "--cov-config",
        "pyproject.toml",
        "--cov-report",
        "term",
        "--cov-report",
        "xml",
    )


@nox.session
def lammps(session: nox.Session) -> None:
    """Run LAMMPS integration tests."""
    install(
        session,
        "deepmd-kit[torch,lmp,cpu]>=3.2.0b0",
        "mpi4py",
        "mpich",
    )
    session.run(
        "pytest",
        "-m",
        "lammps",
        "tests/test_lammps.py",
        env={"DEEPMD_GNN_REQUIRE_LAMMPS": "1"},
    )
