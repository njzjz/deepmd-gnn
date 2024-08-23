"""Test training."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def test_e2e_training() -> None:
    """Test training the MACE model."""
    input_fn = "input.json"
    model_fn = "model.pth"
    # create temp directory and copy example files
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)
        this_dir = Path(__file__).parent
        data_path = this_dir / "data"
        input_path = this_dir / input_fn
        # copy data to tmpdir
        shutil.copytree(data_path, tmpdir / "data")
        # copy input.json to tmpdir
        shutil.copy(input_path, tmpdir / input_fn)

        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "deepmd",
                "--pt",
                "train",
                input_fn,
            ],
            cwd=tmpdir,
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "deepmd",
                "--pt",
                "freeze",
                "-o",
                model_fn,
            ],
            cwd=tmpdir,
        )
        assert (tmpdir / model_fn).exists()
