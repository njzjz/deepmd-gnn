"""MACE plugin for DeePMD-kit."""

import os

from ._version import __version__
from .argcheck import (
    mace_descriptor_args,
    mace_model_args,
)

__email__ = "jinzhe.zeng@ustc.edu.cn"

__all__ = [
    "__version__",
    "mace_descriptor_args",
    "mace_model_args",
]

# make compatible with mace & e3nn & pytorch 2.6
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
