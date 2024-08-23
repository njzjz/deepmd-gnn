"""MACE plugin for DeePMD-kit."""

from ._version import __version__
from .argcheck import mace_model_args

__email__ = "jinzhe.zeng@rutgers.edu"

__all__ = [
    "__version__",
    "mace_model_args",
]
