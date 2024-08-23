"""MACE plugin for DeePMD-kit."""

from ._version import __version__
from .argcheck import mace_model_args
from .mace import MaceModel as MACE  # noqa: N814

__email__ = "jinzhe.zeng@rutgers.edu"

__all__ = [
    "__version__",
    "MACE",
    "mace_model_args",
]
