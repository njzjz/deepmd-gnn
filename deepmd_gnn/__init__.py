"""MACE plugin for DeePMD-kit."""

from ._version import __version__
from .argcheck import (
    mace_descriptor_args,
    mace_model_args,
    nequip_descriptor_args,
    sevennet_descriptor_args,
)

__email__ = "jinzhe.zeng@ustc.edu.cn"

__all__ = [
    "__version__",
    "mace_descriptor_args",
    "mace_model_args",
    "nequip_descriptor_args",
    "sevennet_descriptor_args",
]
