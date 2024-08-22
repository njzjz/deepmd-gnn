# SPDX-License-Identifier: LGPL-3.0-or-later
"""Load OP library."""
from pathlib import Path
import platform

import torch

import deepmd_mace.lib

SHARED_LIB_DIR = Path(deepmd_mace.lib.__path__[0])


def load_library(module_name: str) -> bool:
    """Load OP library.

    Parameters
    ----------
    module_name : str
        Name of the module

    Returns
    -------
    bool
        Whether the library is loaded successfully
    """
    if platform.system() == "Windows":
        ext = ".dll"
        prefix = ""
    else:
        ext = ".so"
        prefix = "lib"

    module_file = (SHARED_LIB_DIR / (prefix + module_name)).with_suffix(ext).resolve()

    torch.ops.load_library(module_file)


load_library("deepmd_mace")
