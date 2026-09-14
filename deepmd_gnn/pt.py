"""PyTorch backend plugin registration."""

import sys


def load() -> None:
    """Entry point placeholder; importing this module registers plugins."""


def _is_partially_initialized(module_name: str, attr_name: str) -> bool:
    module = sys.modules.get(module_name)
    return module is not None and not hasattr(module, attr_name)


def _register() -> None:
    if _is_partially_initialized(
        "deepmd_gnn.mace",
        "MaceModel",
    ) or _is_partially_initialized("deepmd_gnn.nequip", "NequipModel"):
        return

    from deepmd.pt.model.model.model import (  # noqa: PLC0415
        BaseModel as PyTorchBaseModel,
    )

    import deepmd_gnn.mace_descriptor  # noqa: F401, PLC0415
    from deepmd_gnn.mace import MaceModel  # noqa: PLC0415
    from deepmd_gnn.nequip import NequipModel  # noqa: PLC0415

    PyTorchBaseModel.register("mace")(MaceModel)
    PyTorchBaseModel.register("nequip")(NequipModel)


_register()
