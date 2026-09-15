"""PyTorch exportable backend plugin registration."""

import sys

from deepmd_gnn.torch_load_compat import trusted_e3nn_constants


def load() -> None:
    """Entry point placeholder; importing this module registers plugins."""


def _is_partially_initialized(module_name: str, attr_name: str) -> bool:
    module = sys.modules.get(module_name)
    return module is not None and not hasattr(module, attr_name)


def _register() -> None:
    if _is_partially_initialized(
        "deepmd_gnn.mace",
        "MaceModel",
    ):
        return

    from deepmd.pt_expt.model.model import (  # noqa: PLC0415
        BaseModel as ExportableBaseModel,
    )

    from deepmd_gnn.mace import MaceModel  # noqa: PLC0415

    # MaceModel predates DeepMD's explicit atomic-model capability protocol.
    # Mirror BaseAtomicModel's defaults here for capabilities that this wrapper
    # does not override itself. The MACE-specific communication hooks already
    # provide the dense MPI path used by pt_expt/LAMMPS.
    def uses_graph_lower(_model: object) -> bool:
        """Report that MACE uses the dense neighbor-list lower."""
        return False

    def supports_native_spin(_model: object) -> bool:
        """Report that this MACE wrapper has no native-spin input."""
        return False

    def supports_edge_parallel(_model: object) -> bool:
        """Report MACE domain-decomposition support."""
        return True

    def dense_lower_supports_comm(_model: object) -> bool:
        """Report that the dense MACE lower implements MPI communication."""
        return True

    def uses_compact_edge_pairs(_model: object) -> bool:
        """Report that MACE does not use DeepMD compact graph edge pairs."""
        return False

    def graph_edge_dtype(_model: object) -> str:
        """Return the model-agnostic graph edge geometry dtype."""
        return "float64"

    def supports_graph_export(_model: object) -> bool:
        """Keep the default graph-export capability value."""
        return True

    capability_defaults = {
        "uses_graph_lower": uses_graph_lower,
        "supports_native_spin": supports_native_spin,
        "supports_edge_parallel": supports_edge_parallel,
        "dense_lower_supports_comm": dense_lower_supports_comm,
        "uses_compact_edge_pairs": uses_compact_edge_pairs,
        "graph_edge_dtype": graph_edge_dtype,
        "supports_graph_export": supports_graph_export,
    }
    for name, method in capability_defaults.items():
        if not hasattr(MaceModel, name):
            setattr(MaceModel, name, method)

    # The legacy PyTorch BaseModel stores min_nbor_dist directly as a tensor
    # buffer. pt_expt 3.2 assigns the neighbor statistic as a Python float,
    # while native pt_expt models expose a float property backed by a private
    # tensor buffer. Convert that assignment at the compatibility boundary so
    # the inherited get_min_nbor_dist() contract continues to work unchanged.
    original_setattr = MaceModel.__setattr__

    def _pt_expt_setattr(model: object, name: str, value: object) -> None:
        """Accept pt_expt's float assignment to the legacy tensor buffer."""
        if name == "min_nbor_dist" and isinstance(value, (int, float)):
            buffer = vars(model).get("_buffers", {}).get(name)
            if buffer is not None:
                value = buffer.new_tensor(float(value))
        original_setattr(model, name, value)

    MaceModel.__setattr__ = _pt_expt_setattr  # type: ignore[method-assign]

    # NeQuIP 0.6/e3nn specializes atom and edge counts during torch.export.
    # Keep pt_expt registration limited to models with dynamic-shape export.
    ExportableBaseModel.register("mace")(MaceModel)


# Registry imports can load legacy e3nn constants before MaceModel is reached.
with trusted_e3nn_constants():
    _register()
