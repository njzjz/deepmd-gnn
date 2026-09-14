# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generic native ``ScaleShiftMACE`` checkpoint support for descriptors."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, TypedDict

import torch
from ase.data import chemical_symbols
from e3nn import o3
from mace.modules import ScaleShiftMACE, gate_dict, interaction_classes

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_DISTANCE_TRANSFORMS = {
    None: "None",
    "AgnesiTransform": "Agnesi",
    "SoftTransform": "Soft",
}
_RADIAL_BASES = {"BesselBasis": "bessel"}
ALLOWED_MISSING_STATE_DICT_SUFFIXES = ("_zeroed",)
_PLACEHOLDER_GATE = "silu"
_PLACEHOLDER_MLP_IRREPS = "16x0e"


class MaceCheckpointConfig(TypedDict):
    """Constructor state needed to rebuild a supported feature backbone."""

    type_map: list[str]
    r_max: float
    num_radial_basis: int
    num_cutoff_basis: int
    max_ell: int
    interaction_first: str
    interaction: str
    num_interactions: int
    hidden_irreps: str
    pair_repulsion: bool
    apply_cutoff: bool
    use_reduced_cg: bool
    use_so3: bool
    use_agnostic_product: bool
    use_last_readout_only: bool
    use_embedding_readout: bool
    distance_transform: str
    edge_irreps: str | None
    use_edge_irreps_first: bool
    correlation: list[int]
    gate: str
    MLP_irreps: str
    radial_type: str
    radial_MLP: list[int]
    std: float
    avg_num_neighbors: float
    keep_last_layer_irreps: bool
    heads: list[str] | None
    source_dtype: str


@contextmanager
def temporary_default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Temporarily set Torch's process-wide construction dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def load_native_mace_checkpoint(
    model_path: str | Path,
    *,
    device: str | torch.device,
) -> ScaleShiftMACE:
    """Load a trusted native MACE pickle checkpoint."""
    model = torch.load(
        str(model_path),
        map_location=device,
        weights_only=False,
    )
    if not isinstance(model, ScaleShiftMACE):
        msg = (
            "Loaded checkpoint is not a ScaleShiftMACE model: "
            f"{model.__class__.__module__}.{model.__class__.__name__}"
        )
        raise TypeError(msg)
    # ``map_location`` remaps serialized storages, but e3nn TensorProduct
    # checkpoints also contain compiled TorchScript submodules.  An explicit
    # module migration is required to move their Wigner/CG constants; this is
    # the same two-step loading sequence used by MACECalculator.
    return model.to(device)


def persistable_checkpoint_config(
    config: MaceCheckpointConfig | dict[str, Any],
) -> dict[str, Any]:
    """Return JSON-safe constructor metadata for a saved DeePMD model definition."""
    return deepcopy(dict(config))


def validate_mace_state_dict_load(load_result: object) -> None:
    """Reject missing learned weights while keeping reconstructed derived buffers."""
    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    disallowed_missing_keys = [
        key
        for key in missing_keys
        if not key.endswith(ALLOWED_MISSING_STATE_DICT_SUFFIXES)
    ]
    if disallowed_missing_keys or unexpected_keys:
        msg = (
            "Failed to load MACE checkpoint into DeePMD-GNN wrapper. "
            f"missing={disallowed_missing_keys}, unexpected={unexpected_keys}"
        )
        raise RuntimeError(msg)


def _infer_gate(model: ScaleShiftMACE) -> str:
    """Return a MACE constructor gate; placeholders are fine for unused readouts."""
    if not model.readouts:
        return "None"
    last_readout = model.readouts[-1]
    non_linearity = getattr(last_readout, "non_linearity", None)
    if non_linearity is None:
        return _PLACEHOLDER_GATE
    acts = getattr(non_linearity, "acts", None)
    if acts is None or len(acts) != 1:
        return _PLACEHOLDER_GATE
    gate_fn = getattr(acts[0], "f", None)
    if gate_fn is None:
        return "None"
    for name, candidate in gate_dict.items():
        if candidate is not None and candidate is gate_fn:
            return name
    return _PLACEHOLDER_GATE


def _infer_mlp_irreps(model: ScaleShiftMACE) -> str:
    """Return readout MLP irreps, or a placeholder when the last readout is linear."""
    if not model.readouts:
        return _PLACEHOLDER_MLP_IRREPS
    mlp_irreps = getattr(model.readouts[-1], "hidden_irreps", None)
    if mlp_irreps is None:
        return _PLACEHOLDER_MLP_IRREPS
    return str(mlp_irreps)


def _infer_radial_mlp(model: ScaleShiftMACE) -> list[int]:
    layers = [
        layer
        for layer in model.interactions[0].conv_tp_weights
        if hasattr(layer, "weight")
    ]
    if len(layers) < 2:
        msg = "Unsupported radial MLP structure in MACE checkpoint"
        raise ValueError(msg)
    return [int(layer.weight.shape[1]) for layer in layers[:-1]]


def _infer_interactions(model: ScaleShiftMACE) -> tuple[str, str]:
    names = [module.__class__.__name__ for module in model.interactions]
    if not names:
        msg = "Loaded MACE model has no interaction blocks"
        raise ValueError(msg)
    unsupported = [name for name in names if name not in interaction_classes]
    if unsupported:
        msg = f"Unsupported MACE interaction classes: {unsupported}"
        raise ValueError(msg)
    later = names[1:]
    if later and len(set(later)) != 1:
        msg = f"Mixed later interaction classes are unsupported: {names}"
        raise ValueError(msg)
    return names[0], later[0] if later else names[0]


def _infer_correlations(model: ScaleShiftMACE) -> list[int]:
    return [
        int(product.symmetric_contractions.contractions[0].correlation)
        for product in model.products
    ]


def inspect_native_mace_checkpoint(model: ScaleShiftMACE) -> MaceCheckpointConfig:
    """Infer generic single-head feature-backbone construction metadata."""
    atomic_numbers = [int(value) for value in model.atomic_numbers.tolist()]
    if not atomic_numbers or any(
        number < 1 or number >= len(chemical_symbols) for number in atomic_numbers
    ):
        msg = f"Unsupported atomic numbers in MACE checkpoint: {atomic_numbers}"
        raise ValueError(msg)

    raw_heads = getattr(model, "heads", None)
    if raw_heads is not None and not isinstance(raw_heads, (list, tuple)):
        msg = f"Unsupported MACE heads metadata: {raw_heads!r}"
        raise ValueError(msg)
    heads = None if raw_heads is None else list(raw_heads)
    if heads is not None and len(heads) != 1:
        msg = f"Multi-head MACE descriptors are unsupported: heads={heads}"
        raise ValueError(msg)
    if getattr(model, "embedding_specs", None) is not None:
        msg = "Joint-embedding MACE descriptors are unsupported"
        raise ValueError(msg)
    for module in [model, *model.products]:
        cueq_config = getattr(module, "cueq_config", None)
        if cueq_config is not None and bool(getattr(cueq_config, "enabled", False)):
            msg = "cuEquivariance MACE descriptors are unsupported"
            raise ValueError(msg)

    first_interaction, later_interaction = _infer_interactions(model)
    transform = getattr(model.radial_embedding, "distance_transform", None)
    transform_name = None if transform is None else transform.__class__.__name__
    if transform_name not in _DISTANCE_TRANSFORMS:
        msg = f"Unsupported MACE distance transform: {transform_name}"
        raise ValueError(msg)
    radial_name = model.radial_embedding.bessel_fn.__class__.__name__
    if radial_name not in _RADIAL_BASES:
        msg = f"Unsupported MACE radial basis: {radial_name}"
        raise ValueError(msg)

    scale = model.scale_shift.state_dict()["scale"].detach().cpu()
    if scale.numel() != 1:
        msg = f"Single-head MACE scale must be scalar, got shape {tuple(scale.shape)}"
        raise ValueError(msg)

    hidden_irreps = model.interactions[0].hidden_irreps
    final_irreps = model.products[-1].linear.irreps_out
    source_dtype = next(model.parameters()).dtype
    edge_irreps = getattr(model, "edge_irreps", None)
    return {
        "type_map": [chemical_symbols[number] for number in atomic_numbers],
        "r_max": float(model.r_max),
        "num_radial_basis": len(
            model.radial_embedding.bessel_fn.bessel_weights,
        ),
        "num_cutoff_basis": int(model.radial_embedding.cutoff_fn.p.item()),
        "max_ell": max(ir.l for _, ir in model.spherical_harmonics.irreps_out),
        "interaction_first": first_interaction,
        "interaction": later_interaction,
        "num_interactions": int(model.num_interactions),
        "hidden_irreps": str(hidden_irreps),
        "pair_repulsion": bool(getattr(model, "pair_repulsion", False)),
        "apply_cutoff": bool(getattr(model, "apply_cutoff", True)),
        "use_reduced_cg": bool(getattr(model, "use_reduced_cg", True)),
        "use_so3": bool(getattr(model, "use_so3", False)),
        "use_agnostic_product": bool(
            getattr(model, "use_agnostic_product", False),
        ),
        "use_last_readout_only": bool(
            getattr(model, "use_last_readout_only", False),
        ),
        "use_embedding_readout": bool(
            getattr(model, "use_embedding_readout", False),
        ),
        "distance_transform": _DISTANCE_TRANSFORMS[transform_name],
        "edge_irreps": None if edge_irreps is None else str(o3.Irreps(edge_irreps)),
        "use_edge_irreps_first": bool(
            getattr(model, "use_edge_irreps_first", False),
        ),
        "correlation": _infer_correlations(model),
        "gate": _infer_gate(model),
        "MLP_irreps": _infer_mlp_irreps(model),
        "radial_type": _RADIAL_BASES[radial_name],
        "radial_MLP": _infer_radial_mlp(model),
        "std": float(scale.item()),
        "avg_num_neighbors": float(model.interactions[0].avg_num_neighbors),
        "keep_last_layer_irreps": str(final_irreps) == str(hidden_irreps),
        "heads": heads,
        "source_dtype": str(source_dtype).removeprefix("torch."),
    }


class MaceFeatureBackbone(torch.nn.Module):
    """Only the MACE modules that contribute to product-layer features."""

    def __init__(self, model: ScaleShiftMACE) -> None:
        super().__init__()
        self.node_embedding = model.node_embedding
        self.spherical_harmonics = model.spherical_harmonics
        self.radial_embedding = model.radial_embedding
        self.interactions = model.interactions
        self.products = model.products
        # Older native pickles predate MACE's zero-path flags. Derive exactly
        # the same flags from their existing CG tensors before training saves
        # state: Tester scripts a reconstructed model before a strict load, so
        # deserialize()'s missing-buffer allowance cannot repair that workflow.
        for product in self.products:
            for contraction in product.symmetric_contractions.contractions:
                correlation = int(contraction.correlation)
                for order in range(1, correlation + 1):
                    name = (
                        "weights_max_zeroed"
                        if order == correlation
                        else f"weights_{order - 1}_zeroed"
                    )
                    if not hasattr(contraction, name):
                        matrix = getattr(contraction, f"U_matrix_{order}")
                        contraction.register_buffer(name, torch.all(matrix == 0))
        self.register_buffer("atomic_numbers", model.atomic_numbers.detach().clone())


def build_mace_feature_backbone(
    config: MaceCheckpointConfig | dict[str, Any],
) -> MaceFeatureBackbone:
    """Rebuild a feature-only backbone from inspected checkpoint metadata."""
    # Keep this import local: mace_network imports DeePMD's PT environment, whose
    # entry-point registration imports MaceDescriptor and therefore this module.
    from deepmd_gnn.mace_network import make_mace_network  # noqa: PLC0415

    dtype_name = str(config.get("source_dtype", "float64"))
    source_dtype = getattr(torch, dtype_name)
    atomic_numbers = [chemical_symbols.index(name) for name in config["type_map"]]
    with temporary_default_dtype(source_dtype):
        model = make_mace_network(
            r_max=config["r_max"],
            num_radial_basis=config["num_radial_basis"],
            num_cutoff_basis=config["num_cutoff_basis"],
            max_ell=config["max_ell"],
            interaction_first=config["interaction_first"],
            interaction=config["interaction"],
            num_interactions=config["num_interactions"],
            num_elements=len(atomic_numbers),
            hidden_irreps=config["hidden_irreps"],
            atomic_numbers=atomic_numbers,
            avg_num_neighbors=config["avg_num_neighbors"],
            pair_repulsion=config["pair_repulsion"],
            apply_cutoff=config["apply_cutoff"],
            use_reduced_cg=config["use_reduced_cg"],
            use_so3=config["use_so3"],
            use_agnostic_product=config["use_agnostic_product"],
            use_last_readout_only=config["use_last_readout_only"],
            use_embedding_readout=config["use_embedding_readout"],
            distance_transform=config["distance_transform"],
            edge_irreps=config["edge_irreps"],
            use_edge_irreps_first=config["use_edge_irreps_first"],
            correlation=config["correlation"],
            gate=config["gate"],
            MLP_irreps=config["MLP_irreps"],
            std=config["std"],
            radial_MLP=config["radial_MLP"],
            radial_type=config["radial_type"],
            enable_cueq=False,
            script_model=False,
            keep_last_layer_irreps=config["keep_last_layer_irreps"],
            heads=config["heads"],
        )
    return MaceFeatureBackbone(model)


__all__ = [
    "ALLOWED_MISSING_STATE_DICT_SUFFIXES",
    "MaceCheckpointConfig",
    "MaceFeatureBackbone",
    "build_mace_feature_backbone",
    "inspect_native_mace_checkpoint",
    "load_native_mace_checkpoint",
    "persistable_checkpoint_config",
    "temporary_default_dtype",
    "validate_mace_state_dict_load",
]
