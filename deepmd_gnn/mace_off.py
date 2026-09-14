# SPDX-License-Identifier: LGPL-3.0-or-later
"""Conservative helpers for loading selected MACE-OFF checkpoints.

The helpers in this module intentionally support a narrower scope than the
upstream discussion in PR #96: they load ordinary MACE-OFF checkpoints into a
DeePMD-GNN ``MaceModel`` wrapper for standard MD use, but they do *not* infer
DPRc/QM/MM atom-type semantics from the checkpoint.
"""

from __future__ import annotations

import importlib
import io
from contextlib import contextmanager, redirect_stdout
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import torch

from deepmd_gnn.mace_off_cli import download_mace_off_model

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from mace.modules import ScaleShiftMACE

    from deepmd_gnn.mace import MaceModel

_SUPPORTED_DISTANCE_TRANSFORMS = {
    None: "None",
    "AgnesiTransform": "Agnesi",
    "SoftTransform": "Soft",
}

_SUPPORTED_RADIAL_BASES = {
    "BesselBasis": "bessel",
}


class _InferredMaceConfig(TypedDict):
    type_map: list[str]
    r_max: float
    num_radial_basis: int
    num_cutoff_basis: int
    max_ell: int
    interaction: str
    num_interactions: int
    hidden_irreps: str
    pair_repulsion: bool
    distance_transform: str
    correlation: int
    gate: str
    MLP_irreps: str
    radial_type: str
    radial_MLP: list[int]
    std: float
    avg_num_neighbors: float
    keep_last_layer_irreps: bool


@lru_cache(maxsize=1)
def _load_mace_modules() -> tuple[type[ScaleShiftMACE], dict[str, object]]:
    # mace prints optional cuequivariance availability to stdout at import time.
    # Keep library import chatter out of CLI stdout while preserving real errors.
    with redirect_stdout(io.StringIO()):
        mace_modules = importlib.import_module("mace.modules")
    return mace_modules.ScaleShiftMACE, mace_modules.gate_dict


@lru_cache(maxsize=1)
def _load_deepmd_mace_symbols() -> tuple[list[str], type[MaceModel]]:
    # Import deepmd first so its entry-point loader registers DeePMD-GNN models
    # without causing a circular import through deepmd_gnn.mace.
    with redirect_stdout(io.StringIO()):
        importlib.import_module("deepmd.pt.model")
        mace_module = importlib.import_module("deepmd_gnn.mace")
    return mace_module.ELEMENTS, mace_module.MaceModel


@lru_cache(maxsize=1)
def _load_e3nn_script() -> Callable[[torch.nn.Module], torch.nn.Module]:
    with redirect_stdout(io.StringIO()):
        e3nn_jit = importlib.import_module("e3nn.util.jit")
    return e3nn_jit.script


@contextmanager
def _temporary_default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Compatibility alias for the generic checkpoint construction helper."""
    from deepmd_gnn.mace_checkpoint import (  # noqa: PLC0415
        temporary_default_dtype,
    )

    with temporary_default_dtype(dtype):
        yield


def _validate_atomic_numbers(atomic_numbers: list[int]) -> None:
    elements = _load_deepmd_mace_symbols()[0]
    if not atomic_numbers:
        msg = "Loaded model has empty atomic_numbers"
        raise ValueError(msg)
    if any(z < 1 or z > len(elements) for z in atomic_numbers):
        msg = (
            f"Invalid atomic numbers found: {atomic_numbers}. "
            f"Must be between 1 and {len(elements)}"
        )
        raise ValueError(msg)


def _infer_gate_name(mace_model: ScaleShiftMACE) -> str:
    gate_dict = _load_mace_modules()[1]
    if not mace_model.readouts:
        return "None"

    last_readout = mace_model.readouts[-1]
    if not hasattr(last_readout, "non_linearity"):
        return "None"

    acts = getattr(last_readout.non_linearity, "acts", None)
    if acts is None or len(acts) != 1 or not hasattr(acts[0], "f"):
        msg = "Unsupported MACE non-linearity structure"
        raise ValueError(msg)

    gate_fn = acts[0].f
    for gate_name, candidate in gate_dict.items():
        if candidate is not None and candidate is gate_fn:
            return gate_name

    msg = f"Unsupported MACE gate function: {gate_fn}"
    raise ValueError(msg)


def _infer_distance_transform(mace_model: ScaleShiftMACE) -> str:
    transform = getattr(mace_model.radial_embedding, "distance_transform", None)
    transform_name = None if transform is None else transform.__class__.__name__
    if transform_name not in _SUPPORTED_DISTANCE_TRANSFORMS:
        msg = (
            "Unsupported MACE distance transform for conservative loader: "
            f"{transform_name}"
        )
        raise ValueError(msg)
    return _SUPPORTED_DISTANCE_TRANSFORMS[transform_name]


def _infer_radial_type(mace_model: ScaleShiftMACE) -> str:
    basis_name = mace_model.radial_embedding.bessel_fn.__class__.__name__
    if basis_name not in _SUPPORTED_RADIAL_BASES:
        msg = f"Unsupported radial basis for conservative loader: {basis_name}"
        raise ValueError(msg)
    return _SUPPORTED_RADIAL_BASES[basis_name]


def _infer_radial_mlp(mace_model: ScaleShiftMACE) -> list[int]:
    layers = [
        layer
        for layer in mace_model.interactions[0].conv_tp_weights
        if hasattr(layer, "weight")
    ]
    if len(layers) < 2:
        msg = "Unsupported radial MLP structure in MACE checkpoint"
        raise ValueError(msg)
    return [int(layer.weight.shape[1]) for layer in layers[:-1]]


def _infer_interaction_name(mace_model: ScaleShiftMACE) -> str:
    interaction_names = [
        interaction.__class__.__name__ for interaction in mace_model.interactions
    ]
    if not interaction_names:
        msg = "Loaded MACE model has no interaction blocks"
        raise ValueError(msg)
    if interaction_names[0] != "RealAgnosticInteractionBlock":
        msg = (
            "Conservative MACE-OFF loader only supports checkpoints whose first "
            f"interaction block is RealAgnosticInteractionBlock, got {interaction_names[0]}"
        )
        raise ValueError(msg)
    later_names = interaction_names[1:]
    if later_names and len(set(later_names)) != 1:
        msg = f"Mixed later interaction blocks are unsupported: {interaction_names}"
        raise ValueError(msg)
    return later_names[0] if later_names else interaction_names[0]


def _infer_hidden_irreps(mace_model: ScaleShiftMACE) -> str:
    return str(mace_model.interactions[0].hidden_irreps)


def _infer_max_ell(mace_model: ScaleShiftMACE) -> int:
    return max(ir.l for _, ir in mace_model.spherical_harmonics.irreps_out)


def _infer_correlation(mace_model: ScaleShiftMACE) -> int:
    correlations = [
        int(product.symmetric_contractions.contractions[0].correlation)
        for product in mace_model.products
    ]
    if len(set(correlations)) != 1:
        msg = f"Layer-dependent correlation is unsupported: {correlations}"
        raise ValueError(msg)
    return correlations[0]


def _infer_mlp_irreps(mace_model: ScaleShiftMACE) -> str:
    last_readout = mace_model.readouts[-1]
    if not hasattr(last_readout, "hidden_irreps"):
        msg = "Checkpoint does not expose non-linear readout hidden_irreps"
        raise ValueError(msg)
    return str(last_readout.hidden_irreps)


def _infer_scale(mace_model: ScaleShiftMACE) -> float:
    scale_state = mace_model.scale_shift.state_dict()
    return float(scale_state["scale"].detach().cpu().item())


def _infer_avg_num_neighbors(mace_model: ScaleShiftMACE) -> float:
    return float(mace_model.interactions[0].avg_num_neighbors)


def _infer_keep_last_layer_irreps(mace_model: ScaleShiftMACE) -> bool:
    """Infer the constructor flag from the final product's actual output."""
    hidden_irreps = mace_model.interactions[0].hidden_irreps
    final_irreps = mace_model.products[-1].linear.irreps_out
    return str(final_irreps) == str(hidden_irreps)


def _validate_checkpoint_scope(mace_model: ScaleShiftMACE) -> None:
    atomic_numbers = mace_model.atomic_numbers.tolist()
    _validate_atomic_numbers(atomic_numbers)

    heads = getattr(mace_model, "heads", None)
    if heads not in (None, ["Default"]):
        msg = f"Multi-head checkpoints are unsupported: heads={heads}"
        raise ValueError(msg)

    if (
        hasattr(mace_model, "embedding_specs")
        and mace_model.embedding_specs is not None
    ):
        msg = "Joint-embedding checkpoints are unsupported by the conservative loader"
        raise ValueError(msg)

    if bool(getattr(mace_model, "pair_repulsion", False)):
        msg = "Pair-repulsion checkpoints are unsupported by the conservative loader"
        raise ValueError(msg)


def _infer_deepmd_config(mace_model: ScaleShiftMACE) -> _InferredMaceConfig:
    elements = _load_deepmd_mace_symbols()[0]
    _validate_checkpoint_scope(mace_model)
    atomic_numbers = mace_model.atomic_numbers.tolist()

    return {
        "type_map": [elements[z - 1] for z in atomic_numbers],
        "r_max": float(mace_model.r_max),
        "num_radial_basis": len(mace_model.radial_embedding.bessel_fn.bessel_weights),
        "num_cutoff_basis": int(mace_model.radial_embedding.cutoff_fn.p.item()),
        "max_ell": _infer_max_ell(mace_model),
        "interaction": _infer_interaction_name(mace_model),
        "num_interactions": int(mace_model.num_interactions),
        "hidden_irreps": _infer_hidden_irreps(mace_model),
        "pair_repulsion": False,
        "distance_transform": _infer_distance_transform(mace_model),
        "correlation": _infer_correlation(mace_model),
        "gate": _infer_gate_name(mace_model),
        "MLP_irreps": _infer_mlp_irreps(mace_model),
        "radial_type": _infer_radial_type(mace_model),
        "radial_MLP": _infer_radial_mlp(mace_model),
        "std": _infer_scale(mace_model),
        "avg_num_neighbors": _infer_avg_num_neighbors(mace_model),
        "keep_last_layer_irreps": _infer_keep_last_layer_irreps(mace_model),
    }


def _load_mace_checkpoint(model_path: Path, device: str) -> ScaleShiftMACE:
    """Load a trusted local MACE checkpoint object.

    Notes
    -----
    This function uses ``torch.load(..., weights_only=False)`` because the
    conservative loader needs access to the original ``ScaleShiftMACE`` object,
    not just a plain state dict. That implies normal Python unpickling semantics:
    callers should only use trusted checkpoint files, whether downloaded from the
    official download helper or supplied via a trusted local path.
    """
    from deepmd_gnn.mace_checkpoint import (  # noqa: PLC0415
        load_native_mace_checkpoint,
    )

    return load_native_mace_checkpoint(model_path, device=device)


def _validate_load_result(load_result: object) -> None:
    from deepmd_gnn.mace_checkpoint import (  # noqa: PLC0415
        validate_mace_state_dict_load,
    )

    validate_mace_state_dict_load(load_result)


def load_mace_off_model(
    model_name: str | None = "small",
    *,
    sel: int,
    model_path: Path | None = None,
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> MaceModel:
    """Load a supported MACE-OFF checkpoint as a DeePMD-GNN ``MaceModel``.

    Parameters
    ----------
    model_name
        Name of the official MACE-OFF checkpoint to download. Ignored when
        ``model_path`` is provided.
    sel
        Neighbor-list cap for the DeePMD-GNN wrapper. This value is *required*:
        MACE-OFF checkpoints do not store DeePMD's runtime neighbor-list limit,
        and guessing it can silently truncate neighbors.
    model_path
        Local checkpoint path. If omitted, download the selected official model.
    cache_dir
        Cache directory for downloaded checkpoints.
    device
        Device used when loading the original checkpoint.

    Notes
    -----
    This helper intentionally supports a narrower scope than DeePMD-GNN's full
    DPRc machinery. It only infers ordinary element ``type_map`` entries from
    ``atomic_numbers`` and does not infer ``mH`` / ``HW`` / ``OW`` or other
    QM/MM-specific type semantics.
    """
    if sel <= 0:
        msg = f"sel must be positive, got {sel}"
        raise ValueError(msg)

    if model_path is None:
        if model_name is None:
            msg = "Either model_name or model_path must be provided"
            raise ValueError(msg)
        model_path = download_mace_off_model(model_name, cache_dir=cache_dir)
    else:
        model_path = Path(model_path)

    mace_model = _load_mace_checkpoint(model_path, device=device)
    config = _infer_deepmd_config(mace_model)
    mace_model_cls = _load_deepmd_mace_symbols()[1]

    source_dtype = mace_model.atomic_energies_fn.atomic_energies.dtype
    with _temporary_default_dtype(source_dtype):
        deepmd_model = mace_model_cls(
            type_map=config["type_map"],
            sel=sel,
            r_max=config["r_max"],
            num_radial_basis=config["num_radial_basis"],
            num_cutoff_basis=config["num_cutoff_basis"],
            max_ell=config["max_ell"],
            interaction=config["interaction"],
            num_interactions=config["num_interactions"],
            hidden_irreps=config["hidden_irreps"],
            pair_repulsion=config["pair_repulsion"],
            distance_transform=config["distance_transform"],
            correlation=config["correlation"],
            gate=config["gate"],
            MLP_irreps=config["MLP_irreps"],
            radial_type=config["radial_type"],
            radial_MLP=config["radial_MLP"],
            std=config["std"],
            avg_num_neighbors=config["avg_num_neighbors"],
        )

    load_result = deepmd_model.model.load_state_dict(
        mace_model.state_dict(),
        strict=False,
    )
    _validate_load_result(load_result)
    deepmd_model.model = mace_model
    deepmd_model.eval()
    return deepmd_model


def convert_mace_off_to_deepmd(
    output_file: str,
    *,
    sel: int,
    model_name: str | None = "small",
    model_path: Path | None = None,
    cache_dir: Path | None = None,
    device: str = "cpu",
) -> Path:
    """Serialize a loaded MACE-OFF wrapper as a TorchScript DeePMD-GNN model.

    This helper scripts the DeePMD-GNN wrapper and writes it to ``output_file``.
    It does not, by itself, prove end-to-end downstream deployment in external
    engines such as LAMMPS or AMBER; callers should still validate the final
    deployment path they care about.
    """
    model = load_mace_off_model(
        model_name=model_name,
        sel=sel,
        model_path=model_path,
        cache_dir=cache_dir,
        device=device,
    )
    output_path = Path(output_file)
    e3nn_script = _load_e3nn_script()
    model.model = e3nn_script(model.model)
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, str(output_path))
    return output_path


__all__ = [
    "convert_mace_off_to_deepmd",
    "download_mace_off_model",
    "load_mace_off_model",
]
