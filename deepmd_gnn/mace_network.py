# SPDX-License-Identifier: LGPL-3.0-or-later
"""MACE network construction and freeze-time conversion helpers.

This module keeps MACE-package glue out of the DeePMD ``MaceModel`` wrapper.
The functions here build raw/scripted ``ScaleShiftMACE`` instances, preserve
parameter sharing for compile traces, and convert cuEquivariance checkpoints
back to e3nn weights when DeePMD freezes an exportable model.
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from collections.abc import MutableMapping
from typing import Any

import e3nn
import torch
from deepmd.pt.utils import env
from e3nn import o3
from e3nn.util.jit import script
from mace.modules import (
    MACE,
    ScaleShiftMACE,
    gate_dict,
    interaction_classes,
)


def make_cueq_config(enable_cueq: bool) -> object | None:
    """Build the MACE cuEquivariance config when requested."""
    if not enable_cueq:
        return None
    wrapper_ops = importlib.import_module("mace.modules.wrapper_ops")
    config_cls = getattr(wrapper_ops, "CuEquivarianceConfig", None)
    if config_cls is None:
        msg = "enable_cueq=True requires a MACE version with CuEquivarianceConfig"
        raise RuntimeError(msg)
    device_type = getattr(env.DEVICE, "type", str(env.DEVICE))
    config = config_cls(
        enabled=True,
        layout="ir_mul",
        group="O3_e3nn",
        optimize_all=True,
        conv_fusion=(device_type == "cuda"),
    )
    if not getattr(config, "enabled", False):
        msg = (
            "enable_cueq=True requires cuequivariance, cuequivariance-torch, "
            "and cuequivariance-ops-torch-cu12/cu11 to be installed. "
            "Install deepmd-gnn[cueq] for CUDA 12 or deepmd-gnn[cueq-cu11] "
            "for CUDA 11."
        )
        raise RuntimeError(msg)
    return config


def disable_cueq_in_model_params(model_params: MutableMapping[str, Any]) -> bool:
    """Disable MACE cuEquivariance flags in model definition metadata."""
    changed = False
    if model_params.get("type") == "mace" and model_params.get("enable_cueq"):
        model_params["enable_cueq"] = False
        changed = True
    model_dict = model_params.get("model_dict")
    if isinstance(model_dict, MutableMapping):
        for sub_model_params in model_dict.values():
            if isinstance(sub_model_params, MutableMapping):
                changed |= disable_cueq_in_model_params(sub_model_params)
    return changed


def transfer_cueq_to_e3nn(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    *,
    hidden_irreps: str,
    correlation: int,
    num_interactions: int,
) -> None:
    """Transfer MACE cuEquivariance weights to an e3nn MACE model."""
    from mace.cli.convert_cueq_e3nn import (  # noqa: PLC0415
        transfer_weights,
    )

    num_product_irreps = len(o3.Irreps(hidden_irreps).slices()) - 1
    use_reduced_cg = bool(getattr(source_model, "use_reduced_cg", True))
    transfer_weights(
        source_model,
        target_model,
        num_product_irreps,
        correlation,
        num_interactions,
        use_reduced_cg,
    )


def make_mace_network(
    *,
    r_max: float,
    num_radial_basis: int,
    num_cutoff_basis: int,
    max_ell: int,
    interaction: str,
    interaction_first: str = "RealAgnosticInteractionBlock",
    num_interactions: int,
    num_elements: int,
    hidden_irreps: str,
    atomic_numbers: list[int],
    avg_num_neighbors: float,
    pair_repulsion: bool,
    distance_transform: str,
    correlation: int | list[int],
    gate: str,
    MLP_irreps: str,
    std: float,
    radial_MLP: list[int],
    radial_type: str,
    enable_cueq: bool,
    script_model: bool,
    keep_last_layer_irreps: bool = False,
    heads: list[str] | None = None,
    apply_cutoff: bool = True,
    use_reduced_cg: bool = True,
    use_so3: bool = False,
    use_agnostic_product: bool = False,
    use_last_readout_only: bool = False,
    use_embedding_readout: bool = False,
    edge_irreps: str | None = None,
    use_edge_irreps_first: bool = False,
) -> torch.nn.Module:
    """Create a configured MACE subnetwork for DeePMD-GNN."""
    optimization_defaults = None
    if not script_model:
        optimization_defaults = e3nn.get_optimization_defaults()
        e3nn.set_optimization_defaults(jit_script_fx=False)
    try:
        atomic_energies = (
            torch.zeros(num_elements)
            if heads is None
            else torch.zeros((len(heads), num_elements))
        )
        optional_values = {
            "apply_cutoff": (apply_cutoff, True),
            "use_reduced_cg": (use_reduced_cg, True),
            "use_so3": (use_so3, False),
            "use_agnostic_product": (use_agnostic_product, False),
            "use_last_readout_only": (use_last_readout_only, False),
            "use_embedding_readout": (use_embedding_readout, False),
            "edge_irreps": (
                None if edge_irreps is None else o3.Irreps(edge_irreps),
                None,
            ),
            "use_edge_irreps_first": (use_edge_irreps_first, False),
            "heads": (heads, None),
            "keep_last_layer_irreps": (keep_last_layer_irreps, False),
        }
        constructor_parameters = inspect.signature(MACE.__init__).parameters
        optional_kwargs: dict[str, Any] = {}
        for name, (value, default) in optional_values.items():
            if name in constructor_parameters:
                optional_kwargs[name] = value
            elif value != default and not (name == "heads" and value == ["Default"]):
                msg = (
                    f"Installed MACE does not support checkpoint option "
                    f"{name}={value!r}"
                )
                raise ValueError(msg)

        model = ScaleShiftMACE(
            r_max=r_max,
            num_bessel=num_radial_basis,
            num_polynomial_cutoff=num_cutoff_basis,
            max_ell=max_ell,
            interaction_cls=interaction_classes[interaction],
            interaction_cls_first=interaction_classes[interaction_first],
            num_interactions=num_interactions,
            num_elements=num_elements,
            hidden_irreps=o3.Irreps(hidden_irreps),
            atomic_energies=atomic_energies,  # pylint: disable=no-explicit-device,no-explicit-dtype
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=atomic_numbers,
            pair_repulsion=pair_repulsion,
            distance_transform=distance_transform,
            correlation=correlation,
            gate=gate_dict[gate],
            MLP_irreps=o3.Irreps(MLP_irreps),
            atomic_inter_scale=std,
            atomic_inter_shift=0.0,
            radial_MLP=radial_MLP,
            radial_type=radial_type,
            cueq_config=make_cueq_config(enable_cueq),
            **optional_kwargs,
        ).to(env.DEVICE)
    finally:
        if optimization_defaults is not None:
            e3nn.set_optimization_defaults(**optimization_defaults)
    if script_model:
        try:
            return script(model)
        except Exception as exc:
            if not enable_cueq:
                raise
            warnings.warn(
                "Falling back to eager MACE submodel because cuEquivariance "
                f"could not be scripted: {exc}",
                stacklevel=2,
            )
    return model


def _set_module_state_tensor(
    root: torch.nn.Module,
    name: str,
    tensor: torch.Tensor,
) -> None:
    if "." in name:
        module_name, attr_name = name.rsplit(".", 1)
        module = root.get_submodule(module_name)
    else:
        module = root
        attr_name = name
    setattr(module, attr_name, tensor)


def link_module_state(
    target: torch.nn.Module,
    source: torch.nn.Module,
) -> None:
    """Point ``target`` parameters and buffers at the tensors owned by ``source``."""
    source_params = dict(source.named_parameters())
    target_params = dict(target.named_parameters())
    if set(source_params) != set(target_params):
        msg = "scripted and raw MACE parameter names do not match"
        raise RuntimeError(msg)
    for name, param in source_params.items():
        _set_module_state_tensor(target, name, param)

    source_buffers = dict(source.named_buffers())
    target_buffers = dict(target.named_buffers())
    if set(source_buffers) != set(target_buffers):
        msg = "scripted and raw MACE buffer names do not match"
        raise RuntimeError(msg)
    for name, buffer in source_buffers.items():
        _set_module_state_tensor(target, name, buffer)
