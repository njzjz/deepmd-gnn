# SPDX-License-Identifier: LGPL-3.0-or-later
"""NequIP invariant node features as a DeePMD PyTorch descriptor."""

from collections import OrderedDict
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from deepmd.pt.model.descriptor.base_descriptor import BaseDescriptor
from deepmd.pt.utils import env
from deepmd.pt.utils.update_sel import UpdateSel
from deepmd.pt.utils.utils import to_numpy_array, to_torch_tensor
from deepmd.utils.data_system import DeepmdDataSystem
from deepmd.utils.path import DPPath
from deepmd.utils.version import check_version_compatibility
from e3nn import o3
from e3nn.util.jit import script
from nequip.data import AtomicDataDict
from nequip.nn import GraphModel, SequentialGraphNetwork

from deepmd_gnn.nequip import _make_nequip_network

_READOUT_MODULES = ("output_hidden_to_scalar", "total_energy_sum")
_ARCHITECTURE_DEFAULTS = {
    "r_max": 6.0,
    "num_layers": 4,
    "l_max": 2,
    "num_features": 32,
    "nonlinearity_type": "gate",
    "parity": True,
    "num_basis": 8,
    "BesselBasis_trainable": True,
    "PolynomialCutoff_p": 6,
    "invariant_layers": 2,
    "invariant_neurons": 64,
    "use_sc": True,
    "irreps_edge_sh": "0e + 1e",
    "feature_irreps_hidden": "32x0o + 32x0e + 32x1o + 32x1e",
    "chemical_embedding_irreps_out": "32x0e",
    "conv_to_output_hidden_irreps_out": "16x0e",
    "precision": "float32",
}


def _load_serialized_nequip(path: str) -> dict[str, Any]:
    """Load a trusted ``NequipModel.serialize()`` payload."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        msg = "NequIP descriptor artifact must contain a dictionary"
        raise TypeError(msg)
    if payload.get("@class") != "Model" or payload.get("type") != "nequip":
        msg = "Artifact must be a serialized NequipModel payload"
        raise ValueError(msg)
    if "@variables" not in payload:
        msg = "Serialized NequipModel artifact has no @variables"
        raise ValueError(msg)
    check_version_compatibility(payload.get("@version", 1), 1, 1)
    return payload


def _artifact_params(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if not key.startswith("@") and key != "type"
    }


def _make_nequip_backbone(
    params: dict[str, Any],
    ntypes: int,
) -> tuple[GraphModel, o3.Irreps]:
    """Build the legacy EnergyModel and truncate it after its invariant projection."""
    full_model = _make_nequip_network(params, ntypes)
    modules = full_model.model._modules  # noqa: SLF001
    if "conv_to_output_hidden" not in modules:
        msg = "Legacy EnergyModel has no conv_to_output_hidden module"
        raise ValueError(msg)
    names = list(modules)
    stop = names.index("conv_to_output_hidden") + 1
    backbone_modules = OrderedDict((name, modules[name]) for name in names[:stop])
    projected = backbone_modules["conv_to_output_hidden"]
    irreps = projected.irreps_out.get(AtomicDataDict.NODE_FEATURES_KEY)
    if irreps is None:
        msg = "conv_to_output_hidden does not declare node feature irreps"
        raise ValueError(msg)
    irreps = o3.Irreps(irreps)
    if any(ir.l != 0 or ir.p != 1 for _, ir in irreps):
        msg = (
            "NequIP descriptor requires pure invariant 0e "
            f"conv_to_output_hidden features, got {irreps}"
        )
        raise ValueError(msg)
    dtype = getattr(torch, params["precision"])
    graph = SequentialGraphNetwork(backbone_modules)
    return GraphModel(graph, model_dtype=dtype), irreps


@BaseDescriptor.register("nequip")
class NequipDescriptor(BaseDescriptor, torch.nn.Module):
    """Legacy NequIP 0.5/0.6 invariant features for standard DeePMD fittings."""

    mm_types: list[int]

    def __init__(
        self,
        sel: int,
        r_max: float = 6.0,
        num_layers: int = 4,
        l_max: int = 2,
        num_features: int = 32,
        nonlinearity_type: str = "gate",
        parity: bool = True,
        num_basis: int = 8,
        BesselBasis_trainable: bool = True,
        PolynomialCutoff_p: int = 6,
        invariant_layers: int = 2,
        invariant_neurons: int = 64,
        use_sc: bool = True,
        irreps_edge_sh: str = "0e + 1e",
        feature_irreps_hidden: str = "32x0o + 32x0e + 32x1o + 32x1e",
        chemical_embedding_irreps_out: str = "32x0e",
        conv_to_output_hidden_irreps_out: str = "16x0e",
        precision: str = "float32",
        trainable: bool = True,
        model_file: str | None = None,
        config: dict[str, Any] | None = None,
        ntypes: int | None = None,
        type_map: list[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__()
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            msg = f"Unsupported NequIP descriptor arguments: {unknown}"
            raise TypeError(msg)
        if type_map is None:
            if ntypes is None:
                msg = "NequIP descriptor requires type_map"
                raise ValueError(msg)
            type_map = [str(ii) for ii in range(ntypes)]
        if ntypes is not None and ntypes != len(type_map):
            msg = "ntypes does not match the descriptor type_map"
            raise ValueError(msg)
        params: dict[str, Any] = {
            "type_map": list(type_map),
            "sel": sel,
            "r_max": r_max,
            "num_layers": num_layers,
            "l_max": l_max,
            "num_features": num_features,
            "nonlinearity_type": nonlinearity_type,
            "parity": parity,
            "num_basis": num_basis,
            "BesselBasis_trainable": BesselBasis_trainable,
            "PolynomialCutoff_p": PolynomialCutoff_p,
            "invariant_layers": invariant_layers,
            "invariant_neurons": invariant_neurons,
            "use_sc": use_sc,
            "irreps_edge_sh": irreps_edge_sh,
            "feature_irreps_hidden": feature_irreps_hidden,
            "chemical_embedding_irreps_out": chemical_embedding_irreps_out,
            "conv_to_output_hidden_irreps_out": conv_to_output_hidden_irreps_out,
            "precision": precision,
        }
        payload = None
        # Training saves this architecture in the model definition. Once the
        # original artifact is absent, checkpoint weights restore the backbone
        # built from that definition instead of reopening an external file.
        if config is not None:
            params = deepcopy(config)
            if params.get("type_map") != type_map:
                msg = "Saved NequIP config type_map does not match descriptor type_map"
                raise ValueError(msg)
        if model_file is not None and (config is None or Path(model_file).is_file()):
            payload = _load_serialized_nequip(model_file)
            artifact_params = _artifact_params(payload)
            artifact_type_map = artifact_params.get("type_map")
            if artifact_type_map != type_map:
                msg = (
                    "NequIP artifact type_map does not match descriptor type_map: "
                    f"{artifact_type_map!r} != {type_map!r}"
                )
                raise ValueError(msg)
            if sel != artifact_params.get("sel"):
                msg = (
                    "NequIP artifact sel does not match descriptor sel: "
                    f"{artifact_params.get('sel')!r} != {sel!r}"
                )
                raise ValueError(msg)
            conflicting = {
                key: (value, artifact_params.get(key))
                for key, value in params.items()
                if key in _ARCHITECTURE_DEFAULTS
                and value != _ARCHITECTURE_DEFAULTS[key]
                and value != artifact_params.get(key)
            }
            if conflicting:
                msg = (
                    "Explicit NequIP descriptor config conflicts with artifact: "
                    f"{conflicting}"
                )
                raise ValueError(msg)
            params = artifact_params

        self.params = params
        self.type_map = list(type_map)
        self.ntypes = len(self.type_map)
        self.sel = int(params["sel"])
        self.rcut = float(params["r_max"])
        self.num_layers = int(params["num_layers"])
        self.precision = str(params["precision"])
        self.mm_types = [
            ii
            for ii, name in enumerate(self.type_map)
            if name.startswith("m") or name in {"HW", "OW"}
        ]
        self.model, feature_irreps = _make_nequip_backbone(params, self.ntypes)
        self.dim_out = feature_irreps.dim
        if payload is not None:
            self._load_backbone_variables(payload["@variables"])
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)
        self.trainable = trainable

    def __prepare_scriptable__(self) -> "NequipDescriptor":
        """Compile e3nn's traced submodules while keeping the training model eager.

        Legacy e3nn activations require its tracing-aware compiler before the
        complete DeePMD property model can be passed to ``torch.jit.script``.
        A copy preserves the caller's trainable modules and parameter identity.
        """
        descriptor = deepcopy(self)
        descriptor.model = script(descriptor.model)
        return descriptor

    def get_default_chg_spin(self) -> None:
        """Return a concrete TorchScript type for absent charge/spin defaults."""
        return None  # noqa: RET501

    def _load_backbone_variables(self, variables: dict[str, Any]) -> None:
        target = self.model.state_dict()
        source = {
            key: to_torch_tensor(value)
            for key, value in variables.items()
            if key != "e0"
            and not any(
                key.startswith(f"model.{module}.") for module in _READOUT_MODULES
            )
        }
        missing = sorted(set(target) - set(source))
        unexpected = sorted(set(source) - set(target))
        if missing or unexpected:
            msg = (
                "NequIP artifact backbone state mismatch through "
                "conv_to_output_hidden; "
                f"missing={missing}, unexpected={unexpected}"
            )
            raise ValueError(msg)
        incompatible = [
            key
            for key in target
            if tuple(source[key].shape) != tuple(target[key].shape)
        ]
        if incompatible:
            msg = f"NequIP artifact backbone tensor shape mismatch for {incompatible}"
            raise ValueError(msg)
        self.model.load_state_dict(source, strict=True)

    def get_rcut(self) -> float:
        """Return the cutoff radius."""
        return self.rcut

    def get_rcut_smth(self) -> float:
        """Return the cutoff start used by NequIP."""
        return self.rcut

    def get_sel(self) -> list[int]:
        """Return the mixed-type neighbor selection."""
        return [self.sel]

    def get_nsel(self) -> int:
        """Return the total neighbor selection."""
        return self.sel

    def get_ntypes(self) -> int:
        """Return the number of atom types."""
        return self.ntypes

    def get_type_map(self) -> list[str]:
        """Return the atom type names."""
        return self.type_map

    def get_dim_out(self) -> int:
        """Return the invariant feature width."""
        return self.dim_out

    def get_dim_emb(self) -> int:
        """Return the feature width for fitting compatibility."""
        return self.dim_out

    def get_env_protection(self) -> float:
        """Return the unused environment protection value."""
        return 0.0

    def mixed_types(self) -> bool:
        """Return whether a mixed-type neighbor list is required."""
        return True

    def has_message_passing(self) -> bool:
        """Return whether the backbone contains multiple interaction layers."""
        return self.num_layers > 1

    def has_message_passing_across_ranks(self) -> bool:
        """Declare that cross-rank feature exchange is unsupported."""
        return False

    def supports_edge_parallel(self) -> bool:
        """Declare MPI edge-parallel execution unsupported."""
        return False

    def dense_lower_supports_comm(self) -> bool:
        """Declare that the dense lower does not accept communication data."""
        return False

    def need_sorted_nlist_for_lower(self) -> bool:
        """Return whether lower neighbor lists must be sorted."""
        return False

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        """NequIP has no DeePMD descriptor input statistics."""
        del merged, path

    def get_stats(self) -> dict:
        """Return the empty descriptor-statistics mapping."""
        return {}

    def set_stat_mean_and_stddev(
        self,
        mean: torch.Tensor,
        stddev: torch.Tensor,
    ) -> None:
        """Reject statistics because NequIP does not use DeePMD normalization."""
        del mean, stddev

    def get_stat_mean_and_stddev(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the absent DeePMD descriptor statistics."""
        return None, None

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: Any | None = None,  # noqa: ANN401
    ) -> None:
        """Reject type-map changes explicitly."""
        del type_map, model_with_new_type_stat
        msg = "NequIP descriptor does not support changing type_map"
        raise NotImplementedError(msg)

    def share_params(
        self,
        base_class: Any,  # noqa: ANN401
        shared_level: int,
        resume: bool = False,
    ) -> None:
        """Share the complete backbone at level zero."""
        del resume
        if not isinstance(base_class, NequipDescriptor):
            msg = "NequIP descriptors can only share with each other"
            raise TypeError(msg)
        if shared_level != 0:
            msg = "NequIP descriptor supports only full sharing at level 0"
            raise NotImplementedError(msg)
        if self.params != base_class.params:
            msg = "Shared NequIP descriptors must have matching configs"
            raise ValueError(msg)
        self.model = base_class.model

    def forward(
        self,
        coord_ext: torch.Tensor,
        atype_ext: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        fparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Evaluate eager invariant node features without detaching the graph."""
        if comm_dict is not None:
            msg = "NequIP descriptor MPI communication is not supported"
            raise NotImplementedError(msg)
        if fparam is not None or charge_spin is not None:
            msg = "NequIP descriptor does not accept auxiliary inputs"
            raise ValueError(msg)
        nf, nall = atype_ext.shape
        nloc = nlist.shape[1]
        coord = coord_ext.view(nf, nall, 3).to(torch.float64)
        atype = atype_ext.to(torch.int64)
        edge_index = torch.ops.deepmd_gnn.edge_index(
            nlist.to(torch.int64),
            atype,
            torch.tensor(self.mm_types, dtype=torch.int64, device="cpu"),
        ).T[[1, 0]]
        coord_flat = coord.reshape(nf * nall, 3)
        atype_flat = atype.reshape(nf * nall)
        shifts = torch.zeros(
            (edge_index.shape[1], 3),
            dtype=coord.dtype,
            device=coord.device,
        )
        if mapping is not None and nloc < nall:
            mapping_flat = mapping.reshape(nf * nall) + torch.arange(
                0,
                nf * nall,
                nall,
                dtype=mapping.dtype,
                device=mapping.device,
            ).unsqueeze(-1).expand(nf, nall).reshape(-1)
            atom_shifts = coord_flat - coord_flat[mapping_flat]
            shifts = atom_shifts[edge_index[1]] - atom_shifts[edge_index[0]]
            edge_index = mapping_flat[edge_index]
        batch = (
            torch.arange(nf, dtype=torch.int64, device=coord.device)
            .unsqueeze(-1)
            .expand(nf, nall)
            .reshape(-1)
        )
        data = {
            AtomicDataDict.POSITIONS_KEY: coord_flat,
            AtomicDataDict.EDGE_INDEX_KEY: edge_index,
            AtomicDataDict.ATOM_TYPE_KEY: atype_flat,
            "batch": batch,
            "ptr": torch.arange(
                0,
                (nf + 1) * nall,
                nall,
                dtype=torch.int64,
                device=coord.device,
            ),
        }
        if mapping is not None and nloc < nall:
            data[AtomicDataDict.CELL_KEY] = (
                torch.eye(3, dtype=coord.dtype, device=coord.device)
                .unsqueeze(0)
                .expand(nf, 3, 3)
            )
            data[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = shifts
        result = self.model(data)
        features = result[AtomicDataDict.NODE_FEATURES_KEY]
        descriptor = features.view(nf, nall, self.dim_out)[:, :nloc]
        return descriptor.to(env.GLOBAL_PT_FLOAT_PRECISION), None, None, None, None

    def serialize(self) -> dict:
        """Serialize configuration and trained eager backbone state."""
        return {
            "@class": "Descriptor",
            "@version": 1,
            "type": "nequip",
            **self.params,
            "trainable": self.trainable,
            "model_file": None,
            "@variables": {
                key: to_numpy_array(value)
                for key, value in self.model.state_dict().items()
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> "NequipDescriptor":
        """Restore a serialized NequIP descriptor."""
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        data.pop("type", None)
        variables = data.pop("@variables")
        data.pop("model_file", None)
        obj = cls(**data)
        obj._load_backbone_variables(variables)
        return obj

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        """Update the mixed-type neighbor selection."""
        local_jdata = local_jdata.copy()
        model_file = local_jdata.get("model_file")
        if model_file is not None and (
            local_jdata.get("config") is None or Path(model_file).is_file()
        ):
            artifact = _artifact_params(_load_serialized_nequip(model_file))
            local_jdata["r_max"] = artifact["r_max"]
            local_jdata["sel"] = artifact["sel"]
            local_jdata["config"] = artifact
        elif local_jdata.get("config") is not None:
            # --init-model --use-pretrain-script runs neighbor statistics on
            # the saved definition before loading checkpoint weights.
            local_jdata["r_max"] = local_jdata["config"]["r_max"]
            local_jdata["sel"] = local_jdata["config"]["sel"]
        rcut = local_jdata.get("r_max", 6.0)
        min_dist, sel = UpdateSel().update_one_sel(
            train_data,
            type_map,
            rcut,
            local_jdata["sel"],
            mixed_type=True,
        )
        local_jdata["sel"] = sel[0]
        return local_jdata, min_dist
