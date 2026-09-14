# SPDX-License-Identifier: LGPL-3.0-or-later
"""MACE backbone descriptor for DeePMD property fitting."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from deepmd.pt.model.descriptor.base_descriptor import BaseDescriptor
from deepmd.pt.utils import env
from deepmd.pt.utils.utils import to_numpy_array, to_torch_tensor
from deepmd.utils.version import check_version_compatibility
from e3nn import o3

import deepmd_gnn.op  # noqa: F401
from deepmd_gnn.mace_checkpoint import (
    MaceFeatureBackbone,
    build_mace_feature_backbone,
    inspect_native_mace_checkpoint,
    load_native_mace_checkpoint,
    persistable_checkpoint_config,
    validate_mace_state_dict_load,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from deepmd.utils.data_system import DeepmdDataSystem
    from deepmd.utils.path import DPPath


def _scalar_even_indices(irreps: o3.Irreps) -> list[int]:
    """Return flattened indices belonging to invariant, even scalar irreps."""
    indices: list[int] = []
    offset = 0
    for multiplicity, irrep in irreps:
        width = multiplicity * irrep.dim
        if irrep.l == 0 and irrep.p == 1:
            indices.extend(range(offset, offset + width))
        offset += width
    return indices


def _product_output_irreps(model: torch.nn.Module) -> o3.Irreps:
    """Read the actual output irreps of the final MACE product layer."""
    products = getattr(model, "products", None)
    if products is None or len(products) == 0:
        msg = "Loaded MACE model has no product layers"
        raise ValueError(msg)
    linear = getattr(products[-1], "linear", None)
    irreps_out = getattr(linear, "irreps_out", None)
    if irreps_out is None:
        msg = "Final MACE product layer does not expose output irreps"
        raise ValueError(msg)
    return o3.Irreps(irreps_out)


@BaseDescriptor.register("mace")
class MaceDescriptor(BaseDescriptor, torch.nn.Module):
    """Expose final MACE product-layer ``0e`` features as a DeePMD descriptor.

    ``model_path`` is loaded with native Python pickle semantics and therefore
    must point to a trusted local ``ScaleShiftMACE`` checkpoint.
    """

    def __init__(
        self,
        sel: int,
        model_path: str | Path | None = None,
        *,
        type_map: list[str] | None = None,
        ntypes: int | None = None,
        trainable: bool = True,
        config: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__()
        del kwargs
        if not isinstance(sel, int) or isinstance(sel, bool) or sel <= 0:
            msg = f"sel must be an explicit positive integer, got {sel!r}"
            raise ValueError(msg)
        if type_map is None:
            msg = "MACE descriptor requires the model-level type_map"
            raise ValueError(msg)
        if ntypes is not None and ntypes != len(type_map):
            msg = f"ntypes={ntypes} does not match type_map length {len(type_map)}"
            raise ValueError(msg)

        self.sel = int(sel)
        self.type_map = list(type_map)
        self.ntypes = len(self.type_map)
        self.trainable = bool(trainable)
        self.model_path: str | None = None
        source_path = None if model_path is None else Path(model_path)
        if source_path is not None and source_path.is_file():
            model = load_native_mace_checkpoint(
                source_path,
                device=str(env.DEVICE),
            )
            inferred = inspect_native_mace_checkpoint(model)
            checkpoint_type_map = inferred["type_map"]
            if self.type_map != checkpoint_type_map:
                msg = (
                    "Model-level type_map must exactly match checkpoint atomic-number "
                    f"ordering: expected {checkpoint_type_map}, got {self.type_map}"
                )
                raise ValueError(msg)
            self.config = persistable_checkpoint_config(inferred)
            if config is not None:
                config.clear()
                config.update(self.config)
            self.model_path = str(source_path)
            self.backbone = MaceFeatureBackbone(model)
        elif config is not None:
            self.config = persistable_checkpoint_config(config)
            checkpoint_type_map = self.config.get("type_map", self.type_map)
            if self.type_map != checkpoint_type_map:
                msg = (
                    "Serialized MACE descriptor type_map mismatch: "
                    f"expected {checkpoint_type_map}, got {self.type_map}"
                )
                raise ValueError(msg)
            self.model_path = None
            self.backbone = build_mace_feature_backbone(self.config)
        elif source_path is not None:
            msg = f"MACE checkpoint not found: {source_path}"
            raise FileNotFoundError(msg)
        else:
            msg = (
                "Exactly one of model_path (initialization) or config "
                "(deserialization) must be provided"
            )
            raise ValueError(msg)

        self.rcut = float(self.config["r_max"])
        self.num_interactions = int(self.config["num_interactions"])
        output_irreps = _product_output_irreps(self.backbone)
        scalar_indices = _scalar_even_indices(output_irreps)
        if not scalar_indices:
            msg = f"Final MACE product output has no 0e channels: {output_irreps}"
            raise ValueError(msg)
        self.output_irreps = str(output_irreps)
        # Keep these as plain attributes: jit.script promotes registered
        # buffers into state_dict keys, which then fail to load training
        # checkpoints that never saved them.
        self.scalar_even_indices = scalar_indices
        self.backbone_float64 = next(self.backbone.parameters()).dtype == torch.float64
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(self.trainable)

    def has_default_chg_spin(self) -> bool:
        """Declare absent charge/spin defaults for newer DeePMD model exports."""
        return False

    def get_default_chg_spin(self) -> None:
        """Return no charge/spin defaults with a concrete TorchScript type."""
        return None  # noqa: RET501

    def get_rcut(self) -> float:
        """Return the checkpoint cutoff radius."""
        return self.rcut

    def get_rcut_smth(self) -> float:
        """Return the effective smooth cutoff radius."""
        return self.rcut

    def get_sel(self) -> list[int]:
        """Return the mixed-type neighbor-list capacity."""
        return [self.sel]

    def get_ntypes(self) -> int:
        """Return the number of checkpoint elements."""
        return self.ntypes

    def get_type_map(self) -> list[str]:
        """Return checkpoint elements in atomic-number order."""
        return self.type_map

    def get_dim_out(self) -> int:
        """Return the number of final-layer ``0e`` channels."""
        return len(self.scalar_even_indices)

    def get_dim_emb(self) -> int:
        """Return the invariant feature width."""
        return self.get_dim_out()

    def mixed_types(self) -> bool:
        """Declare use of a mixed-type neighbor list."""
        return True

    def has_message_passing(self) -> bool:
        """Return whether the backbone has multiple interaction layers."""
        return self.num_interactions > 1

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
        """Return whether lower neighbor lists need sorting."""
        return False

    def get_env_protection(self) -> float:
        """Return the unused environment-matrix protection value."""
        return 0.0

    def compute_input_stats(
        self,
        merged: Callable[[], list[dict]] | list[dict],
        path: DPPath | None = None,
    ) -> None:
        """Skip input statistics because MACE normalizes internally."""
        del merged, path

    def get_stats(self) -> dict:
        """Return the empty input-statistics collection."""
        return {}

    def set_stat_mean_and_stddev(
        self,
        mean: torch.Tensor,
        stddev: torch.Tensor,
    ) -> None:
        """Ignore external input statistics because MACE uses none."""
        del mean, stddev

    def get_stat_mean_and_stddev(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return empty compatibility statistics."""
        empty = torch.empty(0, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        return empty, empty.clone()

    def share_params(
        self,
        base_class: MaceDescriptor,
        shared_level: int,
        resume: bool = False,
    ) -> None:
        """Share a complete MACE backbone at level zero."""
        del resume
        if not isinstance(base_class, MaceDescriptor):
            msg = "MACE descriptors can only share with MACE descriptors"
            raise TypeError(msg)
        if shared_level != 0:
            msg = "MACE descriptor only supports full-backbone sharing at level 0"
            raise NotImplementedError(msg)
        self.backbone = base_class.backbone
        self.scalar_even_indices = base_class.scalar_even_indices
        self.backbone_float64 = base_class.backbone_float64

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: MaceDescriptor | None = None,
    ) -> None:
        """Reject type-map changes and subsets."""
        del type_map, model_with_new_type_stat
        msg = "MACE descriptor does not support changing or subsetting type_map"
        raise NotImplementedError(msg)

    def _final_product_features(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None,
    ) -> torch.Tensor:
        nf, nloc, _ = nlist.shape
        nall = extended_atype.shape[1]
        positions = extended_coord.view(nf, nall, 3)
        source_dtype = torch.float64 if self.backbone_float64 else torch.float32
        positions_flat = positions.to(source_dtype).flatten(0, 1)
        atype = extended_atype.to(torch.int64)
        edge_index = torch.ops.deepmd_gnn.edge_index(
            nlist.to(torch.int64),
            atype,
            torch.empty(0, dtype=torch.int64, device="cpu"),
        ).T
        vectors = positions_flat[edge_index[1]] - positions_flat[edge_index[0]]
        if nloc < nall:
            if mapping is None:
                msg = "MACE descriptor requires mapping for extended atoms"
                raise ValueError(msg)
            compact_mapping = (
                mapping.to(torch.int64)
                + torch.arange(
                    nf,
                    dtype=torch.int64,
                    device=mapping.device,
                ).unsqueeze(-1)
                * nloc
            )
            edge_index = compact_mapping.reshape(-1)[edge_index]

        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        local_atype = atype[:, :nloc].reshape(-1)
        node_attrs = torch.zeros(
            (nf * nloc, self.ntypes),
            dtype=source_dtype,
            device=positions_flat.device,
        )
        node_attrs.scatter_(
            -1,
            local_atype.unsqueeze(-1),
            1,
        )
        node_feats = self.backbone.node_embedding(node_attrs)
        edge_attrs = self.backbone.spherical_harmonics(vectors)
        edge_feats, cutoff = self.backbone.radial_embedding(
            lengths,
            node_attrs,
            edge_index,
            self.backbone.atomic_numbers,
        )
        first_layer = True
        for interaction, product in zip(  # noqa: B905
            self.backbone.interactions,
            self.backbone.products,
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                cutoff=cutoff,
                first_layer=first_layer,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs,
            )
            first_layer = False
        return node_feats.view(nf, nloc, -1)

    def forward(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
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
        """Compute local-atom invariant final-product features."""
        if comm_dict is not None:
            msg = "MPI communication is out of scope for the MACE descriptor"
            raise NotImplementedError(msg)
        del fparam, charge_spin
        features = self._final_product_features(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
        )
        scalar_index = torch.tensor(
            self.scalar_even_indices,
            dtype=torch.int64,
            device=features.device,
        )
        invariant = torch.index_select(
            features,
            dim=-1,
            index=scalar_index,
        )
        return (
            invariant.to(env.GLOBAL_PT_FLOAT_PRECISION),
            None,
            None,
            None,
            None,
        )

    def serialize(self) -> dict:
        """Serialize architecture and all trained backbone state."""
        config = deepcopy(self.config)
        config["type_map"] = self.type_map
        return {
            "@class": "Descriptor",
            "@version": 1,
            "type": "mace",
            "sel": self.sel,
            "model_path": None,
            "type_map": self.type_map,
            "ntypes": self.ntypes,
            "trainable": self.trainable,
            "config": config,
            "@variables": {
                name: to_numpy_array(value)
                for name, value in self.backbone.state_dict().items()
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> MaceDescriptor:
        """Restore a self-contained serialized MACE descriptor."""
        data = data.copy()
        if data.pop("@class") != "Descriptor" or data.pop("type") != "mace":
            msg = "data is not a serialized MaceDescriptor"
            raise ValueError(msg)
        check_version_compatibility(data.pop("@version"), 1, 1)
        variables = {
            name: to_torch_tensor(value)
            for name, value in data.pop("@variables").items()
        }
        descriptor = cls(**data)
        validate_mace_state_dict_load(
            descriptor.backbone.load_state_dict(variables, strict=False),
        )
        return descriptor

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        """Persist inferred architecture and validate neighbor-list capacity."""
        del train_data
        local_jdata = local_jdata.copy()
        sel = local_jdata.get("sel")
        if not isinstance(sel, int) or isinstance(sel, bool) or sel <= 0:
            msg = "MACE descriptor requires an explicit positive integer sel"
            raise ValueError(msg)
        model_path = local_jdata.get("model_path")
        # Initialization from a trained checkpoint repeats neighbor selection
        # with its saved definition, after the native pickle may have moved.
        if model_path and (
            local_jdata.get("config") is None or Path(model_path).is_file()
        ):
            model = load_native_mace_checkpoint(Path(model_path), device="cpu")
            inferred = persistable_checkpoint_config(
                inspect_native_mace_checkpoint(model),
            )
            checkpoint_type_map = inferred["type_map"]
            if type_map is not None and list(type_map) != checkpoint_type_map:
                msg = (
                    "Model-level type_map must exactly match checkpoint atomic-number "
                    f"ordering: expected {checkpoint_type_map}, got {list(type_map)}"
                )
                raise ValueError(msg)
            local_jdata["config"] = inferred
        return local_jdata, None


__all__ = ["MaceDescriptor"]
