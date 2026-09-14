# SPDX-License-Identifier: LGPL-3.0-or-later
"""SevenNet backbone descriptor for DeePMD property fitting."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch
from deepmd.pt.model.descriptor.base_descriptor import BaseDescriptor
from deepmd.pt.utils import env
from deepmd.pt.utils.utils import to_numpy_array, to_torch_tensor
from deepmd.utils.version import check_version_compatibility

import deepmd_gnn.op  # noqa: F401
from deepmd_gnn.sevennet_checkpoint import (
    build_sevennet_feature_backbone,
    last_feature_irreps,
    load_native_sevennet_feature_backbone,
    load_sevennet_checkpoint_config,
    persistable_checkpoint_config,
    resolve_sevennet_checkpoint_path,
    scalar_even_indices,
    script_feature_backbone,
    validate_sevennet_state_dict_load,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from deepmd.utils.data_system import DeepmdDataSystem
    from deepmd.utils.path import DPPath


@BaseDescriptor.register("sevennet")
class SevenNetDescriptor(BaseDescriptor, torch.nn.Module):
    """Expose last-layer SevenNet ``0e`` features as a DeePMD descriptor.

    ``model_path`` is loaded with native Python pickle semantics and therefore
    must point to a trusted local SevenNet checkpoint or pretrained keyword.
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
            msg = "SevenNet descriptor requires the model-level type_map"
            raise ValueError(msg)
        if ntypes is not None and ntypes != len(type_map):
            msg = f"ntypes={ntypes} does not match type_map length {len(type_map)}"
            raise ValueError(msg)

        self.sel = int(sel)
        self.type_map = list(type_map)
        self.ntypes = len(self.type_map)
        self.trainable = bool(trainable)
        resolved = (
            None if model_path is None else resolve_sevennet_checkpoint_path(model_path)
        )
        if resolved is not None:
            backbone, inferred = load_native_sevennet_feature_backbone(
                resolved,
                device=str(env.DEVICE),
            )
            checkpoint_type_map = inferred["type_map"]
            if self.type_map != checkpoint_type_map:
                msg = (
                    "Model-level type_map must exactly match checkpoint "
                    "chemical_species ordering: "
                    f"expected {checkpoint_type_map}, got {self.type_map}"
                )
                raise ValueError(msg)
            self.config = persistable_checkpoint_config(inferred)
            if config is not None:
                config.clear()
                config.update(self.config)
            self.model_path = str(model_path)
            self.backbone = backbone
        elif config is not None:
            self.config = persistable_checkpoint_config(config)
            checkpoint_type_map = self.config.get("type_map", self.type_map)
            if self.type_map != checkpoint_type_map:
                msg = (
                    "Serialized SevenNet descriptor type_map mismatch: "
                    f"expected {checkpoint_type_map}, got {self.type_map}"
                )
                raise ValueError(msg)
            self.model_path = None
            self.backbone = build_sevennet_feature_backbone(
                self.config,
                device=str(env.DEVICE),
            )
        elif model_path is not None:
            msg = f"SevenNet checkpoint not found: {model_path}"
            raise FileNotFoundError(msg)
        else:
            msg = (
                "Exactly one of model_path (initialization) or config "
                "(deserialization) must be provided"
            )
            raise ValueError(msg)

        self.rcut = float(self.config["cutoff"])
        self.num_interactions = int(self.config["num_convolution_layer"])
        output_irreps = last_feature_irreps(self.backbone)
        indices = scalar_even_indices(output_irreps)
        if not indices:
            msg = f"Final SevenNet node features have no 0e channels: {output_irreps}"
            raise ValueError(msg)
        self.output_irreps = str(output_irreps)
        # Keep these as plain attributes: jit.script promotes registered
        # buffers into state_dict keys, which then fail to load training
        # checkpoints that never saved them.
        self.scalar_even_indices = indices
        self.backbone_float64 = next(self.backbone.parameters()).dtype == torch.float64
        # e3nn script compiles Gate/Activation attributes that recursive
        # torch.jit.script() cannot infer from the eager Python module.
        self.backbone = script_feature_backbone(self.backbone)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(self.trainable)

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
        """Return checkpoint elements in chemical_species order."""
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
        """Skip input statistics because SevenNet normalizes internally."""
        del merged, path

    def get_stats(self) -> dict:
        """Return the empty input-statistics collection."""
        return {}

    def set_stat_mean_and_stddev(
        self,
        mean: torch.Tensor,
        stddev: torch.Tensor,
    ) -> None:
        """Ignore external input statistics because SevenNet uses none."""
        del mean, stddev

    def get_stat_mean_and_stddev(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return empty compatibility statistics."""
        empty = torch.empty(0, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        return empty, empty.clone()

    def share_params(
        self,
        base_class: SevenNetDescriptor,
        shared_level: int,
        resume: bool = False,
    ) -> None:
        """Share a complete SevenNet backbone at level zero."""
        del resume
        if not isinstance(base_class, SevenNetDescriptor):
            msg = "SevenNet descriptors can only share with SevenNet descriptors"
            raise TypeError(msg)
        if shared_level != 0:
            msg = "SevenNet descriptor only supports full-backbone sharing at level 0"
            raise NotImplementedError(msg)
        self.backbone = base_class.backbone
        self.scalar_even_indices = base_class.scalar_even_indices
        self.backbone_float64 = base_class.backbone_float64

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: SevenNetDescriptor | None = None,
    ) -> None:
        """Reject type-map changes and subsets."""
        del type_map, model_with_new_type_stat
        msg = "SevenNet descriptor does not support changing or subsetting type_map"
        raise NotImplementedError(msg)

    def _last_layer_features(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None,
    ) -> torch.Tensor:
        nf, nloc, _ = nlist.shape
        nall = extended_atype.shape[1]
        source_dtype = torch.float64 if self.backbone_float64 else torch.float32
        positions_flat = extended_coord.view(nf, nall, 3).to(source_dtype).flatten(0, 1)
        atype = extended_atype.to(torch.int64)
        # DeePMD op rows are (neighbor, center); SevenNet stores (center, neighbor).
        edge_index = torch.ops.deepmd_gnn.edge_index(
            nlist.to(torch.int64),
            atype,
            torch.empty(0, dtype=torch.int64, device="cpu"),
        ).T[[1, 0]]
        vectors = positions_flat[edge_index[1]] - positions_flat[edge_index[0]]
        if nloc < nall:
            if mapping is None:
                msg = "SevenNet descriptor requires mapping for extended atoms"
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

        local_atype = atype[:, :nloc].reshape(-1)
        batch = (
            torch.arange(nf, dtype=torch.int64, device=atype.device)
            .unsqueeze(-1)
            .expand(nf, nloc)
            .reshape(-1)
        )
        result = self.backbone(
            {
                "x": local_atype,
                "edge_index": edge_index,
                "edge_vec": vectors,
                "batch": batch,
            },
        )
        return result["x"].view(nf, nloc, -1)

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
        """Compute local-atom invariant last-layer features."""
        if comm_dict is not None:
            msg = "MPI communication is out of scope for the SevenNet descriptor"
            raise NotImplementedError(msg)
        del fparam, charge_spin
        features = self._last_layer_features(
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
            "type": "sevennet",
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
    def deserialize(cls, data: dict) -> SevenNetDescriptor:
        """Restore a self-contained serialized SevenNet descriptor."""
        data = data.copy()
        if data.pop("@class") != "Descriptor" or data.pop("type") != "sevennet":
            msg = "data is not a serialized SevenNetDescriptor"
            raise ValueError(msg)
        check_version_compatibility(data.pop("@version"), 1, 1)
        variables = {
            name: to_torch_tensor(value)
            for name, value in data.pop("@variables").items()
        }
        descriptor = cls(**data)
        validate_sevennet_state_dict_load(
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
            msg = "SevenNet descriptor requires an explicit positive integer sel"
            raise ValueError(msg)
        model_path = local_jdata.get("model_path")
        if model_path:
            inferred = load_sevennet_checkpoint_config(model_path)
            checkpoint_type_map = inferred["type_map"]
            if type_map is not None and list(type_map) != checkpoint_type_map:
                msg = (
                    "Model-level type_map must exactly match checkpoint "
                    "chemical_species ordering: "
                    f"expected {checkpoint_type_map}, got {list(type_map)}"
                )
                raise ValueError(msg)
            local_jdata["config"] = inferred
        return local_jdata, None


__all__ = ["SevenNetDescriptor"]
