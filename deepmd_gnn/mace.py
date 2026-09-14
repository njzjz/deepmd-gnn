# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wrapper for MACE models."""

import json
import sys
from collections.abc import MutableMapping
from copy import deepcopy
from typing import Any, Optional, cast

import torch
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.stat import (
    compute_output_stats,
)
from deepmd.pt.utils.update_sel import (
    UpdateSel,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
)
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

import deepmd_gnn.op  # noqa: F401
from deepmd_gnn.autograd import derive_atomic_virial_from_displacement
from deepmd_gnn.deepmd_ops import ensure_border_op_placeholder
from deepmd_gnn.edge import dense_edge_index
from deepmd_gnn.export import (
    clear_export_guards_once as _clear_export_guards_once,
)
from deepmd_gnn.export import (
    pad_nlist_for_export as _pad_nlist_for_export,
)
from deepmd_gnn.mace_network import (
    disable_cueq_in_model_params as _disable_cueq_in_model_params,
)
from deepmd_gnn.mace_network import (
    link_module_state as _link_module_state,
)
from deepmd_gnn.mace_network import (
    make_mace_network as _make_mace_network,
)
from deepmd_gnn.mace_network import (
    transfer_cueq_to_e3nn as _transfer_cueq_to_e3nn,
)
from deepmd_gnn.stat_compat import load_observed_type_stat_compat


def _is_freeze_command() -> bool:
    """Return whether the current process is running a DeePMD freeze command."""
    return "freeze" in sys.argv[1:]


ensure_border_op_placeholder()
(
    _restore_observed_type_from_file,
    _save_observed_type_to_file,
    collect_observed_types,
) = load_observed_type_stat_compat()

ELEMENTS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

PeriodicTable = {
    **{ee: ii + 1 for ii, ee in enumerate(ELEMENTS)},
    **{f"m{ee}": ii + 1 for ii, ee in enumerate(ELEMENTS)},
    "HW": 1,
    "OW": 8,
}


@torch.jit.unused
def _is_make_fx_fake_tensor(value: Any) -> bool:  # noqa: ANN401
    value_type = type(value)
    return (
        value_type.__module__ == "torch._subclasses.fake_tensor"
        and value_type.__name__ == "FakeTensor"
    )


@BaseModel.register("mace")
class MaceModel(BaseModel):
    """Mace model.

    Parameters
    ----------
    type_map : list[str]
        The name of each type of atoms
    sel : int
        Maximum number of neighbor atoms
    r_max : float, optional
        distance cutoff (in Ang)
    num_radial_basis : int, optional
        number of radial basis functions
    num_cutoff_basis : int, optional
        number of basis functions for smooth cutoff
    max_ell : int, optional
        highest ell of spherical harmonics
    interaction : str, optional
        name of interaction block
    num_interactions : int, optional
        number of interactions
    hidden_irreps : str, optional
        hidden irreps
    pair_repulsion : bool
        use amsgrad variant of optimizer
    distance_transform : str, optional
        distance transform
    correlation : int
        correlation order at each layer
    gate : str, optional
        non linearity for last readout
    MLP_irreps : str, optional
        hidden irreps of the MLP in last readout
    radial_type : str, optional
        type of radial basis functions
    radial_MLP : str, optional
        width of the radial MLP
    std : float, optional
        Standard deviation of force components in the training set
    """

    mm_types: list[int]
    _observed_type: Optional[list[str]]
    _use_exportable_edge_index: bool
    _use_exportable_border_op: bool

    def __init__(
        self,
        type_map: list[str],
        sel: int,
        r_max: float = 5.0,
        num_radial_basis: int = 8,
        num_cutoff_basis: int = 5,
        max_ell: int = 3,
        interaction: str = "RealAgnosticResidualInteractionBlock",
        num_interactions: int = 2,
        hidden_irreps: str = "128x0e + 128x1o",
        pair_repulsion: bool = False,
        distance_transform: str = "None",
        correlation: int = 3,
        gate: str = "silu",
        MLP_irreps: str = "16x0e",
        radial_type: str = "bessel",
        radial_MLP: list[int] = [64, 64, 64],  # noqa: B006
        std: float = 1,
        avg_num_neighbors: float | None = None,
        enable_cueq: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self._use_exportable_edge_index = False
        self._use_exportable_border_op = False
        self.params: dict[str, Any] = {
            "type_map": type_map,
            "sel": sel,
            "r_max": r_max,
            "num_radial_basis": num_radial_basis,
            "num_cutoff_basis": num_cutoff_basis,
            "max_ell": max_ell,
            "interaction": interaction,
            "num_interactions": num_interactions,
            "hidden_irreps": hidden_irreps,
            "pair_repulsion": pair_repulsion,
            "distance_transform": distance_transform,
            "correlation": correlation,
            "gate": gate,
            "MLP_irreps": MLP_irreps,
            "radial_type": radial_type,
            "radial_MLP": radial_MLP,
            "std": std,
            "avg_num_neighbors": avg_num_neighbors,
            "enable_cueq": enable_cueq,
        }
        self._disable_cueq_for_freeze = bool(enable_cueq and _is_freeze_command())
        self.type_map = type_map
        self.ntypes = len(type_map)
        self.rcut = r_max
        self.num_interactions = num_interactions
        self._observed_type = None
        atomic_numbers = []
        self.preset_out_bias: dict[str, list] = {"energy": []}
        self.mm_types = []
        self.sel = sel
        if avg_num_neighbors is None:
            avg_num_neighbors = float(sel)
        self.avg_num_neighbors = float(avg_num_neighbors)
        for ii, tt in enumerate(type_map):
            atomic_numbers.append(PeriodicTable[tt])
            if not tt.startswith("m") and tt not in {"HW", "OW"}:
                self.preset_out_bias["energy"].append(None)
            else:
                self.preset_out_bias["energy"].append([0])
                self.mm_types.append(ii)

        self.model = _make_mace_network(
            r_max=r_max,
            num_radial_basis=num_radial_basis,
            num_cutoff_basis=num_cutoff_basis,
            max_ell=max_ell,
            interaction=interaction,
            num_interactions=num_interactions,
            num_elements=self.ntypes,
            hidden_irreps=hidden_irreps,
            atomic_numbers=atomic_numbers,
            avg_num_neighbors=self.avg_num_neighbors,
            pair_repulsion=pair_repulsion,
            distance_transform=distance_transform,
            correlation=correlation,
            gate=gate,
            MLP_irreps=MLP_irreps,
            std=std,
            radial_MLP=radial_MLP,
            radial_type=radial_type,
            enable_cueq=enable_cueq and not self._disable_cueq_for_freeze,
            script_model=not self._disable_cueq_for_freeze,
        )
        self.atomic_numbers = atomic_numbers

    def _enable_cueq_for_runtime(self) -> bool:
        return bool(self.params["enable_cueq"] and not self._disable_cueq_for_freeze)

    def _params_for_serialization(self) -> dict[str, Any]:
        params = self.params.copy()
        if self._disable_cueq_for_freeze:
            params["enable_cueq"] = False
        return params

    def _sync_freeze_model_def_script(self) -> None:
        model_def_script = cast("str", getattr(self, "model_def_script", ""))
        if not self._disable_cueq_for_freeze or not model_def_script:
            return
        try:
            model_params = json.loads(model_def_script)
        except json.JSONDecodeError:
            return
        if _disable_cueq_in_model_params(model_params):
            self.model_def_script = json.dumps(model_params)

    def _make_mace_network_from_params(
        self,
        *,
        enable_cueq: bool,
        script_model: bool,
    ) -> torch.nn.Module:
        return _make_mace_network(
            r_max=self.params["r_max"],
            num_radial_basis=self.params["num_radial_basis"],
            num_cutoff_basis=self.params["num_cutoff_basis"],
            max_ell=self.params["max_ell"],
            interaction=self.params["interaction"],
            num_interactions=self.params["num_interactions"],
            num_elements=self.ntypes,
            hidden_irreps=self.params["hidden_irreps"],
            atomic_numbers=self.atomic_numbers,
            avg_num_neighbors=self.avg_num_neighbors,
            pair_repulsion=self.params["pair_repulsion"],
            distance_transform=self.params["distance_transform"],
            correlation=self.params["correlation"],
            gate=self.params["gate"],
            MLP_irreps=self.params["MLP_irreps"],
            std=self.params["std"],
            radial_MLP=self.params["radial_MLP"],
            radial_type=self.params["radial_type"],
            enable_cueq=enable_cueq,
            script_model=script_model,
        )

    def _make_raw_mace_network_from_current_state(self) -> torch.nn.Module:
        model = cast("torch.nn.Module", self.model)
        raw_model = self._make_mace_network_from_params(
            enable_cueq=self._enable_cueq_for_runtime(),
            script_model=False,
        )
        raw_model.load_state_dict(model.state_dict())
        return raw_model

    def _replace_with_scripted_e3nn_model_for_freeze(
        self,
        raw_model: torch.nn.Module | None = None,
    ) -> torch.nn.Module:
        scripted_model = self._make_mace_network_from_params(
            enable_cueq=False,
            script_model=True,
        )
        if raw_model is not None:
            scripted_model.load_state_dict(raw_model.state_dict())
        self.model = scripted_model
        return scripted_model

    def _adapt_cueq_state_for_freeze(
        self,
        state_dict: MutableMapping[str, Any],
        prefix: str,
    ) -> None:
        if not self._disable_cueq_for_freeze:
            return
        self._sync_freeze_model_def_script()
        extra_state = state_dict.get("_extra_state")
        if isinstance(extra_state, MutableMapping):
            model_params = extra_state.get("model_params")
            if isinstance(model_params, MutableMapping):
                _disable_cueq_in_model_params(model_params)

        model_prefix = f"{prefix}model."
        source_state = {
            key.removeprefix(model_prefix): value
            for key, value in state_dict.items()
            if key.startswith(model_prefix)
        }
        if "products.0.symmetric_contractions.weight" not in source_state:
            self._replace_with_scripted_e3nn_model_for_freeze()
            return

        source_model = self._make_mace_network_from_params(
            enable_cueq=True,
            script_model=False,
        )
        source_model.load_state_dict(source_state)
        target_model = cast("torch.nn.Module", self.model)
        _transfer_cueq_to_e3nn(
            source_model,
            target_model,
            hidden_irreps=self.params["hidden_irreps"],
            correlation=self.params["correlation"],
            num_interactions=self.params["num_interactions"],
        )
        scripted_model = self._replace_with_scripted_e3nn_model_for_freeze(
            target_model,
        )

        for key in list(state_dict):
            if key.startswith(model_prefix):
                del state_dict[key]
        for key, value in scripted_model.state_dict().items():
            state_dict[f"{model_prefix}{key}"] = value

    def _load_from_state_dict(
        self,
        state_dict: MutableMapping[str, Any],
        prefix: str,
        local_metadata: MutableMapping[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        self._adapt_cueq_state_for_freeze(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def atomic_model(self) -> Any:  # noqa: ANN401
        """Provide a compatibility view matching wrapped deepmd-kit models."""
        return self

    @property
    def descriptor(self) -> Any:  # noqa: ANN401
        """Provide descriptor metadata expected by deepmd-kit's pt2 export."""
        return self

    def get_descriptor(self) -> Any:  # noqa: ANN401
        """Return descriptor metadata expected by DeePMD-kit trainers."""
        if not torch.jit.is_scripting():
            self._ensure_compile_trace_model()
        return self.descriptor

    @torch.jit.unused
    def _ensure_compile_trace_model(self) -> torch.nn.Module:
        scripted_model_raw = self.model
        if scripted_model_raw is None:
            msg = "MACE network is not initialized"
            raise RuntimeError(msg)
        scripted_model = cast("torch.nn.Module", scripted_model_raw)
        cached = cast(
            "Optional[torch.nn.Module]",
            self.__dict__.get("_compile_trace_model"),
        )
        cached_source = self.__dict__.get("_compile_trace_model_source")
        if cached is not None and cached_source is scripted_model:
            cached.train(self.training)
            return cached

        raw_model = self._make_raw_mace_network_from_current_state()
        _link_module_state(raw_model, scripted_model)
        raw_model.train(self.training)
        self.__dict__["_compile_trace_model"] = raw_model
        self.__dict__["_compile_trace_model_source"] = scripted_model
        return raw_model

    @torch.jit.unused
    def _forward_lower_common_compile_trace(
        self,
        nloc: int,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor],
        fparam: Optional[torch.Tensor],
        aparam: Optional[torch.Tensor],
        do_atomic_virial: bool,
        comm_dict: Optional[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        compile_trace_model = self.__dict__.get("_compile_trace_model")
        if compile_trace_model is None:
            compile_trace_model = self._ensure_compile_trace_model()

        previous_model = self.model
        previous_exportable_edge_index = self._use_exportable_edge_index
        self.model = compile_trace_model
        self._use_exportable_edge_index = True
        try:
            return self.forward_lower_common(
                nloc,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                None,
                fparam,
                aparam,
                do_atomic_virial,
                comm_dict,
            )
        finally:
            self.model = previous_model
            self._use_exportable_edge_index = previous_exportable_edge_index

    @property
    def observed_type(self) -> Optional[list[str]]:
        """Observed element types collected during statistics."""
        return self._observed_type

    @torch.jit.export
    def get_observed_type_list(self) -> list[str]:
        """Get observed element types collected during statistics."""
        observed = self._observed_type
        if observed is None:
            return []
        observed_type_list = torch.jit.annotate(list[str], [])
        for item in observed:
            observed_type_list.append(item)
        return observed_type_list

    def compute_or_load_stat(
        self,
        sampled_func,  # noqa: ANN001
        stat_file_path: Optional[DPPath] = None,
        preset_observed_type: Optional[list[str]] = None,
    ) -> None:
        """Compute or load the statistics parameters of the model.

        For example, mean and standard deviation of descriptors or the energy bias of
        the fitting net. When `sampled` is provided, all the statistics parameters will
        be calculated (or re-calculated for update), and saved in the
        `stat_file_path`(s). When `sampled` is not provided, it will check the existence
        of `stat_file_path`(s) and load the calculated statistics parameters.

        Parameters
        ----------
        sampled_func
            The sampled data frames from different data systems.
        stat_file_path
            The path to the statistics files.
        preset_observed_type
            Optional observed element types to seed or override
            ``self._observed_type``. This compatibility parameter is accepted for
            newer deepmd-kit versions; when provided, it is used directly instead of
            restoring or collecting observed types from statistics data.
        """
        if preset_observed_type is not None:
            self._observed_type = preset_observed_type
        else:
            if stat_file_path is None:
                observed = collect_observed_types(sampled_func(), self.type_map)
            else:
                restored_observed = _restore_observed_type_from_file(stat_file_path)
                if restored_observed is None:
                    observed = collect_observed_types(sampled_func(), self.type_map)
                    _save_observed_type_to_file(stat_file_path, observed)
                else:
                    observed = restored_observed
            self._observed_type = observed
        bias_out, _ = compute_output_stats(
            sampled_func,
            self.get_ntypes(),
            keys=["energy"],
            stat_file_path=stat_file_path,
            rcond=None,
            preset_bias=self.preset_out_bias,
        )
        if "energy" in bias_out:
            self.model.atomic_energies_fn.atomic_energies = (
                bias_out["energy"]
                .view(self.model.atomic_energies_fn.atomic_energies.shape)
                .to(self.model.atomic_energies_fn.atomic_energies.dtype)
                .to(self.model.atomic_energies_fn.atomic_energies.device)
            )

    @torch.jit.export
    def fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of developer implemented atomic models."""
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy",
                    shape=[1],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
            ],
        )

    def atomic_output_def(self) -> FittingOutputDef:
        """Get the atomic output def used by the exportable backend."""
        return self.fitting_output_def()

    @torch.jit.export
    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.rcut

    @torch.jit.export
    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.type_map

    @torch.jit.export
    def get_sel(self) -> list[int]:
        """Return the number of selected atoms for each type."""
        return [self.sel]

    @torch.jit.export
    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return 0

    @torch.jit.export
    def has_default_fparam(self) -> bool:
        """Return whether default frame parameters are available."""
        return False

    def get_default_fparam(self) -> Optional[torch.Tensor]:
        """Get the default frame parameters."""
        return None

    @torch.jit.export
    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return 0

    @torch.jit.export
    def has_chg_spin_ebd(self) -> bool:
        """Return whether charge-spin embedding is enabled."""
        return False

    @torch.jit.export
    def get_dim_chg_spin(self) -> int:
        """Get the dimension of charge-spin input."""
        return 0

    @torch.jit.export
    def has_default_chg_spin(self) -> bool:
        """Return whether default charge-spin values are available."""
        return False

    def get_default_chg_spin(self) -> Optional[torch.Tensor]:
        """Get the default charge-spin values."""
        return None

    @torch.jit.export
    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return []

    @torch.jit.export
    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False

    @torch.jit.export
    def mixed_types(self) -> bool:
        """Return whether the model is in mixed-types mode.

        If true, the model
        1. assumes total number of atoms aligned across frames;
        2. uses a neighbor list that does not distinguish different atomic types.
        If false, the model
        1. assumes total number of atoms of each atom type aligned across frames;
        2. uses a neighbor list that distinguishes different atomic types.
        """
        return True

    @torch.jit.export
    def has_message_passing(self) -> bool:
        """Return whether the descriptor has message passing."""
        return self.num_interactions > 1

    @torch.jit.export
    def has_message_passing_across_ranks(self) -> bool:
        """Return whether MPI ranks must exchange layer-wise ghost features."""
        return self.has_message_passing()

    @torch.jit.export
    def need_sorted_nlist_for_lower(self) -> bool:
        """Return whether lower-interface neighbor lists must be sorted."""
        return False

    @torch.jit.export
    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        charge_spin: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        coord : torch.Tensor
            The coordinates of atoms.
        atype : torch.Tensor
            The atomic types of atoms.
        box : torch.Tensor, optional
            The box tensor.
        fparam : torch.Tensor, optional
            The frame parameters.
        aparam : torch.Tensor, optional
            The atomic parameters.
        do_atomic_virial : bool, optional
            Whether to compute atomic virial.
        """
        _ = charge_spin
        nloc = atype.shape[1]
        extended_coord, extended_atype, mapping, nlist = (
            extend_input_and_build_neighbor_list(
                coord,
                atype,
                self.rcut,
                self.get_sel(),
                mixed_types=True,
                box=box,
            )
        )
        model_ret_lower = self.forward_lower_common(
            nloc,
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            box=box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=None,
        )
        model_ret = communicate_extended_output(
            model_ret_lower,
            ModelOutputDef(self.fitting_output_def()),
            mapping,
            do_atomic_virial,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        # The output transform recomputes reduced virial from per-atom virials;
        # keep the displacement-gradient value when atom virials are not requested.
        model_predict["virial"] = model_ret_lower["energy_derv_c_redu"].squeeze(-2)
        if do_atomic_virial:
            model_predict["atom_virial"] = model_ret_lower["energy_derv_c"][
                :,
                :nloc,
            ].squeeze(-2)
        return model_predict

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
        charge_spin: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward lower pass of the model.

        Parameters
        ----------
        extended_coord : torch.Tensor
            The extended coordinates of atoms.
        extended_atype : torch.Tensor
            The extended atomic types of atoms.
        nlist : torch.Tensor
            The neighbor list.
        mapping : torch.Tensor, optional
            The mapping tensor.
        fparam : torch.Tensor, optional
            The frame parameters.
        aparam : torch.Tensor, optional
            The atomic parameters.
        do_atomic_virial : bool, optional
            Whether to compute atomic virial.
        comm_dict : dict[str, torch.Tensor], optional
            The communication dictionary.
        """
        _ = charge_spin
        nloc = nlist.shape[1]
        _nf, nall = extended_atype.shape
        # calculate nlist for ghost atoms, as LAMMPS does not calculate it
        if (
            mapping is None
            and comm_dict is None
            and self.num_interactions > 1
            and nloc < nall
        ):
            msg = (
                "Multi-layer MACE lower inference requires either comm_dict "
                "from DeePMD-kit message-passing communication or a mapping "
                "from extended atoms to local atoms."
            )
            raise ValueError(msg)

        if torch.jit.is_scripting():
            model_ret = self.forward_lower_common(
                nloc,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                None,
                fparam,
                aparam,
                do_atomic_virial,
                comm_dict,
            )
        elif _is_make_fx_fake_tensor(extended_coord):
            model_ret = self._forward_lower_common_compile_trace(
                nloc,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam,
                aparam,
                do_atomic_virial,
                comm_dict,
            )
        else:
            model_ret = self.forward_lower_common(
                nloc,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                None,
                fparam,
                aparam,
                do_atomic_virial,
                comm_dict,
            )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
        model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
        if do_atomic_virial:
            model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(-2)
        return model_predict

    def forward_lower_common(
        self,
        nloc: int,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward lower common pass of the model.

        Parameters
        ----------
        extended_coord : torch.Tensor
            The extended coordinates of atoms.
        extended_atype : torch.Tensor
            The extended atomic types of atoms.
        nlist : torch.Tensor
            The neighbor list.
        mapping : torch.Tensor, optional
            The mapping tensor.
        fparam : torch.Tensor, optional
            The frame parameters.
        aparam : torch.Tensor, optional
            The atomic parameters.
        do_atomic_virial : bool, optional
            Whether to compute atomic virial.
        comm_dict : dict[str, torch.Tensor], optional
            The communication dictionary.
        """
        nf, nall = extended_atype.shape
        extended_coord = extended_coord.reshape(nf, nall, 3)
        extended_coord_ = extended_coord
        if fparam is not None:
            msg = "fparam is unsupported"
            raise ValueError(msg)
        if aparam is not None:
            msg = "aparam is unsupported"
            raise ValueError(msg)

        # DeePMD lower interface tensors arrive batched, but MACE consumes a
        # flattened graph. Keep both layouts visible because forces and virials
        # are reshaped back to DeePMD's output convention at the end.
        nlist = nlist.to(torch.int64)
        extended_atype = extended_atype.to(torch.int64)
        nall = extended_coord.shape[1]

        extended_atype_ff = extended_atype.reshape(-1)
        edge_mask = torch.jit.annotate(Optional[torch.Tensor], None)
        # The runtime C++ op returns a sparse edge index; the exportable path
        # uses dense tensor operations so torch.export can trace dynamic shapes.
        if torch.jit.is_scripting() or not getattr(
            self,
            "_use_exportable_edge_index",
            False,
        ):
            edge_index = torch.ops.deepmd_gnn.edge_index(
                nlist,
                extended_atype,
                torch.tensor(self.mm_types, dtype=torch.int64, device="cpu"),
            )
        else:
            edge_index, edge_mask = dense_edge_index(
                nlist,
                extended_atype,
                self.mm_types,
            )
        edge_index = edge_index.T
        indices = extended_atype_ff.unsqueeze(-1)
        oh = torch.zeros(
            (nf * nall, self.ntypes),
            device=extended_atype.device,
            dtype=torch.float64,
        )
        oh.scatter_(dim=-1, index=indices, value=1)
        one_hot = oh.view((nf * nall, self.ntypes))

        default_dtype = self.model.atomic_energies_fn.atomic_energies.dtype
        extended_coord_grad = extended_coord.to(default_dtype)
        extended_coord_grad.requires_grad_(True)  # noqa: FBT003
        extended_coord_ff = extended_coord_grad.flatten(0, 1)
        nedge = edge_index.shape[1]
        shifts = torch.zeros(
            (nedge, 3),
            dtype=default_dtype,
            device=extended_coord_ff.device,
        )
        if (
            self.num_interactions > 1
            and comm_dict is None
            and mapping is not None
            and nloc < nall
        ):
            # Without MPI communication, extended ghost atoms are mapped back to
            # local atoms. The edge index and image shifts must be rewritten
            # together or later MACE layers will see inconsistent geometry.
            mapping_ff = mapping.reshape(-1) + torch.arange(
                0,
                nf * nall,
                nall,
                dtype=mapping.dtype,
                device=mapping.device,
            ).unsqueeze(-1).expand(nf, nall).reshape(-1)
            shifts_atoms = extended_coord_ff - torch.index_select(
                extended_coord_ff,
                0,
                mapping_ff,
            )
            shifts = torch.index_select(
                shifts_atoms,
                0,
                edge_index[1],
            ) - torch.index_select(shifts_atoms, 0, edge_index[0])
            edge_index = mapping_ff[edge_index]
        shifts = shifts.to(default_dtype)
        if edge_mask is not None:
            edge_mask = edge_mask.to(device=shifts.device)
            far_shifts = torch.zeros_like(shifts)
            far_shifts[:, 0] = self.rcut * 10.0 + 10.0
            shifts = torch.where(edge_mask.unsqueeze(-1), shifts, far_shifts)
        one_hot = one_hot.to(default_dtype)

        batch = (
            torch.arange(
                nf,
                dtype=torch.int64,
                device=extended_coord_ff.device,
            )
            .unsqueeze(-1)
            .expand(nf, nall)
            .reshape(-1)
        )
        ptr = torch.arange(
            0,
            (nf + 1) * nall,
            nall,
            dtype=torch.int64,
            device=extended_coord_ff.device,
        )
        weight = torch.ones(
            [nf],
            dtype=extended_coord_ff.dtype,
            device=extended_coord_ff.device,
        )

        compute_displacement = box is not None
        input_dict: dict[str, torch.Tensor] = {
            "edge_index": edge_index,
            "batch": batch,
            "node_attrs": one_hot.to(default_dtype),
            "ptr": ptr,
            "weight": weight,
        }
        displacement = torch.jit.annotate(Optional[torch.Tensor], None)
        if box is not None:
            # DeePMD virials are obtained by differentiating a symmetric cell
            # displacement. This mirrors DeePMD-kit model wrappers and avoids
            # relying on MACE's own virial output convention.
            box_tensor = (
                box.view(nf, 3, 3).to(default_dtype).to(extended_coord_ff.device)
            )
            edge_batch = torch.div(edge_index[0], nall, rounding_mode="floor")
            inv_box = torch.linalg.inv(box_tensor)
            unit_shifts = torch.einsum("ec,ecb->eb", shifts, inv_box[edge_batch])
            displacement = torch.zeros(
                (nf, 3, 3),
                dtype=extended_coord_ff.dtype,
                device=extended_coord_ff.device,
            )
            displacement.requires_grad_(requires_grad=True)
            symmetric_displacement = 0.5 * (
                displacement + displacement.transpose(-1, -2)
            )
            positions = extended_coord_ff + torch.einsum(
                "be,bec->bc",
                extended_coord_ff,
                symmetric_displacement[batch],
            )
            cell = box_tensor + torch.matmul(box_tensor, symmetric_displacement)
            input_dict["positions"] = positions
            input_dict["cell"] = cell
            input_dict["shifts"] = torch.einsum(
                "be,bec->bc",
                torch.round(unit_shifts).to(default_dtype),
                cell[edge_batch],
            )
        else:
            input_dict["positions"] = extended_coord_ff
            input_dict["cell"] = (
                torch.eye(
                    3,
                    dtype=extended_coord_ff.dtype,
                    device=extended_coord_ff.device,
                )
                .unsqueeze(0)
                .expand(nf, 3, 3)
                * 1000.0
            )
            input_dict["shifts"] = shifts

        # Choose the narrowest energy path that matches the caller: eager MACE
        # for normal training/inference, module-by-module execution for export,
        # and explicit feature exchange for MPI message passing.
        if (
            comm_dict is None
            and not torch.jit.is_scripting()
            and getattr(
                self,
                "_use_exportable_edge_index",
                False,
            )
        ):
            atom_energy_all = self.forward_mace_exportable(
                nf,
                nall,
                edge_index,
                one_hot,
                extended_coord_ff,
                shifts,
            ).view(nf, nall)
            atom_energy = atom_energy_all[:, :nloc]
        elif comm_dict is None:
            ret = self.model.forward(
                input_dict,
                compute_force=False,
                compute_virials=False,
                compute_stress=False,
                compute_displacement=False,
                training=self.training,
            )

            atom_energy_all = ret["node_energy"]
            if atom_energy_all is None:
                msg = "atom_energy is None"
                raise ValueError(msg)
            atom_energy_all = atom_energy_all.view(nf, nall)
            atom_energy = atom_energy_all[:, :nloc]
        else:
            atom_energy = self.forward_mace_with_comm(
                nloc,
                nall,
                edge_index,
                one_hot,
                extended_coord_ff,
                shifts,
                comm_dict,
            ).view(nf, nloc)
        energy = torch.sum(atom_energy, dim=1)
        grad_outputs = torch.jit.annotate(
            list[Optional[torch.Tensor]],
            [torch.ones_like(energy)],
        )
        retain_graph = self.training or do_atomic_virial

        # Forces always come from the coordinate gradient. When a box is
        # available, the reduced virial comes from the displacement gradient;
        # otherwise fall back to the position-force outer product.
        atomic_virial_fallback = torch.zeros(
            (nf, nall, 3, 3),
            dtype=extended_coord_ff.dtype,
            device=extended_coord_ff.device,
        )
        if compute_displacement and displacement is not None:
            grads = torch.autograd.grad(
                outputs=[energy],
                inputs=[extended_coord_ff, displacement],
                grad_outputs=grad_outputs,
                retain_graph=retain_graph,
                create_graph=self.training,
                allow_unused=True,
            )
            force_ff = grads[0]
            virial_tensor = grads[1]
            if force_ff is None:
                msg = "force is None"
                raise ValueError(msg)
            if virial_tensor is None:
                virial_tensor = torch.zeros(
                    (nf, 3, 3),
                    dtype=extended_coord_ff.dtype,
                    device=extended_coord_ff.device,
                )
            force = -force_ff.view(nf, nall, 3)
            virial = -virial_tensor.view(nf, 1, 9)
        else:
            force_ff = torch.autograd.grad(
                outputs=[energy],
                inputs=[extended_coord_ff],
                grad_outputs=grad_outputs,
                retain_graph=retain_graph,
                create_graph=self.training,
                allow_unused=True,
            )[0]
            if force_ff is None:
                msg = "force is None"
                raise ValueError(msg)
            force = -force_ff.view(nf, nall, 3)
            atomic_virial_fallback = force.unsqueeze(
                -1,
            ) @ extended_coord_grad.unsqueeze(
                -2,
            )
            virial = torch.sum(atomic_virial_fallback, dim=1).view(nf, 1, 9)

        atomic_virial = torch.zeros(
            (nf, nall, 1, 9),
            dtype=extended_coord_ff.dtype,
            device=extended_coord_ff.device,
        )
        if do_atomic_virial:
            if compute_displacement and displacement is not None:
                atomic_virial[:, :nloc, 0, :] = derive_atomic_virial_from_displacement(
                    atom_energy,
                    displacement,
                    nloc,
                    self.training,
                )
            else:
                atomic_virial[:, :, 0, :] = atomic_virial_fallback.view(nf, nall, 9)

        return {
            "energy_redu": energy.view(nf, 1).to(extended_coord_.dtype),
            "energy_derv_r": force.view(nf, nall, 1, 3).to(extended_coord_.dtype),
            "energy_derv_c_redu": virial.to(extended_coord_.dtype),
            "energy": atom_energy.view(nf, nloc, 1).to(extended_coord_.dtype),
            "energy_derv_c": atomic_virial.to(extended_coord_.dtype),
        }

    def forward_mace_exportable(
        self,
        nf: int,
        nall: int,
        edge_index: torch.Tensor,
        node_attrs: torch.Tensor,
        positions: torch.Tensor,
        shifts: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate node energies through traceable MACE submodules."""
        vectors = torch.index_select(positions, 0, edge_index[1]) - torch.index_select(
            positions,
            0,
            edge_index[0],
        )
        vectors = vectors + shifts
        vectors = vectors.reshape(-1, 3)
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        node_heads = torch.zeros(
            (nf * nall,),
            dtype=torch.int64,
            device=node_attrs.device,
        )
        num_atoms_arange = torch.arange(
            nf * nall,
            dtype=torch.int64,
            device=node_attrs.device,
        )
        node_e0 = self.model.atomic_energies_fn(node_attrs)[
            num_atoms_arange,
            node_heads,
        ]

        node_feats = self.model.node_embedding(node_attrs)
        edge_attrs = self.model.spherical_harmonics(vectors)
        edge_feats, cutoff = self.model.radial_embedding(
            lengths,
            node_attrs,
            edge_index,
            self.model.atomic_numbers,
        )

        if hasattr(self.model, "pair_repulsion"):
            pair_node_energy = self.model.pair_repulsion_fn(
                lengths,
                node_attrs,
                edge_index,
                self.model.atomic_numbers,
            )
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        node_es_list = [pair_node_energy]
        node_feats_list = []

        for i, (interaction, product) in enumerate(
            zip(self.model.interactions, self.model.products),  # noqa: B905
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                cutoff=cutoff,
                first_layer=(i == 0),
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs,
            )
            node_feats_list.append(node_feats)

        for i, readout in enumerate(self.model.readouts):
            feat_idx = -1 if len(self.model.readouts) == 1 else i
            node_es_list.append(
                readout(node_feats_list[feat_idx], node_heads)[
                    num_atoms_arange,
                    node_heads,
                ],
            )

        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.model.scale_shift(node_inter_es, node_heads)
        return node_e0.clone().double() + node_inter_es.clone().double()

    def forward_common_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        charge_spin: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward lower pass with internal DeePMD output names."""
        _ = charge_spin
        return self.forward_lower_common(
            nlist.shape[1],
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
        )

    def forward_common_lower_exportable(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        charge_spin: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        **make_fx_kwargs: object,
    ) -> torch.nn.Module:
        """Trace ``forward_common_lower`` for ``torch.export`` serialization."""
        make_fx_kwargs = make_fx_kwargs.copy()
        model = self
        scripted_model = self.model
        raw_model = self._make_raw_mace_network_from_current_state()
        raw_model.to(extended_coord.device)
        raw_model.train(scripted_model.training)

        def fn(
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            nlist: torch.Tensor,
            mapping: Optional[torch.Tensor],
            fparam: Optional[torch.Tensor],
            aparam: Optional[torch.Tensor],
            charge_spin: Optional[torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            torch._check(extended_coord.shape[1] >= 2)  # noqa: SLF001
            extended_coord = extended_coord.detach().requires_grad_(requires_grad=True)
            nlist = _pad_nlist_for_export(nlist)
            return model.forward_common_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fparam,
                aparam=aparam,
                charge_spin=charge_spin,
                do_atomic_virial=do_atomic_virial,
            )

        previous_exportable_edge_index = self._use_exportable_edge_index
        self._use_exportable_edge_index = True
        self.model = raw_model
        try:
            traced = make_fx(fn, **make_fx_kwargs)(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam,
                aparam,
                charge_spin,
            )
            _clear_export_guards_once(traced)
            return traced
        finally:
            self.model = scripted_model
            self._use_exportable_edge_index = previous_exportable_edge_index

    def forward_common_lower_exportable_with_comm(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor],
        fparam: Optional[torch.Tensor],
        aparam: Optional[torch.Tensor],
        charge_spin: Optional[torch.Tensor],
        send_list: torch.Tensor,
        send_proc: torch.Tensor,
        recv_proc: torch.Tensor,
        send_num: torch.Tensor,
        recv_num: torch.Tensor,
        communicator: torch.Tensor,
        nlocal: torch.Tensor,
        nghost: torch.Tensor,
        do_atomic_virial: bool = False,
        **make_fx_kwargs: object,
    ) -> torch.nn.Module:
        """Trace ``forward_common_lower`` with explicit MPI comm tensors."""
        make_fx_kwargs = make_fx_kwargs.copy()
        model = self
        scripted_model = self.model
        raw_model = self._make_raw_mace_network_from_current_state()
        raw_model.to(extended_coord.device)
        raw_model.train(scripted_model.training)

        def fn(
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            nlist: torch.Tensor,
            mapping: Optional[torch.Tensor],
            fparam: Optional[torch.Tensor],
            aparam: Optional[torch.Tensor],
            charge_spin: Optional[torch.Tensor],
            send_list: torch.Tensor,
            send_proc: torch.Tensor,
            recv_proc: torch.Tensor,
            send_num: torch.Tensor,
            recv_num: torch.Tensor,
            communicator: torch.Tensor,
            nlocal: torch.Tensor,
            nghost: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            torch._check(extended_coord.shape[1] >= 2)  # noqa: SLF001
            extended_coord = extended_coord.detach().requires_grad_(requires_grad=True)
            nlist = _pad_nlist_for_export(nlist)
            comm_dict = {
                "send_list": send_list,
                "send_proc": send_proc,
                "recv_proc": recv_proc,
                "send_num": send_num,
                "recv_num": recv_num,
                "communicator": communicator,
                "nlocal": nlocal,
                "nghost": nghost,
            }
            return model.forward_common_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fparam,
                aparam=aparam,
                charge_spin=charge_spin,
                do_atomic_virial=do_atomic_virial,
                comm_dict=comm_dict,
            )

        previous_exportable_edge_index = self._use_exportable_edge_index
        previous_exportable_border_op = self._use_exportable_border_op
        self._use_exportable_edge_index = True
        self._use_exportable_border_op = True
        self.model = raw_model
        try:
            traced = make_fx(fn, **make_fx_kwargs)(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam,
                aparam,
                charge_spin,
                send_list,
                send_proc,
                recv_proc,
                send_num,
                recv_num,
                communicator,
                nlocal,
                nghost,
            )
            _clear_export_guards_once(traced)
            return traced
        finally:
            self.model = scripted_model
            self._use_exportable_edge_index = previous_exportable_edge_index
            self._use_exportable_border_op = previous_exportable_border_op

    def communicate_node_features(
        self,
        node_feats: torch.Tensor,
        nloc: int,
        nall: int,
        comm_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Communicate local node features to ghost slots using DeePMD border_op."""
        node_feats = node_feats[:nloc]
        if not torch.jit.is_scripting() and getattr(
            self,
            "_use_exportable_border_op",
            False,
        ):
            node_feats = torch.nn.functional.pad(
                node_feats,
                (0, 0, 0, nall - nloc),
                value=0.0,
            )
            return torch.ops.deepmd_export.border_op(
                comm_dict["send_list"],
                comm_dict["send_proc"],
                comm_dict["recv_proc"],
                comm_dict["send_num"],
                comm_dict["recv_num"],
                node_feats,
                comm_dict["communicator"],
                comm_dict["nlocal"],
                comm_dict["nghost"],
            )
        n_padding = nall - nloc
        if n_padding > 0:
            node_feats = torch.nn.functional.pad(
                node_feats,
                (0, 0, 0, n_padding),
                value=0.0,
            )
        ret = torch.ops.deepmd.border_op(
            comm_dict["send_list"],
            comm_dict["send_proc"],
            comm_dict["recv_proc"],
            comm_dict["send_num"],
            comm_dict["recv_num"],
            node_feats,
            comm_dict["communicator"],
            torch.tensor(
                nloc,
                dtype=torch.int32,
                device=torch.device("cpu"),
            ),
            torch.tensor(
                nall - nloc,
                dtype=torch.int32,
                device=torch.device("cpu"),
            ),
        )
        return ret[0]

    def forward_mace_with_comm(
        self,
        nloc: int,
        nall: int,
        edge_index: torch.Tensor,
        node_attrs: torch.Tensor,
        positions: torch.Tensor,
        shifts: torch.Tensor,
        comm_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Evaluate MACE node energies with layer-wise MPI ghost communication."""
        if (
            torch.jit.is_scripting()
            or not getattr(self, "_use_exportable_border_op", False)
        ) and positions.shape[0] != nall:
            msg = "MACE comm_dict lower inference only supports one frame"
            raise ValueError(msg)
        vectors = torch.index_select(positions, 0, edge_index[1]) - torch.index_select(
            positions,
            0,
            edge_index[0],
        )
        vectors = vectors + shifts
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        node_heads = torch.zeros(
            (nall,),
            dtype=torch.int64,
            device=node_attrs.device,
        )
        num_atoms_arange = torch.arange(
            nall,
            dtype=torch.int64,
            device=node_attrs.device,
        )
        node_e0 = self.model.atomic_energies_fn(node_attrs)[
            num_atoms_arange,
            node_heads,
        ][:nloc]

        node_feats = self.model.node_embedding(node_attrs)
        edge_attrs = self.model.spherical_harmonics(vectors)
        edge_feats, cutoff = self.model.radial_embedding(
            lengths,
            node_attrs,
            edge_index,
            self.model.atomic_numbers,
        )

        if hasattr(self.model, "pair_repulsion"):
            pair_node_energy = self.model.pair_repulsion_fn(
                lengths,
                node_attrs,
                edge_index,
                self.model.atomic_numbers,
            )[:nloc]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        node_es_list = torch.jit.annotate(list[torch.Tensor], [pair_node_energy])
        node_feats_list = torch.jit.annotate(list[torch.Tensor], [])

        for i, (interaction, product) in enumerate(
            zip(self.model.interactions, self.model.products),  # noqa: B905
        ):
            node_feats, sc = interaction(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
                cutoff=cutoff,
                first_layer=(i == 0),
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=node_attrs,
            )
            node_feats_list.append(node_feats[:nloc])
            if i < self.num_interactions - 1:
                node_feats = self.communicate_node_features(
                    node_feats,
                    nloc,
                    nall,
                    comm_dict,
                )

        for i, readout in enumerate(self.model.readouts):
            feat_idx = -1 if len(self.model.readouts) == 1 else i
            node_es_list.append(
                readout(node_feats_list[feat_idx], node_heads[:nloc])[
                    torch.arange(
                        nloc,
                        dtype=torch.int64,
                        device=node_attrs.device,
                    ),
                    node_heads[:nloc],
                ],
            )

        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        node_inter_es = self.model.scale_shift(node_inter_es, node_heads[:nloc])
        return node_e0.clone().double() + node_inter_es.clone().double()

    def serialize(self) -> dict:
        """Serialize the model."""
        return {
            "@class": "Model",
            "@version": 1,
            "type": "mace",
            **self._params_for_serialization(),
            "@variables": {
                kk: to_numpy_array(vv) for kk, vv in self.model.state_dict().items()
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> "MaceModel":
        """Deserialize the model."""
        data = data.copy()
        if not (data.pop("@class") == "Model" and data.pop("type") == "mace"):
            msg = "data is not a serialized MaceModel"
            raise ValueError(msg)
        check_version_compatibility(data.pop("@version"), 1, 1)
        variables = {
            kk: to_torch_tensor(vv) for kk, vv in data.pop("@variables").items()
        }
        model = cls(**data)
        model.model.load_state_dict(variables)
        return model

    @torch.jit.export
    def get_nnei(self) -> int:
        """Return the total number of selected neighboring atoms in cut-off radius."""
        return self.sel

    @torch.jit.export
    def get_nsel(self) -> int:
        """Return the total number of selected neighboring atoms in cut-off radius."""
        return self.sel

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: Optional[list[str]],
        local_jdata: dict,
    ) -> tuple[dict, Optional[float]]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statictics
        type_map : list[str], optional
            The name of each type of atoms
        local_jdata : dict
            The local data refer to the current class

        Returns
        -------
        dict
            The updated local data
        float
            The minimum distance between two atoms
        """
        local_jdata_cpy = local_jdata.copy()
        min_nbor_dist, sel = UpdateSel().update_one_sel(
            train_data,
            type_map,
            local_jdata_cpy["r_max"],
            local_jdata_cpy["sel"],
            mixed_type=True,
        )
        local_jdata_cpy["sel"] = sel[0]
        return local_jdata_cpy, min_nbor_dist

    @torch.jit.export
    def model_output_type(self) -> list[str]:
        """Get the output type for the model."""
        return ["energy"]

    def translated_output_def(self) -> dict[str, Any]:
        """Get the translated output def for the model."""
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": deepcopy(out_def_data["energy"]),
            "energy": deepcopy(out_def_data["energy_redu"]),
        }
        output_def["force"] = deepcopy(out_def_data["energy_derv_r"])
        output_def["force"].squeeze(-2)
        output_def["virial"] = deepcopy(out_def_data["energy_derv_c_redu"])
        output_def["virial"].squeeze(-2)
        output_def["atom_virial"] = deepcopy(out_def_data["energy_derv_c"])
        output_def["atom_virial"].squeeze(-2)
        if "mask" in out_def_data:
            output_def["mask"] = deepcopy(out_def_data["mask"])
        return output_def

    def model_output_def(self) -> ModelOutputDef:
        """Get the output def for the model."""
        return ModelOutputDef(self.fitting_output_def())

    @classmethod
    def get_model(cls, model_params: dict) -> "MaceModel":
        """Get the model by the parameters.

        Parameters
        ----------
        model_params : dict
            The model parameters

        Returns
        -------
        BaseBaseModel
            The model
        """
        original_model_params = model_params
        model_params_old = model_params.copy()
        model_params = model_params.copy()
        model_params.pop("type", None)
        precision = model_params.pop("precision", "float32")
        if precision == "float32":
            torch.set_default_dtype(torch.float32)
        elif precision == "float64":
            torch.set_default_dtype(torch.float64)
        else:
            msg = f"precision {precision} not supported"
            raise ValueError(msg)
        model = cls(**model_params)
        if model._disable_cueq_for_freeze:
            model_params_old["enable_cueq"] = False
            if isinstance(original_model_params, MutableMapping):
                original_model_params["enable_cueq"] = False
        model.model_def_script = json.dumps(model_params_old)
        return model
