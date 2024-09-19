# SPDX-License-Identifier: LGPL-3.0-or-later
"""Wrapper for MACE models."""

import json
from copy import deepcopy
from typing import Any, Optional

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
    atomic_virial_corr,
    communicate_extended_output,
)
from deepmd.pt.utils.nlist import (
    build_neighbor_list,
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
from e3nn import (
    o3,
)
from e3nn.util.jit import (
    script,
)
from mace.modules import (
    ScaleShiftMACE,
    gate_dict,
    interaction_classes,
)

import deepmd_mace.op  # noqa: F401

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
    **{ee: ii for ii, ee in enumerate(ELEMENTS)},
    **{f"m{ee}": ii for ii, ee in enumerate(ELEMENTS)},
    "HW": 1,
    "OW": 8,
}


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
        std: Optional[float] = 1,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.params = {
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
        }
        self.type_map = type_map
        self.ntypes = len(type_map)
        self.rcut = r_max
        self.num_interactions = num_interactions
        atomic_numbers = []
        self.preset_out_bias: dict[str, list] = {"energy": []}
        self.mm_types = []
        self.sel = sel
        for ii, tt in enumerate(type_map):
            atomic_numbers.append(PeriodicTable[tt])
            if not tt.startswith("m") and tt not in {"HW", "OW"}:
                self.preset_out_bias["energy"].append(None)
            else:
                self.preset_out_bias["energy"].append([0])
                self.mm_types.append(ii)

        self.model = script(
            ScaleShiftMACE(
                r_max=r_max,
                num_bessel=num_radial_basis,
                num_polynomial_cutoff=num_cutoff_basis,
                max_ell=max_ell,
                interaction_cls=interaction_classes[interaction],
                num_interactions=num_interactions,
                num_elements=self.ntypes,
                hidden_irreps=o3.Irreps(hidden_irreps),
                atomic_energies=torch.zeros(self.ntypes),  # pylint: disable=no-explicit-device,no-explicit-dtype
                avg_num_neighbors=sel,
                atomic_numbers=atomic_numbers,
                pair_repulsion=pair_repulsion,
                distance_transform=distance_transform,
                correlation=correlation,
                gate=gate_dict[gate],
                interaction_cls_first=interaction_classes[
                    "RealAgnosticInteractionBlock"
                ],
                MLP_irreps=o3.Irreps(MLP_irreps),
                atomic_inter_scale=std,
                atomic_inter_shift=0.0,
                radial_MLP=radial_MLP,
                radial_type=radial_type,
            ),
        )
        self.atomic_numbers = atomic_numbers

    def compute_or_load_stat(
        self,
        sampled_func,  # noqa: ANN001
        stat_file_path: Optional[DPPath] = None,
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
        """
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

    @torch.jit.export
    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.rcut * self.num_interactions

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
    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return 0

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
        return False

    @torch.jit.export
    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
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
        model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
        if do_atomic_virial:
            model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-3)
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
        nloc = nlist.shape[1]
        nf, nall = extended_atype.shape
        # calculate nlist for ghost atoms, as LAMMPS does not calculate it
        if mapping is None and self.num_interactions > 1 and nloc < nall:
            nlist = build_neighbor_list(
                extended_coord.view(nf, -1),
                extended_atype,
                nall,
                self.rcut,
                self.sel,
                distinguish_types=False,
            )

        model_ret = self.forward_lower_common(
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
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
        model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
        if do_atomic_virial:
            model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(-3)
        return model_predict

    def forward_lower_common(
        self,
        nloc: int,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
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
        extended_coord = extended_coord.view(nf, nall, 3)
        extended_coord_ = extended_coord
        if fparam is not None:
            msg = "fparam is unsupported"
            raise ValueError(msg)
        if aparam is not None:
            msg = "aparam is unsupported"
            raise ValueError(msg)
        if comm_dict is not None:
            msg = "comm_dict is unsupported"
            raise ValueError(msg)
        nlist = nlist.to(torch.int64)
        extended_atype = extended_atype.to(torch.int64)
        nall = extended_coord.shape[1]

        # loop on nf
        energies = []
        forces = []
        virials = []
        atom_energies = []
        atomic_virials = []
        for ff in range(nf):
            extended_coord_ff = extended_coord[ff]
            extended_atype_ff = extended_atype[ff]
            nlist_ff = nlist[ff]
            edge_index = torch.ops.deepmd_mace.mace_edge_index(
                nlist_ff,
                extended_atype_ff,
                torch.tensor(self.mm_types, dtype=torch.int64, device="cpu"),
            )
            edge_index = edge_index.T
            # to one hot
            indices = extended_atype_ff.unsqueeze(-1)
            oh = torch.zeros(
                (nall, self.ntypes),
                device=extended_atype.device,
                dtype=torch.float64,
            )
            # scatter_ is the in-place version of scatter
            oh.scatter_(dim=-1, index=indices, value=1)
            one_hot = oh.view((nall, self.ntypes))

            # cast to float32
            default_dtype = self.model.atomic_energies_fn.atomic_energies.dtype
            extended_coord_ff = extended_coord_ff.to(default_dtype)
            extended_coord_ff.requires_grad_(True)  # noqa: FBT003
            nedge = edge_index.shape[1]
            if self.num_interactions > 1 and mapping is not None and nloc < nall:
                # shift the edges for ghost atoms, and map the ghost atoms to real atoms
                shifts_atoms = extended_coord_ff - extended_coord_ff[mapping[ff]]
                shifts = shifts_atoms[edge_index[1]] - shifts_atoms[edge_index[0]]
                edge_index = mapping[ff][edge_index]
            else:
                shifts = torch.zeros(
                    (nedge, 3),
                    dtype=torch.float64,
                    device=extended_coord_.device,
                )
            shifts = shifts.to(default_dtype)
            one_hot = one_hot.to(default_dtype)
            # it seems None is not allowed for data
            box = (
                torch.eye(
                    3,
                    dtype=extended_coord_ff.dtype,
                    device=extended_coord_ff.device,
                )
                * 1000.0
            )

            ret = self.model.forward(
                {
                    "positions": extended_coord_ff,
                    "shifts": shifts,
                    "cell": box,
                    "edge_index": edge_index,
                    "batch": torch.zeros(
                        [nall],
                        dtype=torch.int64,
                        device=extended_coord_ff.device,
                    ),
                    "node_attrs": one_hot,
                    "ptr": torch.tensor(
                        [0, nall],
                        dtype=torch.int64,
                        device=extended_coord_ff.device,
                    ),
                    "weight": torch.tensor(
                        [1.0],
                        dtype=extended_coord_ff.dtype,
                        device=extended_coord_ff.device,
                    ),
                },
                compute_force=False,
                compute_virials=False,
                compute_stress=False,
                compute_displacement=False,
                training=self.training,
            )

            atom_energy = ret["node_energy"]
            if atom_energy is None:
                msg = "atom_energy is None"
                raise ValueError(msg)
            atom_energy = atom_energy.view(1, nall).to(extended_coord_.dtype)[:, :nloc]
            energy = torch.sum(atom_energy, dim=1).view(1, 1).to(extended_coord_.dtype)
            grad_outputs: list[Optional[torch.Tensor]] = [
                torch.ones_like(energy),
            ]
            force = torch.autograd.grad(
                outputs=[energy],
                inputs=[extended_coord_ff],
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=self.training,
            )[0]
            if force is None:
                msg = "force is None"
                raise ValueError(msg)
            force = -force
            atomic_virial = force.unsqueeze(-1).to(
                extended_coord_.dtype,
            ) @ extended_coord_ff.unsqueeze(-2).to(
                extended_coord_.dtype,
            )
            if do_atomic_virial:
                extended_virial_corr = atomic_virial_corr(
                    extended_coord_ff.unsqueeze(0),
                    atom_energy,
                )
                atomic_virial = atomic_virial + extended_virial_corr
            force = force.view(1, nall, 3).to(extended_coord_.dtype)
            virial = (
                torch.sum(atomic_virial, dim=0).view(1, 9).to(extended_coord_.dtype)
            )

            energies.append(energy)
            forces.append(force)
            virials.append(virial)
            atom_energies.append(atom_energy)
            atomic_virials.append(atomic_virial)
        energies_t = torch.cat(energies, dim=0)
        forces_t = torch.cat(forces, dim=0)
        virials_t = torch.cat(virials, dim=0)
        atom_energies_t = torch.cat(atom_energies, dim=0)
        atomic_virials_t = torch.cat(atomic_virials, dim=0)

        return {
            "energy_redu": energies_t.view(nf, 1),
            "energy_derv_r": forces_t.view(nf, nall, 1, 3),
            "energy_derv_c_redu": virials_t.view(nf, 1, 9),
            # take the first nloc atoms to match other models
            "energy": atom_energies_t.view(nf, nloc, 1),
            # fake atom_virial
            "energy_derv_c": atomic_virials_t.view(nf, nall, 1, 9),
        }

    def serialize(self) -> dict:
        """Serialize the model."""
        return {
            "@class": "Model",
            "@version": 1,
            "type": "mace",
            **self.params,
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
        output_def["atom_virial"].squeeze(-3)
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
        model.model_def_script = json.dumps(model_params_old)
        return model
