"""Nequip model."""

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
    communicate_extended_output,
)
from deepmd.pt.utils import (
    env,
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
from e3nn.util.jit import (
    script,
)
from nequip.model import model_from_config

from deepmd_gnn import env

@BaseModel.register("nequip")
class NequipModel(BaseModel):
    """Nequip model.

    Parameters
    ----------
    type_map : list[str]
        The name of each type of atoms
    sel : int
        Maximum number of neighbor atoms
    r_max : float, optional
        distance cutoff (in Ang)
    num_layers : int
        number of interaction blocks, we find 3-5 to work best
    l_max : int
        the maximum irrep order (rotation order) for the network's features, l=1 is a good default, l=2 is more accurate but slower
    num_features : int
        the multiplicity of the features, 32 is a good default for accurate network, if you want to be more accurate, go larger, if you want to be faster, go lower
    nonlinearity_type : str
        may be 'gate' or 'norm', 'gate' is recommended
    parity : bool
        whether to include features with odd mirror parityy; often turning parity off gives equally good results but faster networks, so do consider this
    num_basis : int
        number of basis functions used in the radial basis, 8 usually works best
    BesselBasis_trainable : bool
        set true to train the bessel weights
    PolynomialCutoff_p : int
        p-exponent used in polynomial cutoff function, smaller p corresponds to stronger decay with distance
    invariant_layers : int
        number of radial layers, usually 1-3 works best, smaller is faster
    invariant_neurons : int
        number of hidden neurons in radial function, smaller is faster
    use_sc : bool
        use self-connection or not, usually gives big improvement
    irreps_edge_sh : str
        irreps for the chemical embedding of species
    feature_irreps_hidden : str
        irreps used for hidden features, here we go up to lmax=1, with even and odd parities; for more accurate but slower networks, use l=2 or higher, smaller number of features is faster
    chemical_embedding_irreps_out : str
        irreps of the spherical harmonics used for edges. If a single integer, indicates the full SH up to L_max=that_integer
    conv_to_output_hidden_irreps_out : str
        irreps used in hidden layer of output block
    """

    mm_types: list[int]
    e0: torch.Tensor

    def __init__(
        self,
        type_map: list[str],
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
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.params = {
            "type_map": type_map,
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
        self.type_map = type_map
        self.ntypes = len(type_map)
        self.preset_out_bias: dict[str, list] = {"energy": []}
        self.mm_types = []
        self.sel = sel
        self.num_layers = num_layers
        for ii, tt in enumerate(type_map):
            if not tt.startswith("m") and tt not in {"HW", "OW"}:
                self.preset_out_bias["energy"].append(None)
            else:
                self.preset_out_bias["energy"].append([0])
                self.mm_types.append(ii)

        self.rcut = r_max
        self.model = script(
            model_from_config(
                {
                    "model_builders": ["EnergyModel"],
                    "avg_num_neighbors": sel,
                    "chemical_symbols": type_map,
                    "num_types": self.ntypes,
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
                    "model_dtype": precision,
                },
            ),
        )
        self.register_buffer(
            "e0",
            torch.zeros(
                self.ntypes,
                dtype=env.GLOBAL_PT_ENER_FLOAT_PRECISION,
                device=env.DEVICE,
            ),
        )

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
            self.e0 = (
                bias_out["energy"]
                .view(self.e0.shape)
                .to(self.e0.dtype)
                .to(self.e0.device)
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
        if env.DP_GNN_USE_MAPPING:
            return self.rcut
        return self.rcut * self.num_layers

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
            box=box,
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
        # recalculate nlist for ghost atoms
        if mapping is None and self.num_layers > 1 and nloc < nall:
            if env.DP_GNN_USE_MAPPING:
                # when setting DP_GNN_USE_MAPPING, ghost atoms are only built
                # for one message-passing layer
                raise ValueError(
                    "When setting DP_GNN_USE_MAPPING, mapping is required. "
                    "If you are using LAMMPS, set `atom_modify map yes`."
                )
            nlist = build_neighbor_list(
                extended_coord.view(nf, -1),
                extended_atype,
                nall,
                self.rcut * self.num_layers,
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
        do_atomic_virial: bool = False,  # noqa: ARG002
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
        box: Optional[torch.Tensor] = None,
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
        box : torch.Tensor, optional
            The box tensor.
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

        # fake as one frame
        extended_coord_ff = extended_coord.view(nf * nall, 3)
        extended_atype_ff = extended_atype.view(nf * nall)
        edge_index = torch.ops.deepmd_gnn.edge_index(
            nlist,
            extended_atype,
            torch.tensor(self.mm_types, dtype=torch.int64, device="cpu"),
        )
        edge_index = edge_index.T
        # Nequip and MACE have different defination for edge_index
        edge_index = edge_index[[1, 0]]

        # nequip can convert dtype by itself
        default_dtype = torch.float64
        extended_coord_ff = extended_coord_ff.to(default_dtype)
        extended_coord_ff.requires_grad_(True)  # noqa: FBT003

        input_dict = {
            "pos": extended_coord_ff,
            "edge_index": edge_index,
            "atom_types": extended_atype_ff,
        }
        if box is not None and mapping is not None:
            # pass box, map edge index to real
            box_ff = box.to(extended_coord_ff.device)
            input_dict["cell"] = box_ff
            input_dict["pbc"] = torch.zeros(
                3,
                dtype=torch.bool,
                device=box_ff.device,
            )
            batch = torch.arange(nf, device=box_ff.device).repeat(nall)
            input_dict["batch"] = batch
            ptr = torch.arange(
                start=0,
                end=nf * nall + 1,
                step=nall,
                dtype=torch.int64,
                device=batch.device,
            )
            input_dict["ptr"] = ptr
            mapping_ff = mapping.view(nf * nall) + torch.arange(
                0,
                nf * nall,
                nall,
                dtype=mapping.dtype,
                device=mapping.device,
            ).unsqueeze(-1).expand(nf, nall).reshape(-1)
            shifts_atoms = extended_coord_ff - extended_coord_ff[mapping_ff]
            shifts = shifts_atoms[edge_index[1]] - shifts_atoms[edge_index[0]]
            edge_index = mapping_ff[edge_index]
            input_dict["edge_index"] = edge_index
            rec_cell, _ = torch.linalg.inv_ex(box.view(nf, 3, 3))
            edge_cell_shift = torch.einsum(
                "ni,nij->nj",
                shifts,
                rec_cell[batch[edge_index[0]]],
            )
            input_dict["edge_cell_shift"] = edge_cell_shift

        ret = self.model.forward(
            input_dict,
        )

        atom_energy = ret["atomic_energy"]
        if atom_energy is None:
            msg = "atom_energy is None"
            raise ValueError(msg)
        atom_energy = atom_energy.view(nf, nall).to(extended_coord_.dtype)[:, :nloc]
        # adds e0
        atom_energy = atom_energy + self.e0[extended_atype[:, :nloc]].view(
            nf,
            nloc,
        ).to(
            atom_energy.dtype,
        )
        energy = torch.sum(atom_energy, dim=1).view(nf, 1).to(extended_coord_.dtype)
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
        force = force.view(nf, nall, 3).to(extended_coord_.dtype)
        atomic_virial = atomic_virial.view(nf, nall, 1, 9)
        virial = torch.sum(atomic_virial, dim=1).view(nf, 9).to(extended_coord_.dtype)

        return {
            "energy_redu": energy.view(nf, 1),
            "energy_derv_r": force.view(nf, nall, 1, 3),
            "energy_derv_c_redu": virial.view(nf, 1, 9),
            # take the first nloc atoms to match other models
            "energy": atom_energy.view(nf, nloc, 1),
            # fake atom_virial
            "energy_derv_c": atomic_virial.view(nf, nall, 1, 9),
        }

    def serialize(self) -> dict:
        """Serialize the model."""
        return {
            "@class": "Model",
            "@version": 1,
            "type": "mace",
            **self.params,
            "@variables": {
                **{
                    kk: to_numpy_array(vv) for kk, vv in self.model.state_dict().items()
                },
                "e0": to_numpy_array(self.e0),
            },
        }

    @classmethod
    def deserialize(cls, data: dict) -> "NequipModel":
        """Deserialize the model."""
        data = data.copy()
        if not (data.pop("@class") == "Model" and data.pop("type") == "mace"):
            msg = "data is not a serialized NequipModel"
            raise ValueError(msg)
        check_version_compatibility(data.pop("@version"), 1, 1)
        variables = {
            kk: to_torch_tensor(vv) for kk, vv in data.pop("@variables").items()
        }
        model = cls(**data)
        model.e0 = variables.pop("e0")
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
