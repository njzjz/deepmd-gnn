"""Nequip model."""

from copy import deepcopy
from typing import Any

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
from nequip.data import (
    AtomicDataDict,
)
from nequip.model import model_from_config
from nequip.nn import (
    GraphModel,
    GraphModuleMixin,
)
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

import deepmd_gnn.op  # noqa: F401
from deepmd_gnn.autograd import derive_atomic_virial_from_displacement
from deepmd_gnn.deepmd_ops import ensure_border_op_placeholder
from deepmd_gnn.edge import dense_edge_index
from deepmd_gnn.export import pad_nlist_for_export as _pad_nlist_for_export
from deepmd_gnn.stat_compat import load_observed_type_stat_compat

ensure_border_op_placeholder()


_COMM_SEND_LIST_KEY = "_deepmd_gnn_send_list"
_COMM_SEND_PROC_KEY = "_deepmd_gnn_send_proc"
_COMM_RECV_PROC_KEY = "_deepmd_gnn_recv_proc"
_COMM_SEND_NUM_KEY = "_deepmd_gnn_send_num"
_COMM_RECV_NUM_KEY = "_deepmd_gnn_recv_num"
_COMM_COMMUNICATOR_KEY = "_deepmd_gnn_communicator"
_COMM_NLOC_KEY = "_deepmd_gnn_nloc"
_COMM_NGHOST_KEY = "_deepmd_gnn_nghost"
_COMM_KEYS = [
    _COMM_SEND_LIST_KEY,
    _COMM_SEND_PROC_KEY,
    _COMM_RECV_PROC_KEY,
    _COMM_SEND_NUM_KEY,
    _COMM_RECV_NUM_KEY,
    _COMM_COMMUNICATOR_KEY,
    _COMM_NLOC_KEY,
    _COMM_NGHOST_KEY,
]


class _DeepMDBorderCommunication(GraphModuleMixin, torch.nn.Module):
    """Communicate NequIP node features between message-passing layers."""

    def __init__(self, irreps_in: dict[str, Any]) -> None:
        super().__init__()
        irreps_with_comm = dict(irreps_in)
        for key in _COMM_KEYS:
            irreps_with_comm[key] = None
        self._init_irreps(irreps_in=irreps_with_comm, irreps_out={})

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Communicate local node features to ghost slots when comm data is present."""
        if "_deepmd_gnn_nloc" not in data:
            return data

        nloc_tensor = data["_deepmd_gnn_nloc"]
        nghost_tensor = data["_deepmd_gnn_nghost"]
        nloc = int(nloc_tensor.item())
        nghost = int(nghost_tensor.item())

        node_feats = data[AtomicDataDict.NODE_FEATURES_KEY][:nloc]
        if nghost > 0:
            node_feats = torch.nn.functional.pad(
                node_feats,
                (0, 0, 0, nghost),
                value=0.0,
            )

        ret = torch.ops.deepmd.border_op(
            data["_deepmd_gnn_send_list"],
            data["_deepmd_gnn_send_proc"],
            data["_deepmd_gnn_recv_proc"],
            data["_deepmd_gnn_send_num"],
            data["_deepmd_gnn_recv_num"],
            node_feats,
            data["_deepmd_gnn_communicator"],
            nloc_tensor,
            nghost_tensor,
        )
        data[AtomicDataDict.NODE_FEATURES_KEY] = ret[0]
        return data


def _insert_border_communication_modules(model: GraphModel, num_layers: int) -> None:
    """Insert DeePMD border communication after NequIP convolution layers."""
    if num_layers <= 1:
        return

    graph = model.model
    for layer_idx in range(num_layers - 1):
        conv_name = f"layer{layer_idx}_convnet"
        comm_name = f"layer{layer_idx}_deepmd_comm"
        conv_module = graph._modules[conv_name]  # noqa: SLF001
        graph.insert(
            name=comm_name,
            module=_DeepMDBorderCommunication(conv_module.irreps_out),
            after=conv_name,
        )
    for key in _COMM_KEYS:
        if key not in model.model_input_fields:
            model.model_input_fields.append(key)


def _make_nequip_network(params: dict[str, Any], ntypes: int) -> GraphModel:
    nequip_model = model_from_config(
        {
            "model_builders": ["EnergyModel"],
            "avg_num_neighbors": params["sel"],
            "chemical_symbols": params["type_map"],
            "num_types": ntypes,
            "r_max": params["r_max"],
            "num_layers": params["num_layers"],
            "l_max": params["l_max"],
            "num_features": params["num_features"],
            "nonlinearity_type": params["nonlinearity_type"],
            "parity": params["parity"],
            "num_basis": params["num_basis"],
            "BesselBasis_trainable": params["BesselBasis_trainable"],
            "PolynomialCutoff_p": params["PolynomialCutoff_p"],
            "invariant_layers": params["invariant_layers"],
            "invariant_neurons": params["invariant_neurons"],
            "use_sc": params["use_sc"],
            "irreps_edge_sh": params["irreps_edge_sh"],
            "feature_irreps_hidden": params["feature_irreps_hidden"],
            "chemical_embedding_irreps_out": params["chemical_embedding_irreps_out"],
            "conv_to_output_hidden_irreps_out": params[
                "conv_to_output_hidden_irreps_out"
            ],
            "model_dtype": params["precision"],
        },
    )
    _insert_border_communication_modules(nequip_model, params["num_layers"])
    return nequip_model


(
    _restore_observed_type_from_file,
    _save_observed_type_to_file,
    collect_observed_types,
) = load_observed_type_stat_compat()


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
    _observed_type: list[str] | None

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
        self._observed_type = None
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
        nequip_model = _make_nequip_network(self.params, self.ntypes)
        self.model = script(nequip_model.to(env.DEVICE))
        self.register_buffer(
            "e0",
            torch.zeros(
                self.ntypes,
                dtype=env.GLOBAL_PT_ENER_FLOAT_PRECISION,
                device=env.DEVICE,
            ),
        )

    @property
    def atomic_model(self) -> Any:  # noqa: ANN401
        """Provide a compatibility view matching wrapped deepmd-kit models."""
        return self

    @property
    def observed_type(self) -> list[str] | None:
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
        stat_file_path: DPPath | None = None,
        preset_observed_type: list[str] | None = None,
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

    def get_default_fparam(self) -> torch.Tensor | None:
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

    def get_default_chg_spin(self) -> torch.Tensor | None:
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
        return self.num_layers > 1

    @torch.jit.export
    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
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
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        charge_spin: torch.Tensor | None = None,
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
        if (
            self.num_layers > 1
            and nloc < nall
            and mapping is None
            and comm_dict is None
        ):
            msg = (
                "Multi-layer NequIP lower inference requires either comm_dict "
                "from DeePMD-kit message-passing communication or a mapping "
                "from extended atoms to local atoms."
            )
            raise ValueError(msg)
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
            model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(-2)
        return model_predict

    def forward_lower_common(
        self,
        nloc: int,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        box: torch.Tensor | None = None,
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
        nlist = nlist.to(torch.int64)
        extended_atype = extended_atype.to(torch.int64)
        nall = extended_coord.shape[1]

        extended_coord_ff = extended_coord.view(nf * nall, 3)
        extended_atype_ff = extended_atype.view(nf * nall)
        edge_mask = torch.jit.annotate(torch.Tensor | None, None)
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
        # Nequip and MACE have different defination for edge_index
        edge_index = edge_index[[1, 0]]

        # nequip can convert dtype by itself
        default_dtype = torch.float64
        extended_coord_grad = extended_coord.to(default_dtype)
        extended_coord_grad.requires_grad_(requires_grad=True)
        extended_coord_ff = extended_coord_grad.view(nf * nall, 3)

        input_dict: dict[str, torch.Tensor] = {
            "edge_index": edge_index,
            "atom_types": extended_atype_ff,
        }
        nedge = edge_index.shape[1]
        shifts = torch.zeros(
            (nedge, 3),
            dtype=default_dtype,
            device=extended_coord_ff.device,
        )
        has_mapping = False
        if comm_dict is None and mapping is not None and nloc < nall:
            has_mapping = True
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
        if edge_mask is not None:
            edge_mask = edge_mask.to(device=shifts.device)
            far_shifts = torch.zeros_like(shifts)
            far_shifts[:, 0] = self.rcut * 10.0 + 10.0
            shifts = torch.where(edge_mask.unsqueeze(-1), shifts, far_shifts)
        if comm_dict is not None:
            if nf != 1:
                msg = "NequIP comm_dict lower inference only supports one frame"
                raise ValueError(msg)
            input_dict["_deepmd_gnn_send_list"] = comm_dict["send_list"]
            input_dict["_deepmd_gnn_send_proc"] = comm_dict["send_proc"]
            input_dict["_deepmd_gnn_recv_proc"] = comm_dict["recv_proc"]
            input_dict["_deepmd_gnn_send_num"] = comm_dict["send_num"]
            input_dict["_deepmd_gnn_recv_num"] = comm_dict["recv_num"]
            input_dict["_deepmd_gnn_communicator"] = comm_dict["communicator"]
            input_dict["_deepmd_gnn_nloc"] = torch.tensor(
                nloc,
                dtype=torch.int32,
                device=torch.device("cpu"),
            )
            input_dict["_deepmd_gnn_nghost"] = torch.tensor(
                nall - nloc,
                dtype=torch.int32,
                device=torch.device("cpu"),
            )

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

        compute_displacement = box is not None
        displacement = torch.jit.annotate(torch.Tensor | None, None)
        if box is not None:
            box_tensor = (
                box.view(nf, 3, 3).to(default_dtype).to(extended_coord_ff.device)
            )
            input_dict["batch"] = batch
            input_dict["ptr"] = ptr
            input_dict["pbc"] = torch.zeros(
                3,
                dtype=torch.bool,
                device=extended_coord_ff.device,
            )
            edge_batch = torch.div(edge_index[0], nall, rounding_mode="floor")
            inv_box = torch.linalg.inv(box_tensor)
            edge_cell_shift = torch.einsum("ec,ecb->eb", shifts, inv_box[edge_batch])
            displacement = torch.zeros(
                (nf, 3, 3),
                dtype=extended_coord_ff.dtype,
                device=extended_coord_ff.device,
            )
            displacement.requires_grad_(requires_grad=True)
            symmetric_displacement = 0.5 * (
                displacement + displacement.transpose(-1, -2)
            )
            input_dict["pos"] = extended_coord_ff + torch.einsum(
                "be,bec->bc",
                extended_coord_ff,
                symmetric_displacement[batch],
            )
            input_dict["cell"] = box_tensor + torch.matmul(
                box_tensor,
                symmetric_displacement,
            )
            input_dict["edge_cell_shift"] = edge_cell_shift
        else:
            input_dict["pos"] = extended_coord_ff
            if edge_mask is not None or has_mapping:
                input_dict["batch"] = batch
                input_dict["ptr"] = ptr
                input_dict["cell"] = (
                    torch.eye(
                        3,
                        dtype=extended_coord_ff.dtype,
                        device=extended_coord_ff.device,
                    )
                    .unsqueeze(0)
                    .expand(nf, 3, 3)
                )
                input_dict["edge_cell_shift"] = shifts

        ret = self.model.forward(
            input_dict,
        )

        atom_energy_all = ret["atomic_energy"]
        if atom_energy_all is None:
            msg = "atom_energy is None"
            raise ValueError(msg)
        atom_energy_all = atom_energy_all.view(nf, nall)
        atom_energy = atom_energy_all[:, :nloc]
        # adds e0
        atom_energy = atom_energy + self.e0[extended_atype[:, :nloc]].view(
            nf,
            nloc,
        ).to(
            atom_energy.dtype,
        )
        energy = torch.sum(atom_energy, dim=1)
        grad_outputs = torch.jit.annotate(
            list[torch.Tensor | None],
            [torch.ones_like(energy)],
        )
        retain_graph = self.training or do_atomic_virial

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
            atomic_virial_fallback = force.unsqueeze(-1) @ extended_coord_ff.view(
                nf,
                nall,
                3,
            ).unsqueeze(-2)
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

    def forward_common_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
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
            comm_dict=None,
        )

    def forward_common_lower_exportable(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        **make_fx_kwargs: object,
    ) -> torch.nn.Module:
        """Trace ``forward_common_lower`` for ``torch.export`` serialization."""
        make_fx_kwargs = make_fx_kwargs.copy()
        if make_fx_kwargs.get("tracing_mode") == "symbolic":
            make_fx_kwargs["tracing_mode"] = "real"
            make_fx_kwargs.pop("_allow_non_fake_inputs", None)
        model = self
        scripted_model = self.model
        raw_model = _make_nequip_network(self.params, self.ntypes).to(self.e0.device)
        raw_model.load_state_dict(scripted_model.state_dict())
        raw_model.train(scripted_model.training)

        def fn(
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            nlist: torch.Tensor,
            mapping: torch.Tensor | None,
            fparam: torch.Tensor | None,
            aparam: torch.Tensor | None,
            charge_spin: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
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

        self._use_exportable_edge_index = True
        self.model = raw_model
        try:
            return make_fx(fn, **make_fx_kwargs)(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam,
                aparam,
                charge_spin,
            )
        finally:
            self.model = scripted_model
            self._use_exportable_edge_index = False

    def serialize(self) -> dict:
        """Serialize the model."""
        return {
            "@class": "Model",
            "@version": 1,
            "type": "nequip",
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
        if not (data.pop("@class") == "Model" and data.pop("type") == "nequip"):
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
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
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
