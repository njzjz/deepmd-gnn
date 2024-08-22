"""Argument check for the MACE model."""

from __future__ import annotations

from dargs import Argument
from deepmd.utils.argcheck import model_args_plugin


@model_args_plugin.register("mace")
def mace_model_args() -> Argument:
    """Arguments for the MACE model.

    Returns
    -------
    Argument
        Arguments for the MACE model.
    """
    doc_r_max = "distance cutoff (in Ang)"
    doc_num_radial_basis = "number of radial basis functions"
    doc_num_cutoff_basis = "number of basis functions for smooth cutoff"
    doc_max_ell = "highest ell of spherical harmonics"
    doc_interaction = "name of interaction block"
    doc_num_interactions = "number of interactions"
    doc_hidden_irreps = "hidden irreps"
    doc_pair_repulsion = "use amsgrad variant of optimizer"
    doc_distance_transform = "distance transform"
    doc_correlation = "correlation order at each layer"
    doc_gate = "non linearity for last readout"
    doc_mlp_irreps = "hidden irreps of the MLP in last readout"
    doc_radial_type = "type of radial basis functions"
    doc_radial_mlp = "width of the radial MLP"
    doc_std = "Standard deviation of force components in the training set"
    return Argument(
        "mace",
        dict,
        [
            Argument("sel", [int, str], optional=False),
            Argument("r_max", float, optional=True, default=5.0, doc=doc_r_max),
            Argument(
                "num_radial_basis",
                int,
                optional=True,
                default=8,
                doc=doc_num_radial_basis,
            ),
            Argument(
                "num_cutoff_basis",
                int,
                optional=True,
                default=5,
                doc=doc_num_cutoff_basis,
            ),
            Argument("max_ell", int, optional=True, default=3, doc=doc_max_ell),
            Argument(
                "interaction",
                str,
                optional=True,
                default="RealAgnosticResidualInteractionBlock",
                doc=doc_interaction,
            ),
            Argument(
                "num_interactions",
                int,
                optional=True,
                default=2,
                doc=doc_num_interactions,
            ),
            Argument(
                "hidden_irreps",
                str,
                optional=True,
                default="128x0e + 128x1o",
                doc=doc_hidden_irreps,
            ),
            Argument(
                "pair_repulsion",
                bool,
                optional=True,
                default=False,
                doc=doc_pair_repulsion,
            ),
            Argument(
                "distance_transform",
                str,
                optional=True,
                default="None",
                doc=doc_distance_transform,
            ),
            Argument("correlation", int, optional=True, default=3, doc=doc_correlation),
            Argument("gate", str, optional=True, default="silu", doc=doc_gate),
            Argument(
                "MLP_irreps",
                str,
                optional=True,
                default="16x0e",
                doc=doc_mlp_irreps,
            ),
            Argument(
                "radial_type",
                str,
                optional=True,
                default="bessel",
                doc=doc_radial_type,
            ),
            Argument(
                "radial_MLP",
                list[int],
                optional=True,
                default=[64, 64, 64],
                doc=doc_radial_mlp,
            ),
            Argument("std", float, optional=True, doc=doc_std),
        ],
        doc="MACE model",
    )
