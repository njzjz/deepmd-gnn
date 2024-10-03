"""Argument check for the MACE model."""

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
    doc_pair_repulsion = "use pair repulsion term with ZBL potential"
    doc_distance_transform = "distance transform"
    doc_correlation = "correlation order at each layer"
    doc_gate = "non linearity for last readout"
    doc_mlp_irreps = "hidden irreps of the MLP in last readout"
    doc_radial_type = "type of radial basis functions"
    doc_radial_mlp = "width of the radial MLP"
    doc_std = "Standard deviation of force components in the training set"
    doc_precision = "Precision of the model, float32 or float64"
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
            Argument("std", float, optional=True, doc=doc_std, default=1),
            Argument(
                "precision",
                str,
                optional=True,
                default="float32",
                doc=doc_precision,
            ),
        ],
        doc="MACE model",
    )


@model_args_plugin.register("nequip")
def nequip_model_args() -> Argument:
    """Arguments for the NequIP model."""
    doc_sel = "Maximum number of neighbor atoms."
    doc_r_max = "distance cutoff (in Ang)"
    doc_num_layers = "number of interaction blocks, we find 3-5 to work best"
    doc_l_max = "the maximum irrep order (rotation order) for the network's features, l=1 is a good default, l=2 is more accurate but slower"
    doc_num_features = "the multiplicity of the features, 32 is a good default for accurate network, if you want to be more accurate, go larger, if you want to be faster, go lower"
    doc_nonlinearity_type = "may be 'gate' or 'norm', 'gate' is recommended"
    doc_parity = "whether to include features with odd mirror parityy; often turning parity off gives equally good results but faster networks, so do consider this"
    doc_num_basis = (
        "number of basis functions used in the radial basis, 8 usually works best"
    )
    doc_besselbasis_trainable = "set true to train the bessel weights"
    doc_polynomialcutoff_p = "p-exponent used in polynomial cutoff function, smaller p corresponds to stronger decay with distance"
    doc_invariant_layers = (
        "number of radial layers, usually 1-3 works best, smaller is faster"
    )
    doc_invariant_neurons = (
        "number of hidden neurons in radial function, smaller is faster"
    )
    doc_use_sc = "use self-connection or not, usually gives big improvement"
    doc_irreps_edge_sh = "irreps for the chemical embedding of species"
    doc_feature_irreps_hidden = "irreps used for hidden features, here we go up to lmax=1, with even and odd parities; for more accurate but slower networks, use l=2 or higher, smaller number of features is faster"
    doc_chemical_embedding_irreps_out = "irreps of the spherical harmonics used for edges. If a single integer, indicates the full SH up to L_max=that_integer"
    doc_conv_to_output_hidden_irreps_out = "irreps used in hidden layer of output block"
    doc_precision = "Precision of the model, float32 or float64"
    return Argument(
        "nequip",
        dict,
        [
            Argument(
                "sel",
                [int, str],
                optional=False,
                doc=doc_sel,
            ),
            Argument(
                "r_max",
                float,
                optional=True,
                default=6.0,
                doc=doc_r_max,
            ),
            Argument(
                "num_layers",
                int,
                optional=True,
                default=4,
                doc=doc_num_layers,
            ),
            Argument(
                "l_max",
                int,
                optional=True,
                default=2,
                doc=doc_l_max,
            ),
            Argument(
                "num_features",
                int,
                optional=True,
                default=32,
                doc=doc_num_features,
            ),
            Argument(
                "nonlinearity_type",
                str,
                optional=True,
                default="gate",
                doc=doc_nonlinearity_type,
            ),
            Argument(
                "parity",
                bool,
                optional=True,
                default=True,
                doc=doc_parity,
            ),
            Argument(
                "num_basis",
                int,
                optional=True,
                default=8,
                doc=doc_num_basis,
            ),
            Argument(
                "BesselBasis_trainable",
                bool,
                optional=True,
                default=True,
                doc=doc_besselbasis_trainable,
            ),
            Argument(
                "PolynomialCutoff_p",
                int,
                optional=True,
                default=6,
                doc=doc_polynomialcutoff_p,
            ),
            Argument(
                "invariant_layers",
                int,
                optional=True,
                default=2,
                doc=doc_invariant_layers,
            ),
            Argument(
                "invariant_neurons",
                int,
                optional=True,
                default=64,
                doc=doc_invariant_neurons,
            ),
            Argument(
                "use_sc",
                bool,
                optional=True,
                default=True,
                doc=doc_use_sc,
            ),
            Argument(
                "irreps_edge_sh",
                str,
                optional=True,
                default="0e + 1e",
                doc=doc_irreps_edge_sh,
            ),
            Argument(
                "feature_irreps_hidden",
                str,
                optional=True,
                default="32x0o + 32x0e + 32x1o + 32x1e",
                doc=doc_feature_irreps_hidden,
            ),
            Argument(
                "chemical_embedding_irreps_out",
                str,
                optional=True,
                default="32x0e",
                doc=doc_chemical_embedding_irreps_out,
            ),
            Argument(
                "conv_to_output_hidden_irreps_out",
                str,
                optional=True,
                default="16x0e",
                doc=doc_conv_to_output_hidden_irreps_out,
            ),
            Argument(
                "precision",
                str,
                optional=True,
                default="float32",
                doc=doc_precision,
            ),
        ],
        doc="Nequip model",
    )
