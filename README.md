# DeePMD-kit plugin for various graph neural network models

[![DOI:10.1021/acs.jcim.4c02441](https://img.shields.io/badge/DOI-10.1021%2Facs.jcim.4c02441-blue)](https://doi.org/10.1021/acs.jcim.4c02441)
[![Citations](https://citations.njzjz.win/10.1021/acs.jcim.4c02441)](https://doi.org/10.1021/acs.jcim.4c02441)
[![conda install](https://img.shields.io/conda/dn/conda-forge/deepmd-gnn?label=conda%20install)](https://anaconda.org/conda-forge/deepmd-gnn)
[![PyPI - Version](https://img.shields.io/pypi/v/deepmd-gnn)](https://pypi.org/p/deepmd-gnn)

`deepmd-gnn` is a [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) plugin for various graph neural network (GNN) models, which connects DeePMD-kit and atomistic GNN packages by enabling GNN models in DeePMD-kit.

Supported packages and models include:

- [MACE](https://github.com/ACEsuit/mace) (PyTorch version)
- [NequIP](https://github.com/mir-group/nequip) (PyTorch version)

After [installing the plugin](#installation), you can train the GNN models using DeePMD-kit, run active learning cycles for the GNN models using [DP-GEN](https://github.com/deepmodeling/dpgen), and perform simulations with MACE and NequIP models using molecular dynamic packages supported by DeePMD-kit, such as [LAMMPS](https://github.com/lammps/lammps) and [AMBER](https://ambermd.org/).
You can follow [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/) to train the GNN models using its PyTorch backend, after using the specific [model parameters](#parameters).

## Highlights

The official MACE and NequIP packages remain the reference implementations of
their model architectures. DeePMD-GNN focuses on what is useful beyond the
standalone official workflows: bringing those equivariant GNN models into the
DeePMD-kit ecosystem for training, active learning, and production molecular
simulations.

- **Unified DeePMD-kit workflow**: train and freeze MACE or NequIP models with
  `dp --pt train` and `dp --pt freeze`, using DeePMD-kit data pipelines,
  inputs, losses, and model serialization instead of maintaining a separate
  workflow for each upstream package.
- **Broader MD engine access**: run frozen GNN models through molecular dynamics
  interfaces supported by DeePMD-kit, including LAMMPS and AMBER, rather than
  relying only on the engines and plugins maintained by the official MACE or
  NequIP projects.
- **Parallel periodic simulations**: use DeePMD-kit's message-passing
  communication path for LAMMPS/MPI runs. MACE and NequIP models advertise
  their message passing to DeePMD-kit, so ghost-atom features can be exchanged
  between message-passing layers through `border_op`.
- **Active learning and QM/MM integration**: plug MACE or NequIP models into
  DP-GEN active learning loops and DeePMD-kit DPRc/AmberTools workflows through
  the same interface used by other DeePMD-kit models.
- **Deployment-oriented MACE extras**: export MACE models with the
  DeePMD-kit/LAMMPS PyTorch exportable backend (`pt_expt`), train with optional
  cuEquivariance acceleration, use DeePMD-kit's `torch.compile` path, and
  conservatively convert selected official MACE-OFF checkpoints for downstream
  validation in DeePMD-kit-supported engines.

## Credits

If you use this software, please cite the following paper:

- Jinzhe Zeng, Timothy J. Giese, Duo Zhang, Han Wang, Darrin M. York, DeePMD-GNN: A DeePMD-kit Plugin for External Graph Neural Network Potentials, _J. Chem. Inf. Model._, 2025, 65, 7, 3154-3160, DOI: [10.1021/acs.jcim.4c02441](https://doi.org/10.1021/acs.jcim.4c02441). [![Citations](https://citations.njzjz.win/10.1021/acs.jcim.4c02441)](https://badge.dimensions.ai/details/doi/10.1021/acs.jcim.4c02441)

## Installation

### Install via conda

If you are in a [conda environment](https://docs.deepmodeling.com/faq/conda.html) where DeePMD-kit is already installed from the conda-forge channel,
you can use `conda` to install the DeePMD-GNN plugin:

```sh
conda install deepmd-gnn -c conda-forge
```

### Build from source

First, clone this repository:

```sh
git clone https://gitlab.com/RutgersLBSR/deepmd-gnn
cd deepmd-gnn
```

#### Python interface plugin

Python 3.10 or above is required. A C++ compiler that supports C++ 17 is required.
NVCC is required to build CUDA OPs.

```sh
pip install .
```

Only PyTorch 2.10 or above is supported.

#### C++ interface plugin

DeePMD-kit version should be v3.0.0b4 or later.

Follow [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/install/install-from-source.html#install-the-c-interface) to install DeePMD-kit C++ interface with PyTorch backend support and other related MD packages.
After that, you can build the plugin

```sh
# Assume libtorch has been contained in CMAKE_PREFIX_PATH
mkdir -p build
cd build
cmake .. -D CMAKE_INSTALL_PREFIX=/prefix/to/install
cmake --build . -j8
cmake --install .
```

`libdeepmd_gnn.so` will be installed into the directory you assign.
When using any DeePMD-kit C++ interface, set the following environment variable in advance:

```sh
export DP_PLUGIN_PATH=/prefix/to/install/lib/libdeepmd_gnn.so
```

## Usage

Follow [Parameters section](#parameters) to prepare a DeePMD-kit input file.

```sh
dp --pt train input.json
dp --pt freeze
```

A frozen model file named `frozen_model.pth` will be generated. You can use it in the MD packages or other interfaces.
For details, follow [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/).

### Pretrained MACE descriptors for properties

The regular PyTorch backend can use a trusted native
`mace.modules.ScaleShiftMACE` checkpoint as the descriptor of a standard
DeePMD property model. See
[`examples/property/mace/input.json`](examples/property/mace/input.json):

```json
"model": {
  "type": "standard",
  "type_map": ["H", "O"],
  "descriptor": {
    "type": "mace",
    "model_path": "./trusted_single_head_mace_mp_or_mpa.model",
    "sel": 64,
    "trainable": true
  },
  "fitting_net": {
    "type": "property",
    "property_name": "band_gap",
    "task_dim": 1
  }
}
```

`model_path` uses Python pickle loading, so only load checkpoints from a
trusted source. Neighbor statistics persist the inferred backbone architecture
into the saved DeePMD checkpoint, so `dp test` and freeze no longer need the
original `.model` file. The model-level `type_map` must exactly match the
checkpoint's atomic-number ordering; reordered or subset maps are rejected.
`sel` is an explicit DeePMD neighbor-list capacity and cannot be inferred from
MACE. Official single-head MACE-MP-0/MPA-0 89-element checkpoints therefore
require:

```text
["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg",
 "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
 "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
 "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
 "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
 "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
 "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au",
 "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac"]
```

The descriptor returns only invariant even-scalar (`0e`) channels from the
final MACE product layer. Its embedding, interaction, and product weights are
trainable by default together with the DeePMD `PropertyFittingNet`; MACE atomic
energies, energy readouts, scale/shift, and pair-repulsion terms do not seed
the property head. Property-label mean and standard deviation remain DeePMD
statistics. Generic single-head native `ScaleShiftMACE` architectures are
reconstructed from the checkpoint, including independent first/later
interaction classes, layer-dependent correlation, named heads, pair
repulsion, and mixed final irreps. This covers compatible MACE-MP-0 and MPA-0
checkpoints without changing the separate conservative MACE-OFF energy-model
conversion path.

This descriptor path currently supports only `dp --pt`. The `pt_expt`
backend, cuEquivariance checkpoint conversion, MACE-MH/multi-head checkpoints,
type-map subsets, MPI descriptor communication, LAMMPS deployment, and DPRc
descriptor semantics are out of scope.

### Exporting MACE models with the PyTorch exportable backend

MACE models can also be trained and frozen with DeePMD-kit's PyTorch exportable
backend (`pt_expt`):

```sh
dp --pt-expt train input.json
dp --pt-expt freeze -o frozen_model.pt2
```

Use this path when you need a `.pt2` model for the DeePMD-kit/LAMMPS
PyTorch exportable runtime. The `pt_expt` path currently supports MACE models;
NequIP models should still use the regular `--pt` workflow.

The `pt_expt` workflow requires DeePMD-kit 3.2.0 or later. Multi-layer MACE
models include the extra communication artifact needed by LAMMPS/MPI inside the
exported `.pt2` package, so the resulting file can be passed to LAMMPS in the
same way as other DeePMD-kit frozen models.

#### Training MACE with cuEquivariance

MACE training can use cuEquivariance kernels when the installed MACE package
provides `CuEquivarianceConfig` and the cuEquivariance runtime packages are
available. Install the matching extra for the CUDA runtime used by PyTorch:

```sh
pip install "deepmd-gnn[cueq]"
```

The `cueq` extra installs the CUDA 12 cuEquivariance op package. For CUDA 11
environments, use `deepmd-gnn[cueq-cu11]` instead. When installing from a
source checkout, use `pip install ".[cueq]"` or `pip install ".[cueq-cu11]"`.

Then enable it in the `model` section:

```json
"model": {
  "type": "mace",
  "enable_cueq": true
}
```

Then train with either PyTorch backend:

```sh
dp --pt train input.json
dp --pt-expt train input.json
```

On a single RTX 5090, using the water example data with `sel = "auto"`,
`hidden_irreps = "128x0e + 128x1o"`, `batch_size = 1`, and
`disp_freq = 100`, 2000-step runs were used to sample steady-state training
speed:

| backend        | cuEquivariance | steady avg after step 300 | speedup |
| -------------- | -------------: | ------------------------: | ------: |
| `dp --pt`      |             no |             0.1955 s/step |   1.00x |
| `dp --pt`      |            yes |             0.1061 s/step |   1.84x |
| `dp --pt-expt` |             no |             0.2025 s/step |   1.00x |
| `dp --pt-expt` |            yes |             0.1007 s/step |   2.01x |

The cuEquivariance effect is similar in both backends, but not bit-for-bit
identical: in this run, the steady cuEquivariance speed differed by about 5%.
The first cuEquivariance step includes a one-time kernel initialization cost
(about 56 s in this run), which is amortized over long training jobs.

cuEquivariance accelerates training only. When freezing a checkpoint whose
input had `"enable_cueq": true`, deepmd-gnn disables cuEquivariance for the
frozen model and converts the MACE weights back to the e3nn format. This lets
both `dp --pt freeze` and `dp --pt-expt freeze` export normally, but the frozen
model does not use cuEquivariance kernels.

#### Training MACE with `torch.compile`

MACE training through the `pt_expt` backend can use DeePMD-kit's native
`torch.compile` path. Enable it in the `training` section:

```json
"training": {
  "enable_compile": true
}
```

Then run the regular exportable-backend training command:

```sh
dp --pt-expt train input.json
```

On a single RTX 5090, using the water example data with `sel = "auto"`,
`hidden_irreps = "128x0e + 128x1o"`, `batch_size = 1`, and
`disp_freq = 100`, a 2000-step run was used to sample the steady-state training
speed:

| mode            | steady avg after step 200 | speedup |
| --------------- | ------------------------: | ------: |
| eager `pt_expt` |             0.2030 s/step |   1.00x |
| `torch.compile` |             0.1495 s/step |   1.36x |

The first compiled step includes a one-time Inductor trace/compile cost
(97.17 s in this run), which is amortized over normal long training jobs.

Do not enable `enable_cueq` and `enable_compile` together for now. With
cuEquivariance enabled, DeePMD-kit's symbolic `make_fx` compile trace fails in
the cuEquivariance `uniform_1d` fake-tensor path with a data-dependent symbolic
shape guard.

### Running LAMMPS + GNN models with period boundary conditions

GNN models use message passing neural networks,
so the neighbor list built with traditional cutoff radius will not work,
since the ghost atoms also need to build neighbor list.
The MACE and NequIP models work like regular DeePMD-kit message-passing models.
No extra environment variable is needed when freezing the model.
They advertise message passing to DeePMD-kit, so LAMMPS can use the regular
neighbor list with a cutoff radius of $r_c$ and DeePMD-kit communicates
the ghost-atom features through MPI between message-passing layers.
This requires a DeePMD-kit build whose LAMMPS/PyTorch interface provides
the message-passing communication op, `border_op`.

When `border_op` communication is available, DeePMD-kit/LAMMPS uses that
MPI-capable path. The atom-map path is a serial fallback for runs that do not
receive a DeePMD message-passing `comm_dict`; it maps ghost atoms to their
corresponding real atoms on the same process and is not a substitute for MPI
communication. Request the mapping when using this fallback in LAMMPS
(also requires DeePMD-kit v3.0.0rc0 or above).

```lammps
atom_modify map array
```

## Parameters

### MACE

To use the MACE model, set `"type": "mace"` in the `model` section of the training script.
Below is default values for the MACE model, most of which follows default values in the MACE package:

```json
"model": {
  "type": "mace",
  "type_map": [
    "O",
    "H"
  ],
  "r_max": 5.0,
  "sel": "auto",
  "num_radial_basis": 8,
  "num_cutoff_basis": 5,
  "max_ell": 3,
  "interaction": "RealAgnosticResidualInteractionBlock",
  "num_interactions": 2,
  "hidden_irreps": "128x0e + 128x1o",
  "pair_repulsion": false,
  "distance_transform": "None",
  "correlation": 3,
  "gate": "silu",
  "MLP_irreps": "16x0e",
  "radial_type": "bessel",
  "radial_MLP": [64, 64, 64],
  "std": 1.0,
  "precision": "float32",
  "enable_cueq": false
}
```

### NequIP

```json
"model": {
  "type": "nequip",
  "type_map": [
    "O",
    "H"
  ],
  "r_max": 5.0,
  "sel": "auto",
  "num_layers": 4,
  "l_max": 2,
  "num_features": 32,
  "nonlinearity_type": "gate",
  "parity": true,
  "num_basis": 8,
  "BesselBasis_trainable": true,
  "PolynomialCutoff_p": 6,
  "invariant_layers": 2,
  "invariant_neurons": 64,
  "use_sc": true,
  "irreps_edge_sh": "0e + 1e",
  "feature_irreps_hidden": "32x0o + 32x0e + 32x1o + 32x1e",
  "chemical_embedding_irreps_out": "32x0e",
  "conv_to_output_hidden_irreps_out": "16x0e",
  "precision": "float32"
}
```

## DPRc support

In `deepmd-gnn`, the GNN model can be used in a [DPRc](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dprc.html) way.
Type maps that starts with `m` (such as `mH`) or `OW` or `HW` will be recognized as MM types.
Two MM atoms will not build edges with each other.
Such GNN+DPRc model can be directly used in AmberTools24.

## Conservative MACE-OFF checkpoint loading

`deepmd_gnn.mace_off` provides a conservative bridge from selected official
MACE-OFF checkpoints into DeePMD-GNN. The goal is to recover the parts that are
really stored in the checkpoint while avoiding guesses about wrapper-only
runtime semantics.

In practice, this path intentionally does less inference than the broader
DPRc/QM/MM machinery:

- it infers ordinary element `type_map` entries from checkpoint
  `atomic_numbers`
- it restores checkpoint-side model semantics such as `avg_num_neighbors` from
  the original `ScaleShiftMACE` object instead of reusing DeePMD's runtime
  neighbor cap
- it requires an explicit `sel` value, because `sel` is a DeePMD runtime
  neighbor-list cap and is not stored in the MACE-OFF checkpoint
- it does **not** infer DPRc/MM labels such as `mH`, `mC`, `HW`, or `OW`
- `load_mace_off_model()` keeps the original eager MACE checkpoint model inside
  the wrapper for closer native/wrapper parity, while
  `convert_mace_off_to_deepmd()` scripts the model only at export time
- it uses `torch.load(..., weights_only=False)` to recover the original
  `ScaleShiftMACE` object, so callers should only use trusted checkpoints

Example:

```sh
deepmd-gnn mace-off convert mace_off23_small_dp.pt --model off23_small --sel 64
```

This produces a scripted DeePMD-GNN wrapper. That serialization step is useful,
but it is still a good idea to validate the final downstream workflow you care
about (for example in LAMMPS or AMBER).

## Examples

- [examples/water](examples/water)
- [examples/dprc](examples/dprc)
- [examples/property/mace](examples/property/mace)
