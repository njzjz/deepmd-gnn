# DeePMD-kit plugin for various graph neural network models

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/python-template)](https://pypi.org/p/python-template) -->

`deepmd-gnn` is a [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) plugin for various graph neural network (GNN) models, which connects DeePMD-kit and atomistic GNN packages by enabling GNN models in DeePMD-kit.

Supported packages and models include:

- [MACE](https://github.com/ACEsuit/mace) (PyTorch version)
- [NequIP](https://github.com/mir-group/nequip) (PyTorch version)

After [installing the plugin](#installation), you can train the GNN models using DeePMD-kit, run active learning cycles for the GNN models using [DP-GEN](https://github.com/deepmodeling/dpgen), perform simulations with the MACE model using molecular dynamic packages supported by DeePMD-kit, such as [LAMMPS](https://github.com/lammps/lammps) and [AMBER](https://ambermd.org/).
You can follow [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/) to train the GNN models using its PyTorch backend, after using the specific [model parameters](#parameters).

## Credits

If you use this software, please cite the following unpublish paper:

- Jinzhe Zeng, Timothy J. Giese, Duo Zhang, Han Wang, Darrin M. York, DeePMD-GNN: A DeePMD-kit Plugin for External Graph Neural Network Potentials, unpublished.

We will update the credit information once it is published.

## Installation

First, clone this repository:

```sh
git clone https://github.com/njzjz/deepmd-gnn
cd deepmd-gnn
```

### Python interface plugin

Python 3.9 or above is required. A C++ compiler that supports C++ 14 (for PyTorch 2.0) or C++ 17 (for PyTorch 2.1 or above) is required.

Assume you have installed [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) (v3.0.0b2 or above) and [PyTorch](https://github.com/pytorch/pytorch) in an environment, then execute

```sh
# expose PyTorch CMake modules
export CMAKE_PREFIX_PATH=$(python -c "import torch;print(torch.utils.cmake_prefix_path)")

pip install .
```

### C++ interface plugin

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
  "precision": "float32"
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

## Examples

- [examples/water](examples/water)
- [examples/dprc](examples/dprc)
