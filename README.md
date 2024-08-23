# MACE plugin for DeePMD-kit

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/python-template)](https://pypi.org/p/python-template) -->

`deepmd-mace` is a [MACE](https://github.com/ACEsuit/mace) plugin for [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit), which connects MACE (PyTorch version) and DeePMD-kit by enabling MACE models in DeePMD-kit PyTorch backend.

After [installing the plugin](#installation), you can train the MACE model using DeePMD-kit, run active learning cycles for the MACE model using [DP-GEN](https://github.com/deepmodeling/dpgen), perform simulations with the MACE model using molecular dynamic packages supported by DeePMD-kit, such as [LAMMPS](https://github.com/lammps/lammps) and [AMBER](https://ambermd.org/).
You can follow [DeePMD-kit documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/) to train the MACE models using its PyTorch backend, after using the specific [MACE parameters](#parameters).

## Installation

First, clone this repository:

```sh
git clone https://github.com/njzjz/deepmd-mace
cd deepmd-mace
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

DeePMD-kit needs to support [customized OP library in C++ interface](https://github.com/deepmodeling/deepmd-kit/pull/4073) (available after Aug 23, 2024).

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

`libdeepmd_mace.so` will be installed into the directory you assign.
When using any DeePMD-kit C++ interface, set the following environment variable in advance:

```sh
export DP_PLUGIN_PATH=/prefix/to/install/lib/libdeepmd_mace.so
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
  "std": null
}
```

## DPRc support

In `deepmd-mace`, the MACE model can be used in a [DPRc](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dprc.html) way.
Type maps that starts with `m` (such as `mH`) or `OW` or `HW` will be recognized as MM types.
Two MM atoms will not build edges with each other.
Such MACE+DPRc model can be directly used in AmberTools24.

## Examples

- [examples/water]
- [examples/dprc]
