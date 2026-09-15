# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: INP001, T201
"""Write a tiny SevenNet checkpoint and dummy property data for smoke training."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import sevenn
import sevenn._keys as KEY  # noqa: N812
import torch
from sevenn._const import DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG
from sevenn.model_build import build_E3_equivariant_model
from sevenn.util import chemical_species_preprocess

HERE = Path(__file__).resolve().parent


def _tiny_config() -> dict:
    config = deepcopy(DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG)
    config.update(chemical_species_preprocess(["H", "O"]))
    config[KEY.CUTOFF] = 3.0
    config[KEY.NODE_FEATURE_MULTIPLICITY] = 4
    config[KEY.LMAX] = 1
    config[KEY.NUM_CONVOLUTION] = 2
    config[KEY.CONV_DENOMINATOR] = 4.0
    config[KEY.SHIFT] = 0.0
    config[KEY.SCALE] = 1.0
    config[KEY.CONVOLUTION_WEIGHT_NN_HIDDEN_NEURONS] = [8]
    config["radial_basis"] = {
        "radial_basis_name": "bessel",
        "bessel_basis_num": 4,
    }
    config["version"] = sevenn.__version__
    config["dtype"] = "float32"
    return config


def main() -> None:
    """Create the local checkpoint and DeePMD property system used by input.json."""
    checkpoint = HERE / "tiny_sevennet.pth"
    config = _tiny_config()
    model = build_E3_equivariant_model(config)
    torch.save(
        {
            "config": config,
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "time": "smoke",
            "hash": "smoke",
        },
        checkpoint,
    )

    system = HERE / "data"
    data_set = system / "set.000"
    data_set.mkdir(parents=True, exist_ok=True)
    # type_map is [H, O]; this molecule is O, H, H.
    (system / "type.raw").write_text("1\n0\n0\n")
    (system / "type_map.raw").write_text("H\nO\n")
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]],
            [[0.0, 0.0, 0.0], [0.8, 0.2, 0.1], [-0.1, 0.9, 0.4]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [-0.3, 1.1, 0.2]],
            [[0.0, 0.0, 0.0], [0.7, 0.3, 0.0], [-0.2, 0.8, 0.5]],
            [[0.0, 0.0, 0.0], [0.95, 0.05, 0.1], [-0.15, 1.05, 0.2]],
            [[0.0, 0.0, 0.0], [0.85, 0.15, -0.1], [-0.25, 0.95, 0.35]],
            [[0.0, 0.0, 0.0], [1.05, 0.0, 0.05], [-0.1, 1.15, 0.15]],
            [[0.0, 0.0, 0.0], [0.75, 0.25, 0.05], [-0.3, 0.85, 0.25]],
        ],
        dtype=np.float64,
    )
    np.save(data_set / "coord.npy", coordinates.reshape(len(coordinates), -1))
    np.save(
        data_set / "box.npy",
        np.tile((np.eye(3) * 8.0).reshape(1, 9), (len(coordinates), 1)),
    )
    np.save(
        data_set / "band_gap.npy",
        np.array(
            [[0.2], [0.7], [-0.4], [1.1], [0.3], [0.9], [-0.2], [0.5]],
            dtype=np.float64,
        ),
    )
    print(f"checkpoint: {checkpoint}")
    print(f"data:       {system}")
    print(f"type_map:   {config['chemical_species']}")
    print("next: dp --pt train input.json")


if __name__ == "__main__":
    main()
