# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the legacy NequIP invariant descriptor."""
# ruff: noqa: D103

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from deepmd.pt.model.model import get_model
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list
from nequip.data import AtomicDataDict

from deepmd_gnn.nequip import NequipModel, _make_nequip_network
from deepmd_gnn.nequip_descriptor import NequipDescriptor

PARAMS = {
    "type_map": ["O", "H"],
    "sel": 8,
    "r_max": 3.0,
    "num_layers": 2,
    "l_max": 1,
    "num_features": 4,
    "feature_irreps_hidden": "4x0e + 4x1o",
    "chemical_embedding_irreps_out": "4x0e",
    "conv_to_output_hidden_irreps_out": "3x0e",
}


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    model = NequipModel(**PARAMS)
    path = tmp_path / "nequip.pt"
    torch.save(model.serialize(), path)
    return path


@pytest.fixture
def descriptor(artifact: Path) -> NequipDescriptor:
    return NequipDescriptor(
        sel=PARAMS["sel"],
        type_map=PARAMS["type_map"],
        model_file=str(artifact),
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.3, 1.2, 0.4]]],
    )
    atype = torch.tensor([[0, 1, 1]], dtype=torch.int64)
    nlist = torch.tensor([[[1, 2], [0, 2], [0, 1]]], dtype=torch.int64)
    return coord, atype, nlist


def test_metadata_and_strict_pretrained_mapping(
    artifact: Path,
    descriptor: NequipDescriptor,
) -> None:
    assert descriptor.get_dim_out() == 3
    assert descriptor.get_sel() == [8]
    assert descriptor.get_rcut() == 3.0
    assert descriptor.mixed_types()
    assert descriptor.has_message_passing()
    assert not descriptor.has_message_passing_across_ranks()
    assert not descriptor.supports_edge_parallel()
    assert not descriptor.dense_lower_supports_comm()
    assert all(parameter.requires_grad for parameter in descriptor.parameters())

    payload = torch.load(artifact, weights_only=False)
    payload["@variables"].pop(
        "model.conv_to_output_hidden.linear.weight",
    )
    broken = artifact.with_name("broken.pt")
    torch.save(payload, broken)
    with pytest.raises(ValueError, match="backbone state mismatch"):
        NequipDescriptor(
            sel=8,
            type_map=["O", "H"],
            model_file=str(broken),
        )
    with pytest.raises(ValueError, match="type_map"):
        NequipDescriptor(sel=8, type_map=["H", "O"], model_file=str(artifact))
    with pytest.raises(ValueError, match="config conflicts"):
        NequipDescriptor(
            sel=8,
            type_map=["O", "H"],
            model_file=str(artifact),
            num_layers=3,
        )


def test_features_match_original_graph(
    descriptor: NequipDescriptor,
) -> None:
    coord, atype, nlist = _inputs()
    actual = descriptor(coord, atype, nlist)[0]

    original = _make_nequip_network(descriptor.params, 2)
    original.load_state_dict(
        {
            **descriptor.model.state_dict(),
            **{
                key: value
                for key, value in original.state_dict().items()
                if key.startswith("model.output_hidden_to_scalar.")
            },
        },
        strict=True,
    )
    edge_index = torch.ops.deepmd_gnn.edge_index(
        nlist,
        atype,
        torch.empty(0, dtype=torch.int64),
    ).T[[1, 0]]
    data = {
        AtomicDataDict.POSITIONS_KEY: coord.reshape(-1, 3).to(torch.float32),
        AtomicDataDict.EDGE_INDEX_KEY: edge_index,
        AtomicDataDict.ATOM_TYPE_KEY: atype.reshape(-1),
        "batch": torch.zeros(3, dtype=torch.int64),
        "ptr": torch.tensor([0, 3], dtype=torch.int64),
    }
    for name, module in original.model._modules.items():  # noqa: SLF001
        data = module(data)
        if name == "conv_to_output_hidden":
            break
    expected = data[AtomicDataDict.NODE_FEATURES_KEY].view(1, 3, 3)
    torch.testing.assert_close(actual, expected.to(actual.dtype))


def test_shape_rotation_invariance_and_mapping(
    descriptor: NequipDescriptor,
) -> None:
    coord, atype, nlist = _inputs()
    features = descriptor(coord, atype, nlist)[0]
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )
    rotated = descriptor(coord @ rotation.T, atype, nlist)[0]
    assert features.shape == (1, 3, 3)
    torch.testing.assert_close(features, rotated, atol=2e-6, rtol=2e-6)

    extended_coord = torch.cat((coord, coord[:, 1:2]), 1)
    extended_atype = torch.cat((atype, atype[:, 1:2]), 1)
    mapping = torch.tensor([[0, 1, 2, 1]], dtype=torch.int64)
    mapped_nlist = nlist.clone()
    mapped_nlist[0, 0, 0] = 3
    mapped = descriptor(
        extended_coord,
        extended_atype,
        mapped_nlist,
        mapping=mapping,
    )[0]
    torch.testing.assert_close(features, mapped, atol=2e-6, rtol=2e-6)

    box = torch.eye(3, dtype=torch.float64).reshape(1, 9) * 6.0
    periodic_features = []
    for periodic_coord in (
        torch.tensor([[[0.1, 0, 0], [5.9, 0, 0]]], dtype=torch.float64),
        torch.tensor([[[0.1, 0, 0], [-0.1, 0, 0]]], dtype=torch.float64),
    ):
        coord_ext, atype_ext, pbc_mapping, pbc_nlist = (
            extend_input_and_build_neighbor_list(
                periodic_coord,
                torch.tensor([[0, 1]]),
                descriptor.get_rcut(),
                descriptor.get_sel(),
                mixed_types=True,
                box=box,
            )
        )
        periodic_features.append(
            descriptor(
                coord_ext,
                atype_ext,
                pbc_nlist,
                mapping=pbc_mapping,
            )[0],
        )
    torch.testing.assert_close(periodic_features[0], periodic_features[1])


def test_property_gradients_optimizer_and_roundtrip(
    artifact: Path,
) -> None:
    config = {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "nequip",
            "sel": 8,
            "model_file": str(artifact),
        },
        "fitting_net": {
            "type": "property",
            "property_name": "band_prop",
            "task_dim": 1,
            "neuron": [4],
            "precision": "float32",
        },
    }
    model = get_model(config)
    coord, atype, _ = _inputs()
    descriptor_before = {
        key: value.detach().clone()
        for key, value in model.get_descriptor().state_dict().items()
    }
    fitting_before = {
        key: value.detach().clone()
        for key, value in model.get_fitting_net().state_dict().items()
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss = model(coord, atype)["band_prop"].square().sum()
    optimizer.zero_grad()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.get_descriptor().parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.get_fitting_net().parameters()
    )
    optimizer.step()
    assert any(
        not torch.equal(descriptor_before[key], value)
        for key, value in model.get_descriptor().state_dict().items()
    )
    assert any(
        not torch.equal(fitting_before[key], value)
        for key, value in model.get_fitting_net().state_dict().items()
    )

    restored = NequipDescriptor.deserialize(model.get_descriptor().serialize())
    torch.testing.assert_close(
        model.get_descriptor()(coord, atype, _inputs()[2])[0],
        restored(coord, atype, _inputs()[2])[0],
    )


def test_readout_and_e0_do_not_enter_descriptor(
    artifact: Path,
    descriptor: NequipDescriptor,
) -> None:
    payload = torch.load(artifact, weights_only=False)
    descriptor_keys = set(descriptor.model.state_dict())
    assert "e0" not in descriptor_keys
    assert not any("output_hidden_to_scalar" in key for key in descriptor_keys)
    assert all(
        np.shape(payload["@variables"][key])
        == tuple(descriptor.model.state_dict()[key].shape)
        for key in descriptor_keys
    )


def test_dp_train_property_updates_backbone(
    tmp_path: Path,
    artifact: Path,
) -> None:
    data = tmp_path / "data"
    set_dir = data / "set.000"
    set_dir.mkdir(parents=True)
    (data / "type.raw").write_text("0\n1\n")
    (data / "type_map.raw").write_text("O\nH\n")
    (data / "nopbc").touch()
    np.save(
        set_dir / "coord.npy",
        np.array(
            [[0, 0, 0, 1, 0, 0], [0, 0, 0, 1.1, 0.1, 0]],
            dtype=np.float64,
        ),
    )
    np.save(
        set_dir / "band_prop.npy",
        np.array([[0.5], [0.7]], dtype=np.float64),
    )
    checkpoint = tmp_path / "model.ckpt"
    config = {
        "model": {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "nequip",
                "sel": 8,
                "model_file": str(artifact),
            },
            "fitting_net": {
                "type": "property",
                "property_name": "band_prop",
                "task_dim": 1,
                "neuron": [4],
                "precision": "float32",
            },
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 10,
            "start_lr": 0.01,
            "stop_lr": 0.001,
        },
        "loss": {
            "type": "property",
            "metric": ["mae"],
            "loss_func": "smooth_mae",
            "beta": 1.0,
        },
        "training": {
            "training_data": {"systems": [str(data)], "batch_size": 1},
            "numb_steps": 2,
            "disp_freq": 1,
            "save_freq": 1,
            "save_ckpt": str(checkpoint),
            "disp_file": str(tmp_path / "lcurve.out"),
        },
    }
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps(config))
    dp = Path(sys.executable).with_name("dp")
    env_vars = os.environ.copy()
    env_vars["PYTHONWARNINGS"] = "ignore"
    subprocess.run(
        [str(dp), "--pt", "train", str(input_file)],
        cwd=tmp_path,
        env=env_vars,
        check=True,
        capture_output=True,
        text=True,
    )
    trained = torch.load(
        tmp_path / "model.ckpt-2.pt",
        map_location="cpu",
        weights_only=False,
    )["model"]
    initial = torch.load(artifact, weights_only=False)["@variables"]
    prefix = "model.Default.atomic_model.descriptor.model."
    changed = []
    for key, value in trained.items():
        if key.startswith(prefix + "model.layer"):
            artifact_key = key.removeprefix(prefix)
            changed.append(
                not torch.equal(
                    value.cpu(),
                    torch.as_tensor(initial[artifact_key]),
                ),
            )
    assert changed
    assert any(changed)
