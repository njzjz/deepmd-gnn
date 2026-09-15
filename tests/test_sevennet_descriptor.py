# SPDX-License-Identifier: LGPL-3.0-or-later
"""Focused tests for the SevenNet property descriptor."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from deepmd.pt.model.descriptor.base_descriptor import BaseDescriptor
from deepmd.pt.model.model import get_model
from deepmd.pt.utils import env
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list
from e3nn.o3 import Irreps

pytest.importorskip("sevenn")

import sevenn
import sevenn._const as sevenn_const
import sevenn._keys as KEY  # noqa: N812
from sevenn._const import DEFAULT_E3_EQUIVARIANT_MODEL_CONFIG
from sevenn.atom_graph_data import AtomGraphData
from sevenn.model_build import build_E3_equivariant_model
from sevenn.train.dataload import unlabeled_atoms_to_graph
from sevenn.util import chemical_species_preprocess, load_checkpoint

from deepmd_gnn.sevennet_checkpoint import (
    ENERGY_MODULE_NAMES,
    _dtype_from_name,
    _infer_state_dtype,
    json_safe_value,
    last_feature_irreps,
    load_sevennet_checkpoint_config,
    persistable_checkpoint_config,
    reject_unsupported_sevennet,
    restore_sevenn_config,
    scalar_even_indices,
    validate_sevennet_state_dict_load,
)
from deepmd_gnn.sevennet_descriptor import SevenNetDescriptor


def _tiny_sevenn_config() -> dict:
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


def _write_sevennet_checkpoint(path: Path, *, use_modality: bool = False) -> Path:
    config = _tiny_sevenn_config()
    model = build_E3_equivariant_model(config)
    if use_modality:
        config["use_modality"] = True
        config["_modal_map"] = {"pbe": 0, "scan": 1}
    torch.save(
        {
            "config": config,
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "time": "test",
            "hash": "test",
        },
        path,
    )
    return path


@pytest.fixture
def sevennet_checkpoint(tmp_path: Path) -> Path:
    """Create a tiny trusted local SevenNet checkpoint."""
    return _write_sevennet_checkpoint(tmp_path / "small_sevennet.pth")


def _inputs(
    descriptor: SevenNetDescriptor,
    coord: torch.Tensor | None = None,
    box: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if coord is None:
        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]]],
            dtype=torch.float64,
        )
    coord = coord.to(env.DEVICE)
    atype = torch.tensor([[1, 0, 0]], dtype=torch.int64, device=env.DEVICE)
    if box is not None:
        box = box.to(env.DEVICE)
    coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
        coord.reshape(1, -1),
        atype,
        descriptor.get_rcut(),
        descriptor.get_sel(),
        mixed_types=True,
        box=box,
    )
    return coord_ext, atype_ext, nlist, mapping


def _descriptor_output(
    descriptor: SevenNetDescriptor,
    coord: torch.Tensor | None = None,
    box: torch.Tensor | None = None,
) -> torch.Tensor:
    coord_ext, atype_ext, nlist, mapping = _inputs(descriptor, coord, box)
    return descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0]


def _native_last_layer(
    checkpoint: Path,
    coord: torch.Tensor,
    box: torch.Tensor | None = None,
) -> torch.Tensor:
    model = load_checkpoint(str(checkpoint)).build_model().to(env.DEVICE)
    model.eval()
    atoms = Atoms(
        "OHH",
        positions=coord.detach().cpu().numpy()[0],
        cell=None if box is None else box.detach().cpu().numpy().reshape(3, 3),
        pbc=box is not None,
    )
    graph = unlabeled_atoms_to_graph(atoms, cutoff=3.0)
    data = AtomGraphData.from_numpy_dict(graph)
    data[KEY.BATCH] = torch.zeros(len(atoms), dtype=torch.int64)
    data = model._preprocess(data)  # noqa: SLF001
    for name, module in model.named_children():
        if name in ENERGY_MODULE_NAMES:
            break
        data = module(data)
    return data["x"].view(1, len(atoms), -1)


def test_constructor_registry_and_metadata(sevennet_checkpoint: Path) -> None:
    """Descriptor registration and metadata follow DeePMD's contract."""
    descriptor = BaseDescriptor(
        type="sevennet",
        model_path=str(sevennet_checkpoint),
        sel=16,
        type_map=["H", "O"],
        ntypes=2,
    )
    assert isinstance(descriptor, SevenNetDescriptor)
    assert descriptor.get_type_map() == ["H", "O"]
    assert descriptor.get_sel() == [16]
    assert descriptor.get_nsel() == 16
    assert descriptor.get_rcut() == pytest.approx(3.0)
    assert descriptor.get_dim_out() == 4
    assert descriptor.get_ntypes() == 2
    assert descriptor.get_rcut_smth() == pytest.approx(3.0)
    assert descriptor.get_dim_emb() == 4
    assert descriptor.get_env_protection() == 0.0
    assert descriptor.mixed_types()
    assert descriptor.has_message_passing()
    assert not descriptor.has_message_passing_across_ranks()
    assert not descriptor.supports_edge_parallel()
    assert not descriptor.dense_lower_supports_comm()
    assert not descriptor.need_sorted_nlist_for_lower()
    assert descriptor.has_default_chg_spin() is False
    descriptor.get_default_chg_spin()
    descriptor.compute_input_stats([])
    assert descriptor.get_stats() == {}
    assert all(parameter.requires_grad for parameter in descriptor.parameters())
    descriptor.set_stat_mean_and_stddev(torch.ones(1), torch.ones(1))
    mean, stddev = descriptor.get_stat_mean_and_stddev()
    assert mean.numel() == 0
    assert stddev.numel() == 0
    assert descriptor.output_irreps.replace(" ", "") == "4x0e"
    assert scalar_even_indices(Irreps("4x0e")) == [0, 1, 2, 3]

    local_config = {
        "type": "sevennet",
        "model_path": str(sevennet_checkpoint),
        "sel": 16,
    }
    updated, min_distance = BaseDescriptor.update_sel(
        None,
        ["H", "O"],
        local_config,
    )
    assert updated["model_path"] == str(sevennet_checkpoint)
    assert updated["sel"] == 16
    assert updated["config"]["type_map"] == ["H", "O"]
    json.dumps(updated)
    assert min_distance is None


def test_descriptor_supports_first_import_in_fresh_process() -> None:
    """Importing the descriptor first must not recurse through the PT entry point."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from deepmd_gnn.sevennet_descriptor import SevenNetDescriptor; "
                "assert SevenNetDescriptor.__name__ == 'SevenNetDescriptor'"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_constructor_requires_exact_checkpoint_type_map(
    sevennet_checkpoint: Path,
) -> None:
    """Checkpoint element order must equal the model type map."""
    with pytest.raises(ValueError, match="exactly match checkpoint"):
        SevenNetDescriptor(
            model_path=sevennet_checkpoint,
            sel=16,
            type_map=["O", "H"],
        )


def test_constructor_rejects_modal_checkpoint(tmp_path: Path) -> None:
    """Multi-fidelity SevenNet checkpoints cannot define a descriptor backbone."""
    checkpoint = _write_sevennet_checkpoint(
        tmp_path / "modal.pth",
        use_modality=True,
    )
    with pytest.raises(ValueError, match="modal"):
        SevenNetDescriptor(
            model_path=checkpoint,
            sel=16,
            type_map=["H", "O"],
        )


def test_constructor_and_runtime_error_paths(
    sevennet_checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover constructor, update_sel, share, and forward rejection paths."""
    with pytest.raises(ValueError, match="positive integer"):
        SevenNetDescriptor(
            model_path=sevennet_checkpoint,
            sel=0,
            type_map=["H", "O"],
        )
    with pytest.raises(ValueError, match="requires the model-level type_map"):
        SevenNetDescriptor(model_path=sevennet_checkpoint, sel=16)
    with pytest.raises(ValueError, match="ntypes="):
        SevenNetDescriptor(
            model_path=sevennet_checkpoint,
            sel=16,
            type_map=["H", "O"],
            ntypes=3,
        )
    with pytest.raises(TypeError, match="Unsupported SevenNet descriptor arguments"):
        SevenNetDescriptor(
            model_path=sevennet_checkpoint,
            sel=16,
            type_map=["H", "O"],
            unknown=True,
        )
    with pytest.raises(FileNotFoundError, match="not found"):
        SevenNetDescriptor(
            model_path=tmp_path / "missing.pth",
            sel=16,
            type_map=["H", "O"],
        )
    with pytest.raises(FileNotFoundError, match="not found"):
        load_sevennet_checkpoint_config("not-a-real-sevennet-keyword")
    with pytest.raises(ValueError, match="Exactly one of model_path"):
        SevenNetDescriptor(sel=16, type_map=["H", "O"])
    with pytest.raises(ValueError, match="Serialized SevenNet descriptor type_map"):
        SevenNetDescriptor(
            sel=16,
            type_map=["O", "H"],
            config=load_sevennet_checkpoint_config(sevennet_checkpoint),
        )
    frozen = SevenNetDescriptor(
        model_path=sevennet_checkpoint,
        sel=16,
        type_map=["H", "O"],
        trainable=False,
    )
    assert all(not parameter.requires_grad for parameter in frozen.parameters())
    inferred_config: dict = {"stale": True}
    SevenNetDescriptor(
        model_path=sevennet_checkpoint,
        sel=16,
        type_map=["H", "O"],
        config=inferred_config,
    )
    assert "stale" not in inferred_config
    assert inferred_config["type_map"] == ["H", "O"]

    monkeypatch.setattr(
        "deepmd_gnn.sevennet_descriptor.last_feature_irreps",
        lambda _model: Irreps("1x1o"),
    )
    with pytest.raises(ValueError, match="no 0e channels"):
        SevenNetDescriptor(
            model_path=sevennet_checkpoint,
            sel=16,
            type_map=["H", "O"],
        )
    monkeypatch.undo()

    descriptor = SevenNetDescriptor(
        model_path=sevennet_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    shared = SevenNetDescriptor(
        model_path=sevennet_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    shared.share_params(descriptor, 0)
    assert shared.backbone is descriptor.backbone
    with pytest.raises(TypeError, match="can only share"):
        shared.share_params(object(), 0)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="level 0"):
        shared.share_params(descriptor, 1)
    with pytest.raises(NotImplementedError, match="changing or subsetting type_map"):
        descriptor.change_type_map(["H"])

    coord_ext, atype_ext, nlist, mapping = _inputs(descriptor)
    with pytest.raises(NotImplementedError, match="MPI communication"):
        descriptor(
            coord_ext,
            atype_ext,
            nlist,
            mapping=mapping,
            comm_dict={"send": torch.zeros(1)},
        )
    coord = torch.tensor(
        [[[0.1, 0.2, 0.3], [3.8, 0.2, 0.3], [0.2, 3.7, 0.4]]],
        dtype=torch.float64,
    )
    box = (torch.eye(3, dtype=torch.float64) * 4.0).reshape(1, 9)
    coord_ext, atype_ext, nlist, mapping = _inputs(descriptor, coord, box)
    assert atype_ext.shape[1] > nlist.shape[1]
    with pytest.raises(ValueError, match="requires mapping"):
        descriptor(coord_ext, atype_ext, nlist, mapping=None)

    with pytest.raises(ValueError, match="serialized SevenNetDescriptor"):
        SevenNetDescriptor.deserialize(
            {
                "@class": "Descriptor",
                "type": "mace",
                "@version": 1,
            },
        )
    with pytest.raises(ValueError, match="explicit positive integer sel"):
        SevenNetDescriptor.update_sel(None, ["H", "O"], {"sel": True})
    updated, min_distance = SevenNetDescriptor.update_sel(
        None,
        None,
        {"sel": 16, "model_path": str(sevennet_checkpoint)},
    )
    assert updated["config"]["type_map"] == ["H", "O"]
    assert min_distance is None
    unchanged, _ = SevenNetDescriptor.update_sel(None, ["H", "O"], {"sel": 16})
    assert "config" not in unchanged
    with pytest.raises(ValueError, match="exactly match checkpoint"):
        SevenNetDescriptor.update_sel(
            None,
            ["O", "H"],
            {
                "sel": 16,
                "model_path": str(sevennet_checkpoint),
            },
        )


def test_checkpoint_helpers_cover_unsupported_and_json_paths(
    tmp_path: Path,
) -> None:
    """Reject unsupported variants and keep persisted configs JSON-safe."""
    reject_cases = (
        ({"use_modality": True}, "Multi-fidelity"),
        ({"_modal_map": {"pbe": 0}}, "modal map"),
        ({"cuequivariance_config": {"use": True}}, "cuEquivariance"),
        ({"use_flash_tp": True}, "FlashTP"),
        ({"use_oeq": True}, "OpenEquivariance"),
        ({"use_mliap": True}, "MLIAP"),
    )
    for config, match in reject_cases:
        with pytest.raises(ValueError, match=match):
            reject_unsupported_sevennet(config)

    payload = json_safe_value(
        {
            1: torch.tensor([1.5]),
            "arr": np.array([2, 3]),
            "i": np.int64(4),
            "f": np.float64(1.25),
            "b": np.bool_(True),  # noqa: FBT003
            "p": tmp_path / "x",
            "t": (1, 2),
        },
    )
    assert payload["1"] == [1.5]
    assert payload["arr"] == [2, 3]
    assert payload["i"] == 4
    assert payload["f"] == pytest.approx(1.25)
    assert payload["b"] is True
    assert payload["p"].endswith("x")
    assert payload["t"] == [1, 2]
    json.dumps(payload)

    restored = restore_sevenn_config({"_type_map": {"0": 1, "1": 8}})
    assert restored["_type_map"] == {0: 1, 1: 8}

    persistable = persistable_checkpoint_config(
        {
            "type_map": ["H"],
            "source_dtype": "float16",
            "sevenn_config": {
                "cutoff": 3.0,
                "num_convolution_layer": 2,
                "chemical_species": ["H"],
                "dtype": "float16",
            },
        },
    )
    assert persistable["source_dtype"] == "float32"
    assert _dtype_from_name("float64") == torch.float64
    assert _dtype_from_name("float32") == torch.float32
    assert _infer_state_dtype({"n": torch.tensor(1)}) == torch.float32
    assert (
        _infer_state_dtype({"w": torch.zeros(1, dtype=torch.float64)}) == torch.float64
    )

    empty = torch.nn.Module()
    with pytest.raises(ValueError, match="no equivariant_gate"):
        last_feature_irreps(empty)

    class _Gate(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = None

    class _Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_equivariant_gate = _Gate()

    with pytest.raises(ValueError, match="does not expose"):
        last_feature_irreps(_Backbone())

    assert scalar_even_indices(Irreps("2x0e+1x1o+1x0o+1x0e")) == [0, 1, 6]

    broken = tmp_path / "not_a_checkpoint.pth"
    torch.save([1, 2, 3], broken)
    with pytest.raises(TypeError, match="dictionary"):
        load_sevennet_checkpoint_config(broken)

    class _LoadResult:
        missing_keys = ("weight",)
        unexpected_keys = ()

    with pytest.raises(RuntimeError, match="Failed to load"):
        validate_sevennet_state_dict_load(_LoadResult())


def test_forward_shape_rotation_invariance_and_gradient(
    sevennet_checkpoint: Path,
) -> None:
    """Final ``0e`` features have local-atom shape, are invariant, and train."""
    descriptor = SevenNetDescriptor(
        model_path=sevennet_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    output = _descriptor_output(descriptor)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]]],
        dtype=torch.float64,
    )
    rotated = _descriptor_output(descriptor, coord @ rotation.T)
    assert output.shape == (1, 3, 4)
    torch.testing.assert_close(output, rotated, rtol=2e-5, atol=2e-6)
    output.square().sum().backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in descriptor.backbone.parameters()
    )


def test_final_features_match_native_sevennet_forward(
    sevennet_checkpoint: Path,
) -> None:
    """The compact graph adapter preserves last-layer SevenNet features."""
    descriptor = SevenNetDescriptor(
        model_path=sevennet_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]]],
        dtype=torch.float64,
    )
    native = _native_last_layer(sevennet_checkpoint, coord)
    actual = _descriptor_output(descriptor, coord)
    torch.testing.assert_close(
        actual,
        native.to(env.GLOBAL_PT_FLOAT_PRECISION),
        rtol=1e-5,
        atol=1e-6,
    )


def test_periodic_mapping_and_coordinate_gradient(
    sevennet_checkpoint: Path,
) -> None:
    """Periodic images follow compact mapping without breaking gradients."""
    descriptor = SevenNetDescriptor(
        model_path=sevennet_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    coord = torch.tensor(
        [[[0.1, 0.2, 0.3], [3.8, 0.2, 0.3], [0.2, 3.7, 0.4]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    box = (torch.eye(3, dtype=torch.float64) * 4.0).reshape(1, 9)
    coord_ext, atype_ext, nlist, mapping = _inputs(descriptor, coord, box)
    assert atype_ext.shape[1] > nlist.shape[1]
    output = descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0]
    output.square().sum().backward()
    assert coord.grad is not None
    assert torch.count_nonzero(coord.grad)

    translated = coord.detach().clone()
    translated[:, 1, 0] -= 4.0
    translated_output = _descriptor_output(descriptor, translated, box)
    torch.testing.assert_close(output, translated_output, rtol=2e-5, atol=2e-6)

    native = _native_last_layer(sevennet_checkpoint, coord.detach(), box)
    torch.testing.assert_close(
        output.detach(),
        native.to(env.GLOBAL_PT_FLOAT_PRECISION),
        rtol=1e-5,
        atol=1e-6,
    )


def test_compact_periodic_mapping_handles_multiple_frames(
    sevennet_checkpoint: Path,
) -> None:
    """Compact node indices preserve independent frames with periodic images."""
    descriptor = SevenNetDescriptor(
        model_path=sevennet_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    coord = torch.tensor(
        [
            [[0.1, 0.2, 0.3], [3.8, 0.2, 0.3], [0.2, 3.7, 0.4]],
            [[0.2, 0.1, 0.4], [3.7, 0.3, 0.2], [0.3, 3.6, 0.5]],
        ],
        dtype=torch.float64,
        device=env.DEVICE,
    )
    atype = torch.tensor(
        [[1, 0, 0], [1, 0, 0]],
        dtype=torch.int64,
        device=env.DEVICE,
    )
    box = (
        torch.eye(3, dtype=torch.float64, device=env.DEVICE)
        .mul(4.0)
        .reshape(1, 9)
        .expand(2, 9)
    )
    coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
        coord.reshape(2, -1),
        atype,
        descriptor.get_rcut(),
        descriptor.get_sel(),
        mixed_types=True,
        box=box,
    )
    batched = descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0]
    per_frame = []
    for frame in range(2):
        frame_output = _descriptor_output(
            descriptor,
            coord[frame : frame + 1],
            box[frame : frame + 1],
        )
        per_frame.append(frame_output)
    torch.testing.assert_close(batched, torch.cat(per_frame))


def test_descriptor_forward_is_torchscriptable(sevennet_checkpoint: Path) -> None:
    """``dp test`` scripts the eager checkpoint; forward must not use Python builtins."""
    descriptor = SevenNetDescriptor(
        model_path=sevennet_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    coord_ext, atype_ext, nlist, mapping = _inputs(descriptor)
    expected = descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0]
    scripted = torch.jit.script(descriptor)
    actual = scripted(coord_ext, atype_ext, nlist, mapping)[0]
    torch.testing.assert_close(actual, expected)

    model = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "sevennet",
                "model_path": str(sevennet_checkpoint),
                "sel": 16,
            },
            "fitting_net": {
                "type": "property",
                "property_name": "band_gap",
                "task_dim": 1,
                "neuron": [8, 8],
                "precision": "float64",
            },
        },
    )
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]]],
        dtype=torch.float64,
        device=env.DEVICE,
    ).reshape(1, -1)
    atype = torch.tensor([[1, 0, 0]], dtype=torch.int64, device=env.DEVICE)
    expected_pred = model(coord, atype)["band_gap"]
    eager_state = model.state_dict()
    scripted_model = torch.jit.script(model)
    scripted_model.load_state_dict(eager_state)
    actual_pred = scripted_model(coord, atype)["band_gap"]
    torch.testing.assert_close(actual_pred, expected_pred)


def test_property_model_composition_and_optimizer_step(
    sevennet_checkpoint: Path,
) -> None:
    """Property fitting updates both backbone and fitting parameters."""
    model = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "sevennet",
                "model_path": str(sevennet_checkpoint),
                "sel": 16,
            },
            "fitting_net": {
                "type": "property",
                "property_name": "band_gap",
                "task_dim": 2,
                "neuron": [8, 8],
                "precision": "float64",
            },
        },
    )
    descriptor = model.get_descriptor()
    fitting = model.get_fitting_net()
    backbone_parameter = next(descriptor.backbone.parameters())
    fitting_parameter = next(fitting.parameters())
    backbone_before = backbone_parameter.detach().clone()
    fitting_before = fitting_parameter.detach().clone()

    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]]],
        dtype=torch.float64,
        device=env.DEVICE,
    ).reshape(1, -1)
    atype = torch.tensor([[1, 0, 0]], dtype=torch.int64, device=env.DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss = model(coord, atype)["band_gap"].square().sum()
    optimizer.zero_grad()
    loss.backward()
    assert backbone_parameter.grad is not None
    assert torch.count_nonzero(backbone_parameter.grad)
    assert fitting_parameter.grad is not None
    assert torch.count_nonzero(fitting_parameter.grad)
    optimizer.step()
    assert not torch.equal(backbone_before, backbone_parameter)
    assert not torch.equal(fitting_before, fitting_parameter)


def test_get_model_restores_without_source_checkpoint(
    sevennet_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Saved model definitions reconstruct the backbone without the native file."""
    updated, _ = SevenNetDescriptor.update_sel(
        None,
        ["H", "O"],
        {
            "type": "sevennet",
            "model_path": str(sevennet_checkpoint),
            "sel": 16,
        },
    )
    original = SevenNetDescriptor(
        model_path=sevennet_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    missing = tmp_path / "missing_source.pth"
    restored = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                **updated,
                "model_path": str(missing),
            },
            "fitting_net": {
                "type": "property",
                "property_name": "band_gap",
                "task_dim": 1,
                "neuron": [8],
                "precision": "float64",
            },
        },
    )
    validate = restored.get_descriptor().backbone.load_state_dict(
        original.backbone.state_dict(),
        strict=False,
    )
    assert not validate.missing_keys
    torch.testing.assert_close(
        _descriptor_output(restored.get_descriptor()),
        _descriptor_output(original),
    )


def test_checkpoint_feature_parity_and_serialization_roundtrip(
    sevennet_checkpoint: Path,
) -> None:
    """Checkpoint state and features survive DeePMD serialization."""
    descriptor = SevenNetDescriptor(
        model_path=sevennet_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    expected = _descriptor_output(descriptor)
    serialized = descriptor.serialize()
    assert serialized["model_path"] is None
    json.dumps({k: v for k, v in serialized.items() if k != "@variables"})
    restored = BaseDescriptor.deserialize(serialized)
    actual = _descriptor_output(restored)
    torch.testing.assert_close(actual, expected)
    for name, value in descriptor.backbone.state_dict().items():
        torch.testing.assert_close(
            value.cpu(),
            restored.backbone.state_dict()[name].cpu(),
        )


def test_dp_property_training_smoke(
    sevennet_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Run one real ``dp --pt train`` property step and restore without the source."""
    system = tmp_path / "property_data"
    data_set = system / "set.000"
    data_set.mkdir(parents=True)
    (system / "type.raw").write_text("1\n0\n0\n")
    (system / "type_map.raw").write_text("H\nO\n")
    coordinates = np.array(
        [
            [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]],
            [[0.0, 0.0, 0.0], [0.8, 0.2, 0.1], [-0.1, 0.9, 0.4]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.2], [-0.3, 1.1, 0.2]],
            [[0.0, 0.0, 0.0], [0.7, 0.3, 0.0], [-0.2, 0.8, 0.5]],
        ],
        dtype=np.float64,
    )
    np.save(data_set / "coord.npy", coordinates.reshape(4, -1))
    np.save(
        data_set / "box.npy",
        np.tile((np.eye(3) * 8.0).reshape(1, 9), (4, 1)),
    )
    np.save(
        data_set / "band_gap.npy",
        np.array([[0.2], [0.7], [-0.4], [1.1]], dtype=np.float64),
    )

    input_data = {
        "model": {
            "type": "standard",
            "type_map": ["H", "O"],
            "data_stat_nbatch": 1,
            "descriptor": {
                "type": "sevennet",
                "model_path": str(sevennet_checkpoint.resolve()),
                "sel": 16,
            },
            "fitting_net": {
                "type": "property",
                "property_name": "band_gap",
                "task_dim": 1,
                "neuron": [8],
                "precision": "float64",
                "seed": 11,
            },
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 1,
            "start_lr": 1e-2,
            "stop_lr": 1e-3,
        },
        "loss": {
            "type": "property",
            "loss_func": "mse",
        },
        "optimizer": {
            "type": "Adam",
        },
        "training": {
            "training_data": {
                "systems": [str(system.resolve())],
                "batch_size": 2,
            },
            "numb_steps": 1,
            "seed": 17,
            "disp_freq": 1,
            "save_freq": 1,
            "save_ckpt": "model.ckpt",
        },
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(input_data))
    env_vars = os.environ.copy()
    env_vars.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run(
        [sys.executable, "-m", "deepmd", "--pt", "train", input_path.name],
        cwd=tmp_path,
        env=env_vars,
        check=True,
        timeout=180,
    )

    checkpoint_pointer = tmp_path / "checkpoint"
    assert checkpoint_pointer.is_file()
    saved_checkpoint = Path(checkpoint_pointer.read_text().strip())
    if not saved_checkpoint.is_absolute():
        saved_checkpoint = tmp_path / saved_checkpoint
    assert saved_checkpoint.is_file()

    extra_params = torch.load(
        saved_checkpoint,
        map_location="cpu",
        weights_only=False,
    )["model"]["_extra_state"]["model_params"]
    assert extra_params["descriptor"]["config"]["type_map"] == ["H", "O"]
    hidden_source = sevennet_checkpoint.with_name("hidden_source.pth")
    sevennet_checkpoint.rename(hidden_source)
    from deepmd.pt.infer.inference import Tester  # noqa: PLC0415
    from deepmd.pt.utils.env import DEVICE  # noqa: PLC0415

    tester = Tester(str(saved_checkpoint))
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [-0.2, 1.0, 0.3]]],
        dtype=torch.float64,
        device=DEVICE,
    ).reshape(1, -1)
    atype = torch.tensor([[1, 0, 0]], dtype=torch.int64, device=DEVICE)
    prediction, _, _ = tester.wrapper(coord, atype)
    assert torch.isfinite(prediction["band_gap"]).all()


def test_persistable_config_roundtrip_is_json_safe(sevennet_checkpoint: Path) -> None:
    """update_sel config can be written into a DeePMD JSON model definition."""
    persistable = load_sevennet_checkpoint_config(sevennet_checkpoint)
    json.dumps(persistable)
    again = persistable_checkpoint_config(persistable)
    assert again["type_map"] == ["H", "O"]
    reject_unsupported_sevennet(again["sevenn_config"])


@pytest.mark.slow
def test_bundled_sevennet0_inspect_only() -> None:
    """Inspect the packaged 7net-0 metadata without building the large backbone."""
    path = Path(sevenn_const.SEVENNET_0_11Jul2024)
    if not path.is_file():
        pytest.skip("bundled 7net-0 checkpoint is not installed")
    persistable = load_sevennet_checkpoint_config(path)
    assert persistable["cutoff"] == pytest.approx(5.0)
    assert len(persistable["type_map"]) == 89
    assert persistable["type_map"][0] == "Ac"
    json.dumps(persistable)
