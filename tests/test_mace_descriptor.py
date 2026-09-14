# SPDX-License-Identifier: LGPL-3.0-or-later
"""Focused tests for the MACE property descriptor."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from deepmd.pt.model.descriptor.base_descriptor import BaseDescriptor
from deepmd.pt.model.model import get_model
from deepmd.pt.utils import env
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list
from e3nn import o3
from mace.calculators import mace_mp
from mace.modules import MACE

from deepmd_gnn.mace_checkpoint import (
    inspect_native_mace_checkpoint,
    load_native_mace_checkpoint,
    validate_mace_state_dict_load,
)
from deepmd_gnn.mace_descriptor import (
    MaceDescriptor,
    _scalar_even_indices,
)
from deepmd_gnn.mace_network import make_mace_network
from deepmd_gnn.mace_off import download_mace_off_model


def _write_mace_checkpoint(
    path: Path,
    *,
    keep_last_layer_irreps: bool,
    pair_repulsion: bool = False,
    interaction_first: str = "RealAgnosticInteractionBlock",
    heads: list[str] | None = None,
    correlation: int | list[int] = 2,
    num_interactions: int = 2,
    gate: str = "silu",
) -> Path:
    """Write a tiny native checkpoint with no network access."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = make_mace_network(
            r_max=3.0,
            num_radial_basis=4,
            num_cutoff_basis=5,
            max_ell=1,
            interaction_first=interaction_first,
            interaction="RealAgnosticResidualInteractionBlock",
            num_interactions=num_interactions,
            num_elements=2,
            hidden_irreps="2x0e + 2x1o",
            atomic_numbers=[1, 8],
            avg_num_neighbors=4.0,
            pair_repulsion=pair_repulsion,
            distance_transform="None",
            correlation=correlation,
            gate=gate,
            MLP_irreps="4x0e",
            std=1.0,
            radial_MLP=[8, 8],
            radial_type="bessel",
            enable_cueq=False,
            script_model=False,
            keep_last_layer_irreps=keep_last_layer_irreps,
            heads=heads,
        )
        with torch.no_grad():
            atomic_energies = model.atomic_energies_fn.atomic_energies
            atomic_energies.copy_(
                torch.linspace(
                    3.0,
                    4.0,
                    atomic_energies.numel(),
                    dtype=atomic_energies.dtype,
                    device=atomic_energies.device,
                ).reshape_as(atomic_energies),
            )
    finally:
        torch.set_default_dtype(old_dtype)
    torch.save(model, path)
    return path


@pytest.fixture
def mace_checkpoint(tmp_path: Path) -> Path:
    """Create a MACE-0.3.5-compatible trusted local checkpoint."""
    return _write_mace_checkpoint(
        tmp_path / "small_mace.model",
        keep_last_layer_irreps=False,
    )


@pytest.fixture
def mixed_mace_checkpoint(tmp_path: Path) -> Path:
    """Create a mixed-final-irrep checkpoint when supported by MACE."""
    if "keep_last_layer_irreps" not in inspect.signature(MACE.__init__).parameters:
        pytest.skip("installed MACE predates mixed final product irreps")
    return _write_mace_checkpoint(
        tmp_path / "mixed_mace.model",
        keep_last_layer_irreps=True,
    )


@pytest.fixture
def mpa_like_checkpoint(tmp_path: Path) -> Path:
    """Combine the architecture details that distinguish MACE-MPA-0."""
    return _write_mace_checkpoint(
        tmp_path / "mpa-like.model",
        keep_last_layer_irreps=True,
        pair_repulsion=True,
        interaction_first="RealAgnosticResidualInteractionBlock",
        heads=["default"],
        correlation=[2, 3],
    )


def _inputs(
    descriptor: MaceDescriptor,
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
    descriptor: MaceDescriptor,
    coord: torch.Tensor | None = None,
) -> torch.Tensor:
    coord_ext, atype_ext, nlist, mapping = _inputs(descriptor, coord)
    return descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0]


def test_constructor_registry_and_metadata(mace_checkpoint: Path) -> None:
    """Descriptor registration and metadata follow DeePMD's contract."""
    descriptor = BaseDescriptor(
        type="mace",
        model_path=str(mace_checkpoint),
        sel=16,
        type_map=["H", "O"],
        ntypes=2,
    )
    assert isinstance(descriptor, MaceDescriptor)
    assert descriptor.get_type_map() == ["H", "O"]
    assert descriptor.get_sel() == [16]
    assert descriptor.get_nsel() == 16
    assert descriptor.get_rcut() == pytest.approx(3.0)
    assert descriptor.get_dim_out() == 2
    assert descriptor.mixed_types()
    assert descriptor.has_message_passing()
    assert not descriptor.has_message_passing_across_ranks()
    assert all(parameter.requires_grad for parameter in descriptor.parameters())
    descriptor.set_stat_mean_and_stddev(torch.ones(1), torch.ones(1))
    mean, stddev = descriptor.get_stat_mean_and_stddev()
    assert mean.numel() == 0
    assert stddev.numel() == 0

    local_config = {
        "type": "mace",
        "model_path": str(mace_checkpoint),
        "sel": 16,
    }
    updated, min_distance = BaseDescriptor.update_sel(
        None,
        ["H", "O"],
        local_config,
    )
    assert updated["model_path"] == str(mace_checkpoint)
    assert updated["sel"] == 16
    assert updated["config"]["type_map"] == ["H", "O"]
    assert updated["config"]["hidden_irreps"]
    json.dumps(updated)
    assert min_distance is None


def test_descriptor_supports_first_import_in_fresh_process() -> None:
    """Importing the descriptor first must not recurse through the PT entry point."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from deepmd_gnn.mace_descriptor import MaceDescriptor; "
                "assert MaceDescriptor.__name__ == 'MaceDescriptor'"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_constructor_requires_exact_checkpoint_type_map(
    mace_checkpoint: Path,
) -> None:
    """Checkpoint element order must equal the model type map."""
    with pytest.raises(ValueError, match="exactly match checkpoint"):
        MaceDescriptor(
            model_path=mace_checkpoint,
            sel=16,
            type_map=["O", "H"],
        )


def test_constructor_rejects_truly_multi_head_checkpoint(tmp_path: Path) -> None:
    """Only one named MACE head can define a descriptor backbone."""
    checkpoint = _write_mace_checkpoint(
        tmp_path / "multi-head.model",
        keep_last_layer_irreps=False,
        heads=["first", "second"],
    )
    with pytest.raises(ValueError, match="Multi-head MACE descriptors"):
        MaceDescriptor(
            model_path=checkpoint,
            sel=16,
            type_map=["H", "O"],
        )


def test_feature_backbone_accepts_linear_or_ungated_readouts(tmp_path: Path) -> None:
    """Energy-head readout metadata is unused and must not block inspection."""
    linear_path = _write_mace_checkpoint(
        tmp_path / "linear_readout.model",
        keep_last_layer_irreps=False,
        num_interactions=1,
    )
    linear_native = load_native_mace_checkpoint(linear_path, device=env.DEVICE)
    linear_config = inspect_native_mace_checkpoint(linear_native)
    assert not hasattr(linear_native.readouts[-1], "hidden_irreps")
    assert linear_config["MLP_irreps"]
    assert linear_config["gate"]
    linear_descriptor = MaceDescriptor(
        model_path=linear_path,
        sel=16,
        type_map=["H", "O"],
    )
    assert _descriptor_output(linear_descriptor).shape[-1] == 2

    ungated_path = _write_mace_checkpoint(
        tmp_path / "ungated_readout.model",
        keep_last_layer_irreps=False,
        gate="None",
    )
    ungated_native = load_native_mace_checkpoint(ungated_path, device=env.DEVICE)
    ungated_config = inspect_native_mace_checkpoint(ungated_native)
    assert ungated_config["gate"] == "None"
    ungated_descriptor = MaceDescriptor(
        model_path=ungated_path,
        sel=16,
        type_map=["H", "O"],
    )
    assert _descriptor_output(ungated_descriptor).shape[-1] == 2


def test_deserialize_keeps_derived_zeroed_buffers(mace_checkpoint: Path) -> None:
    """Older native pickles omit reconstructed ``*_zeroed`` buffers."""
    validate_mace_state_dict_load(
        SimpleNamespace(
            missing_keys=["products.0.symmetric_contractions.weights_0_zeroed"],
            unexpected_keys=[],
        ),
    )
    with pytest.raises(RuntimeError, match=r"node_embedding\.linear\.weight"):
        validate_mace_state_dict_load(
            SimpleNamespace(
                missing_keys=["node_embedding.linear.weight"],
                unexpected_keys=[],
            ),
        )

    descriptor = MaceDescriptor(
        model_path=mace_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    serialized = descriptor.serialize()
    serialized["@variables"] = {
        name: value
        for name, value in serialized["@variables"].items()
        if not name.endswith("_zeroed")
    }
    restored = BaseDescriptor.deserialize(serialized)
    torch.testing.assert_close(
        _descriptor_output(restored),
        _descriptor_output(descriptor),
    )


def test_legacy_checkpoint_supports_strict_scripted_state_restore(
    mace_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Derived flags must exist before a training checkpoint is saved."""
    native = torch.load(mace_checkpoint, map_location="cpu", weights_only=False)
    for module in native.modules():
        for name, _ in list(module.named_buffers(recurse=False)):
            if name.endswith("_zeroed"):
                delattr(module, name)
    legacy_path = tmp_path / "legacy.model"
    torch.save(native, legacy_path)
    descriptor = MaceDescriptor(
        model_path=legacy_path,
        sel=16,
        type_map=["H", "O"],
    )
    restored = MaceDescriptor(sel=16, config=descriptor.config, type_map=["H", "O"])
    scripted = torch.jit.script(restored)
    scripted.load_state_dict(descriptor.state_dict())
    inputs = _inputs(descriptor)
    torch.testing.assert_close(scripted(*inputs)[0], descriptor(*inputs)[0])


def test_mpa_like_checkpoint_roundtrip_and_gradient(
    mpa_like_checkpoint: Path,
) -> None:
    """MPA-like native checkpoints retain generic backbone architecture."""
    descriptor = MaceDescriptor(
        model_path=mpa_like_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    assert descriptor.config["interaction_first"] == (
        "RealAgnosticResidualInteractionBlock"
    )
    assert descriptor.config["interaction"] == ("RealAgnosticResidualInteractionBlock")
    assert descriptor.config["heads"] == ["default"]
    assert descriptor.config["pair_repulsion"] is True
    assert descriptor.config["correlation"] == [2, 3]
    assert descriptor.config["keep_last_layer_irreps"] is True
    state_names = set(descriptor.backbone.state_dict())
    assert not any("pair_repulsion" in name for name in state_names)
    assert not any("atomic_energies" in name for name in state_names)
    assert not any("readout" in name for name in state_names)
    assert not any("scale_shift" in name for name in state_names)

    output = _descriptor_output(descriptor)
    output.square().sum().backward()
    assert output.shape == (1, 3, 2)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in descriptor.backbone.parameters()
    )

    restored = MaceDescriptor.deserialize(descriptor.serialize())
    torch.testing.assert_close(
        _descriptor_output(restored),
        output.detach(),
    )


@pytest.mark.slow
def test_real_medium_mpa0_checkpoint_with_official_helper(tmp_path: Path) -> None:
    """Exercise official medium-mpa-0 when a local path or network is enabled."""
    model_source = os.environ.get("MACE_MPA0_PATH")
    if model_source is None:
        if os.environ.get("RUN_NETWORK_TESTS") != "1":
            pytest.skip(
                "set MACE_MPA0_PATH or RUN_NETWORK_TESTS=1 for medium-mpa-0",
            )
        model_source = "medium-mpa-0"
    native = mace_mp(
        model=model_source,
        device=str(env.DEVICE),
        return_raw_model=True,
    )
    model_path = tmp_path / "medium-mpa-0.model"
    torch.save(native, model_path)
    native = load_native_mace_checkpoint(model_path, device=env.DEVICE)
    config = inspect_native_mace_checkpoint(native)
    descriptor = MaceDescriptor(
        model_path=model_path,
        sel=64,
        type_map=config["type_map"],
    )
    assert config["interaction_first"] == native.interactions[0].__class__.__name__
    assert config["interaction"] == native.interactions[-1].__class__.__name__
    assert config["heads"] == list(native.heads)
    assert config["pair_repulsion"] is bool(
        getattr(native, "pair_repulsion", False),
    )
    output = _descriptor_output(descriptor)
    output.square().sum().backward()
    assert torch.isfinite(output).all()
    assert any(parameter.grad is not None for parameter in descriptor.parameters())


def test_forward_shape_and_rotation_invariance(mace_checkpoint: Path) -> None:
    """Final ``0e`` features have local-atom shape and are invariant."""
    descriptor = MaceDescriptor(
        model_path=mace_checkpoint,
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
    assert output.shape == (1, 3, 2)
    torch.testing.assert_close(output, rotated, rtol=2e-6, atol=2e-7)


def test_descriptor_forward_is_torchscriptable(mace_checkpoint: Path) -> None:
    """``dp test`` scripts the eager checkpoint; forward must not use Python builtins."""
    descriptor = MaceDescriptor(
        model_path=mace_checkpoint,
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
                "type": "mace",
                "model_path": str(mace_checkpoint),
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
    helper_buffers = (
        "_scalar_indices",
        "_stat_mean",
        "_stat_stddev",
        "_backbone_probe",
    )
    eager_state = model.state_dict()
    assert not any(any(name in key for name in helper_buffers) for key in eager_state)
    scripted_model = torch.jit.script(model)
    scripted_model.load_state_dict(eager_state)
    actual_pred = scripted_model(coord, atype)["band_gap"]
    torch.testing.assert_close(actual_pred, expected_pred)


def test_periodic_mapping_and_coordinate_gradient(mace_checkpoint: Path) -> None:
    """Periodic images follow MaceModel mapping without breaking gradients."""
    descriptor = MaceDescriptor(
        model_path=mace_checkpoint,
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
    embedded_node_counts = []

    def record_node_count(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        embedded_node_counts.append(inputs[0].shape[0])

    handle = descriptor.backbone.node_embedding.register_forward_pre_hook(
        record_node_count,
    )
    try:
        output = descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0]
    finally:
        handle.remove()
    assert embedded_node_counts == [nlist.shape[0] * nlist.shape[1]]
    output.square().sum().backward()
    assert coord.grad is not None
    assert torch.count_nonzero(coord.grad)

    translated = coord.detach().clone()
    translated[:, 1, 0] -= 4.0
    translated_ext, translated_type, translated_nlist, translated_mapping = _inputs(
        descriptor,
        translated,
        box,
    )
    translated_output = descriptor(
        translated_ext,
        translated_type,
        translated_nlist,
        mapping=translated_mapping,
    )[0]
    torch.testing.assert_close(output, translated_output, rtol=2e-6, atol=2e-7)


def test_final_features_match_native_mace_forward(mace_checkpoint: Path) -> None:
    """The compact graph adapter preserves MACE's final product features."""
    native = load_native_mace_checkpoint(mace_checkpoint, device=env.DEVICE)
    descriptor = MaceDescriptor(
        model_path=mace_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    coord_ext, atype_ext, nlist, mapping = _inputs(descriptor)
    del mapping
    nf, nloc, _ = nlist.shape
    source_dtype = next(native.parameters()).dtype
    positions = coord_ext.view(nf * nloc, 3).to(source_dtype)
    atype = atype_ext[:, :nloc].to(torch.int64)
    edge_index = torch.ops.deepmd_gnn.edge_index(
        nlist.to(torch.int64),
        atype_ext.to(torch.int64),
        torch.empty(0, dtype=torch.int64, device="cpu"),
    ).T
    node_attrs = torch.zeros(
        (nf * nloc, descriptor.ntypes),
        dtype=source_dtype,
        device=env.DEVICE,
    )
    node_attrs.scatter_(-1, atype.reshape(-1, 1), 1)
    native_output = native(
        {
            "positions": positions,
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "shifts": torch.zeros(
                (edge_index.shape[1], 3),
                dtype=source_dtype,
                device=env.DEVICE,
            ),
            "cell": torch.zeros(
                (nf, 3, 3),
                dtype=source_dtype,
                device=env.DEVICE,
            ),
            "batch": torch.arange(nf, device=env.DEVICE)
            .unsqueeze(-1)
            .expand(nf, nloc)
            .reshape(-1),
            "ptr": torch.arange(
                0,
                (nf + 1) * nloc,
                nloc,
                dtype=torch.int64,
                device=env.DEVICE,
            ),
        },
        compute_force=False,
    )["node_feats"]
    assert native_output is not None
    final_dim = native.products[-1].linear.irreps_out.dim
    native_final = native_output[:, -final_dim:].view(nf, nloc, final_dim)
    expected = torch.index_select(
        native_final,
        -1,
        torch.tensor(
            descriptor.scalar_even_indices,
            dtype=torch.int64,
            device=env.DEVICE,
        ),
    ).to(env.GLOBAL_PT_FLOAT_PRECISION)
    actual = descriptor(coord_ext, atype_ext, nlist)[0]
    torch.testing.assert_close(actual, expected)


def test_compact_periodic_mapping_handles_multiple_frames(
    mace_checkpoint: Path,
) -> None:
    """Compact node indices preserve independent frames with periodic images."""
    descriptor = MaceDescriptor(
        model_path=mace_checkpoint,
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
    assert atype_ext.shape[1] > nlist.shape[1]
    batched = descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0]

    per_frame = []
    for frame in range(2):
        frame_coord_ext, frame_atype_ext, frame_mapping, frame_nlist = (
            extend_input_and_build_neighbor_list(
                coord[frame : frame + 1].reshape(1, -1),
                atype[frame : frame + 1],
                descriptor.get_rcut(),
                descriptor.get_sel(),
                mixed_types=True,
                box=box[frame : frame + 1],
            )
        )
        per_frame.append(
            descriptor(
                frame_coord_ext,
                frame_atype_ext,
                frame_nlist,
                mapping=frame_mapping,
            )[0],
        )
    torch.testing.assert_close(batched, torch.cat(per_frame))


def test_mixed_irrep_selects_only_zero_even_content(
    mixed_mace_checkpoint: Path,
) -> None:
    """Mixed outputs exclude vector and odd scalar channels."""
    assert _scalar_even_indices(o3.Irreps("2x0e + 1x1o + 1x0o")) == [0, 1]
    descriptor = MaceDescriptor(
        model_path=mixed_mace_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    coord_ext, atype_ext, nlist, mapping = _inputs(descriptor)
    all_features = descriptor._final_product_features(  # noqa: SLF001
        coord_ext,
        atype_ext,
        nlist,
        mapping,
    )
    output = descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0]
    assert descriptor.output_irreps == "2x0e+2x1o"
    torch.testing.assert_close(output, all_features[..., :2])


def test_property_model_composition_and_optimizer_step(
    mace_checkpoint: Path,
) -> None:
    """Property fitting updates both backbone and fitting parameters."""
    model = get_model(
        {
            "type": "standard",
            "type_map": ["H", "O"],
            "descriptor": {
                "type": "mace",
                "model_path": str(mace_checkpoint),
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
    assert torch.count_nonzero(model.atomic_model.out_bias) == 0
    backbone_parameter = next(descriptor.backbone.node_embedding.parameters())
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
    mace_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Saved model definitions reconstruct the backbone without the native pickle."""
    updated, _ = MaceDescriptor.update_sel(
        None,
        ["H", "O"],
        {
            "type": "mace",
            "model_path": str(mace_checkpoint),
            "sel": 16,
        },
    )
    original = MaceDescriptor(
        model_path=mace_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    missing = tmp_path / "missing_source.model"
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
    validate_mace_state_dict_load(
        restored.get_descriptor().backbone.load_state_dict(
            original.backbone.state_dict(),
            strict=False,
        ),
    )
    torch.testing.assert_close(
        _descriptor_output(restored.get_descriptor()),
        _descriptor_output(original),
    )


def test_dp_property_training_smoke(
    mace_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """Run one real ``dp --pt train`` property step and save backbone state."""
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
                "type": "mace",
                "model_path": str(mace_checkpoint.resolve()),
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
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run(
        [sys.executable, "-m", "deepmd", "--pt", "train", input_path.name],
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=120,
    )

    checkpoint_pointer = tmp_path / "checkpoint"
    assert checkpoint_pointer.is_file()
    saved_checkpoint = Path(checkpoint_pointer.read_text().strip())
    if not saved_checkpoint.is_absolute():
        saved_checkpoint = tmp_path / saved_checkpoint
    assert saved_checkpoint.is_file()

    initial_state = torch.load(
        mace_checkpoint,
        map_location="cpu",
        weights_only=False,
    ).state_dict()
    trained_state = torch.load(
        saved_checkpoint,
        map_location="cpu",
        weights_only=False,
    )["model"]
    compared = 0
    changed = 0
    for source_name, initial_value in initial_state.items():
        if not initial_value.is_floating_point():
            continue
        suffix = f"descriptor.backbone.{source_name}"
        matches = [name for name in trained_state if name.endswith(suffix)]
        if not matches:
            continue
        compared += 1
        if not torch.equal(initial_value.cpu(), trained_state[matches[0]].cpu()):
            changed += 1
    assert compared > 0
    assert changed > 0

    extra_params = trained_state["_extra_state"]["model_params"]
    assert extra_params["descriptor"]["config"]["type_map"] == ["H", "O"]
    hidden_source = mace_checkpoint.with_name("hidden_source.model")
    mace_checkpoint.rename(hidden_source)
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

    subprocess.run(
        [
            sys.executable,
            "-m",
            "deepmd",
            "--pt",
            "train",
            input_path.name,
            "--init-model",
            str(saved_checkpoint),
            "--use-pretrain-script",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=120,
    )


def test_checkpoint_feature_parity_and_serialization_roundtrip(
    mace_checkpoint: Path,
) -> None:
    """Checkpoint state and features survive DeePMD serialization."""
    descriptor = MaceDescriptor(
        model_path=mace_checkpoint,
        sel=16,
        type_map=["H", "O"],
    )
    original_state = torch.load(
        mace_checkpoint,
        map_location="cpu",
        weights_only=False,
    ).state_dict()
    for name, value in descriptor.backbone.state_dict().items():
        torch.testing.assert_close(value.cpu(), original_state[name].cpu())

    expected = _descriptor_output(descriptor)
    serialized = descriptor.serialize()
    assert serialized["model_path"] is None
    restored = BaseDescriptor.deserialize(serialized)
    actual = _descriptor_output(restored)
    torch.testing.assert_close(actual, expected)
    for name, value in descriptor.backbone.state_dict().items():
        torch.testing.assert_close(
            value.cpu(),
            restored.backbone.state_dict()[name].cpu(),
        )


@pytest.mark.slow
def test_off23_small_checkpoint_features_and_gradients(tmp_path: Path) -> None:
    """The official MACE-OFF23-small checkpoint is a trainable descriptor."""
    model_path = download_mace_off_model("off23_small", cache_dir=tmp_path)
    descriptor = MaceDescriptor(
        model_path=model_path,
        sel=64,
        type_map=["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"],
    )
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9266, 0.0]]],
        dtype=torch.float64,
        device=env.DEVICE,
    )
    atype = torch.tensor([[3, 0, 0]], dtype=torch.int64, device=env.DEVICE)
    coord_ext, atype_ext, mapping, nlist = extend_input_and_build_neighbor_list(
        coord.reshape(1, -1),
        atype,
        descriptor.get_rcut(),
        descriptor.get_sel(),
        mixed_types=True,
        box=None,
    )
    output = descriptor(coord_ext, atype_ext, nlist, mapping=mapping)[0]
    output.square().mean().backward()

    assert output.shape == (1, 3, 96)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in descriptor.backbone.parameters()
    )
    restored = BaseDescriptor.deserialize(descriptor.serialize())
    torch.jit.script(restored).load_state_dict(descriptor.state_dict())
    torch.testing.assert_close(
        restored(coord_ext, atype_ext, nlist, mapping=mapping)[0],
        output.detach(),
    )
