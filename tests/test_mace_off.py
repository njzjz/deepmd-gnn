# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for conservative MACE-OFF loading helpers."""

from __future__ import annotations

import tempfile
import urllib.error
from pathlib import Path
from typing import Any

import deepmd.pt.model  # noqa: F401
import pytest
import torch
from deepmd.pt.utils.nlist import extend_input_and_build_neighbor_list

from deepmd_gnn.mace import MaceModel
from deepmd_gnn.mace_network import make_mace_network
from deepmd_gnn.mace_off import (
    _infer_deepmd_config,
    _load_mace_checkpoint,
    convert_mace_off_to_deepmd,
    download_mace_off_model,
    load_mace_off_model,
)
from deepmd_gnn.mace_off_cli import (
    MACE_OFF_MODEL_SHA256,
    MACE_OFF_MODEL_URLS,
    MACE_OFF_MODELS,
    _download_file,
    get_mace_off_cache_dir,
)


def _native_mace_reference_outputs(
    mace_model: torch.nn.Module,
    *,
    coord: torch.Tensor,
    atype: torch.Tensor,
    box: torch.Tensor,
    sel: int,
    ntypes: int,
) -> dict[str, torch.Tensor]:
    """Evaluate a native MACE checkpoint on the same graph path as the wrapper."""
    nloc = atype.shape[1]
    extended_coord, extended_atype, mapping, nlist = (
        extend_input_and_build_neighbor_list(
            coord,
            atype,
            float(mace_model.r_max),
            [sel],
            mixed_types=True,
            box=box,
        )
    )
    nf, nall = extended_atype.shape

    extended_coord = extended_coord.view(nf, nall, 3)
    extended_coord_ff = extended_coord.view(nf * nall, 3)
    default_dtype = mace_model.atomic_energies_fn.atomic_energies.dtype
    extended_coord_ff = extended_coord_ff.to(default_dtype)
    extended_coord_ff.requires_grad_(requires_grad=True)

    extended_atype = extended_atype.to(torch.int64)
    edge_index = torch.ops.deepmd_gnn.edge_index(
        nlist.to(torch.int64),
        extended_atype,
        torch.tensor([], dtype=torch.int64, device="cpu"),
    ).T

    shifts = torch.zeros(
        (edge_index.shape[1], 3),
        dtype=default_dtype,
        device=extended_coord_ff.device,
    )
    if mapping is not None and nloc < nall:
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

    one_hot = torch.zeros(
        (nf * nall, ntypes),
        dtype=default_dtype,
        device=extended_coord_ff.device,
    )
    one_hot.scatter_(
        dim=-1,
        index=extended_atype.view(nf * nall).unsqueeze(-1),
        value=1,
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
    weight = torch.ones(
        [nf],
        dtype=default_dtype,
        device=extended_coord_ff.device,
    )
    cell = box.view(nf, 3, 3).to(default_dtype).to(extended_coord_ff.device)
    edge_batch = torch.div(edge_index[0], nall, rounding_mode="floor")
    inv_box = torch.linalg.inv(cell)
    unit_shifts = torch.einsum("ec,ecb->eb", shifts, inv_box[edge_batch])
    displacement = torch.zeros(
        (nf, 3, 3),
        dtype=extended_coord_ff.dtype,
        device=extended_coord_ff.device,
    )
    displacement.requires_grad_(requires_grad=True)
    symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
    positions = extended_coord_ff + torch.einsum(
        "be,bec->bc",
        extended_coord_ff,
        symmetric_displacement[batch],
    )
    cell_def = cell + torch.matmul(cell, symmetric_displacement)

    ret = mace_model.forward(
        {
            "positions": positions,
            "cell": cell_def,
            "shifts": torch.einsum(
                "be,bec->bc",
                torch.round(unit_shifts).to(default_dtype),
                cell_def[edge_batch],
            ),
            "edge_index": edge_index,
            "batch": batch,
            "node_attrs": one_hot,
            "ptr": ptr,
            "weight": weight,
        },
        compute_force=False,
        compute_virials=False,
        compute_stress=False,
        compute_displacement=False,
        training=False,
    )

    atom_energy = ret["node_energy"]
    if atom_energy is None:
        msg = "Native MACE model returned no node_energy"
        raise ValueError(msg)
    atom_energy = atom_energy.view(nf, nall)[:, :nloc]
    energy = atom_energy.sum(dim=1).view(nf)

    grads = torch.autograd.grad(
        outputs=[energy],
        inputs=[extended_coord_ff, displacement],
        grad_outputs=[torch.ones_like(energy)],
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    force = grads[0]
    virial = grads[1]
    if force is None:
        msg = "Native MACE model returned no force gradient"
        raise ValueError(msg)
    force = -force.view(nf, nall, 3)
    if virial is None:
        virial_out = torch.zeros((nf, 9), dtype=coord.dtype, device=force.device)
    else:
        virial_out = (-virial).view(nf, 9).to(coord.dtype)

    if mapping is not None:
        force_local = torch.scatter_reduce(
            torch.zeros((nf, nloc, 3), dtype=coord.dtype, device=force.device),
            1,
            mapping.unsqueeze(-1).expand(nf, nall, 3),
            force.to(coord.dtype),
            reduce="sum",
        )
        force_out = force_local
    else:
        force_out = force[:, :nloc, :].to(coord.dtype)

    return {
        "energy": energy.view(nf, 1).to(coord.dtype),
        "force": force_out,
        "virial": virial_out,
    }


def test_mace_off_model_urls_are_raw_github_paths() -> None:
    """Canonical model URLs should point to the real raw GitHub checkpoint paths."""
    assert MACE_OFF_MODELS["off23_small"].endswith("mace_off23/MACE-OFF23_small.model")
    assert MACE_OFF_MODELS["off23_medium"].endswith(
        "mace_off23/MACE-OFF23_medium.model",
    )
    assert MACE_OFF_MODELS["off23_large"].endswith("mace_off23/MACE-OFF23_large.model")
    assert MACE_OFF_MODELS["off24_medium"].endswith(
        "mace_off24/MACE-OFF24_medium.model",
    )


def test_mace_off_model_urls_include_ghfast_mirror() -> None:
    """Each model should expose both canonical GitHub and ghfast mirror URLs."""
    for model_name, canonical_url in MACE_OFF_MODELS.items():
        assert MACE_OFF_MODEL_URLS[model_name] == [
            canonical_url,
            f"https://ghfast.top/{canonical_url}",
        ]


def test_mace_off_model_sha256_map_has_all_supported_models() -> None:
    """Every supported official checkpoint should have an expected SHA256."""
    assert set(MACE_OFF_MODEL_SHA256) == set(MACE_OFF_MODELS)
    assert all(len(digest) == 64 for digest in MACE_OFF_MODEL_SHA256.values())


def test_get_mace_off_cache_dir_uses_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cache directory should honor XDG_CACHE_HOME when it is set."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cache_dir = get_mace_off_cache_dir()
    assert cache_dir == tmp_path / "deepmd-gnn" / "mace-off"
    assert cache_dir.exists()


def test_download_mace_off_model_rejects_unknown_name(tmp_path: Path) -> None:
    """Unknown checkpoint names should fail clearly."""
    with pytest.raises(ValueError, match="Unknown MACE-OFF model"):
        download_mace_off_model("not-a-model", cache_dir=tmp_path)


def test_download_mace_off_model_uses_alias_and_cached_file(tmp_path: Path) -> None:
    """Compatibility aliases should resolve to the canonical cached filename."""
    cached_file = tmp_path / "MACE-OFF23_small.model"
    cached_file.write_bytes(b"cached")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "deepmd_gnn.mace_off_cli._validate_model_file",
            lambda *_args, **_kwargs: True,
        )
        model_path = download_mace_off_model("small", cache_dir=tmp_path)
    assert model_path == cached_file


def test_download_mace_off_model_redownloads_on_sha256_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cached files with mismatched SHA256 should be replaced from the source URL."""
    cached_file = tmp_path / "MACE-OFF23_small.model"
    cached_file.write_bytes(b"stale")
    recorded: dict[str, str] = {}
    sha_calls = {"count": 0}

    def fake_download_file(url: str, destination: Path) -> None:
        recorded["url"] = url
        recorded["destination"] = str(destination)
        Path(destination).write_bytes(b"downloaded")

    def fake_sha256sum(_path: Path) -> str:
        sha_calls["count"] += 1
        if sha_calls["count"] == 1:
            return "different"
        return MACE_OFF_MODEL_SHA256["off23_small"]

    monkeypatch.setattr("deepmd_gnn.mace_off_cli._download_file", fake_download_file)
    monkeypatch.setattr("deepmd_gnn.mace_off_cli._sha256sum", fake_sha256sum)
    monkeypatch.setattr(
        "deepmd_gnn.mace_off_cli._rank_download_urls",
        lambda urls: urls,
    )

    model_path = download_mace_off_model("off23_small", cache_dir=tmp_path)

    assert model_path.exists()
    assert model_path.read_bytes() == b"downloaded"
    assert recorded["url"] == MACE_OFF_MODELS["off23_small"]
    assert Path(recorded["destination"]) == cached_file


def test_download_mace_off_model_rejects_bad_download_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Downloads should fail if the fetched file does not match the expected SHA256."""
    monkeypatch.setattr(
        "deepmd_gnn.mace_off_cli._download_file",
        lambda _url, destination: Path(destination).write_bytes(b"bad"),
    )
    monkeypatch.setattr(
        "deepmd_gnn.mace_off_cli._sha256sum",
        lambda _path: "bad-digest",
    )
    monkeypatch.setattr(
        "deepmd_gnn.mace_off_cli._rank_download_urls",
        lambda urls: urls[:1],
    )

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        download_mace_off_model("off23_small", cache_dir=tmp_path)


def test_download_mace_off_model_falls_back_to_ghfast_mirror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Mirror URLs should be attempted when the canonical GitHub source fails."""
    attempts: list[str] = []

    def fake_download_file(url: str, destination: Path) -> None:
        attempts.append(url)
        if len(attempts) == 1:
            msg = "primary unavailable"
            raise urllib.error.URLError(msg)
        Path(destination).write_bytes(b"downloaded")

    monkeypatch.setattr("deepmd_gnn.mace_off_cli._download_file", fake_download_file)
    monkeypatch.setattr(
        "deepmd_gnn.mace_off_cli._sha256sum",
        lambda _path: MACE_OFF_MODEL_SHA256["off23_small"],
    )
    monkeypatch.setattr(
        "deepmd_gnn.mace_off_cli._rank_download_urls",
        lambda urls: urls,
    )

    model_path = download_mace_off_model("off23_small", cache_dir=tmp_path)

    assert model_path == tmp_path / "MACE-OFF23_small.model"
    assert attempts == [
        MACE_OFF_MODELS["off23_small"],
        MACE_OFF_MODEL_URLS["off23_small"][1],
    ]


def test_download_file_uses_unique_temp_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Concurrent downloads to one cache path should not share a temp filename."""
    destination = tmp_path / "MACE-OFF23_small.model"
    temp_paths: list[Path] = []

    class FakeResponse:
        def __enter__(self) -> object:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            return b""

    real_named_temporary_file = tempfile.NamedTemporaryFile

    def fake_named_temporary_file(*args: Any, **kwargs: Any) -> Any:
        tmp_file = real_named_temporary_file(*args, **kwargs)
        temp_paths.append(Path(tmp_file.name))
        return tmp_file

    monkeypatch.setattr(
        "deepmd_gnn.mace_off_cli.shutil.copyfileobj",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "deepmd_gnn.mace_off_cli.tempfile.NamedTemporaryFile",
        fake_named_temporary_file,
    )
    monkeypatch.setattr(
        "deepmd_gnn.mace_off_cli.urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )

    _download_file(MACE_OFF_MODELS["off23_small"], destination)
    _download_file(MACE_OFF_MODELS["off23_small"], destination)

    assert len(temp_paths) == 2
    assert temp_paths[0] != temp_paths[1]
    assert all(path.parent == tmp_path for path in temp_paths)
    assert all(not path.exists() for path in temp_paths)


def test_load_mace_off_model_rejects_non_positive_sel() -> None:
    """Sel is a required positive runtime neighbor cap."""
    with pytest.raises(ValueError, match="sel must be positive"):
        load_mace_off_model(model_name=None, model_path=Path("dummy.model"), sel=0)


def _make_off_policy_model(
    *,
    interaction_first: str = "RealAgnosticInteractionBlock",
    pair_repulsion: bool = False,
) -> torch.nn.Module:
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return make_mace_network(
            r_max=3.0,
            num_radial_basis=4,
            num_cutoff_basis=5,
            max_ell=1,
            interaction_first=interaction_first,
            interaction="RealAgnosticResidualInteractionBlock",
            num_interactions=2,
            num_elements=2,
            hidden_irreps="2x0e",
            atomic_numbers=[1, 8],
            avg_num_neighbors=4.0,
            pair_repulsion=pair_repulsion,
            distance_transform="None",
            correlation=2,
            gate="silu",
            MLP_irreps="4x0e",
            std=1.0,
            radial_MLP=[8, 8],
            radial_type="bessel",
            enable_cueq=False,
            script_model=False,
        )
    finally:
        torch.set_default_dtype(old_dtype)


def test_off_policy_rejects_named_lowercase_head() -> None:
    """Descriptor-compatible head names must not relax OFF conversion."""
    model = _make_off_policy_model()
    model.heads = ["default"]
    with pytest.raises(ValueError, match="Multi-head checkpoints"):
        _infer_deepmd_config(model)


def test_off_policy_rejects_pair_repulsion() -> None:
    """Descriptor-only pair-repulsion support must remain outside OFF."""
    model = _make_off_policy_model(pair_repulsion=True)
    with pytest.raises(ValueError, match="Pair-repulsion checkpoints"):
        _infer_deepmd_config(model)


def test_off_policy_rejects_residual_first_interaction() -> None:
    """MPA-style residual first blocks must remain outside OFF conversion."""
    model = _make_off_policy_model(
        interaction_first="RealAgnosticResidualInteractionBlock",
    )
    with pytest.raises(ValueError, match="first interaction block"):
        _infer_deepmd_config(model)


@pytest.mark.slow
def test_download_real_model_uses_existing_url(tmp_path: Path) -> None:
    """The official off23_small checkpoint should still be downloadable."""
    model_path = download_mace_off_model("off23_small", cache_dir=tmp_path)
    assert model_path.exists()
    assert model_path.stat().st_size > 0


@pytest.mark.slow
def test_infer_config_from_real_off23_small_checkpoint(tmp_path: Path) -> None:
    """A real checkpoint should yield the expected conservative config."""
    model_path = download_mace_off_model("off23_small", cache_dir=tmp_path)
    mace_model = _load_mace_checkpoint(model_path, device="cpu")
    config = _infer_deepmd_config(mace_model)

    assert config["type_map"] == ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    assert config["r_max"] == pytest.approx(4.5)
    assert config["num_radial_basis"] == 8
    assert config["num_cutoff_basis"] == 5
    assert config["max_ell"] == 3
    assert config["interaction"] == "RealAgnosticResidualInteractionBlock"
    assert config["num_interactions"] == 2
    assert config["hidden_irreps"] == "96x0e"
    assert config["pair_repulsion"] is False
    assert config["distance_transform"] == "None"
    assert config["correlation"] == 3
    assert config["gate"] == "silu"
    assert config["MLP_irreps"] == "16x0e"
    assert config["radial_type"] == "bessel"
    assert config["radial_MLP"] == [64, 64, 64]
    assert config["std"] == pytest.approx(1.0757, abs=1e-4)


@pytest.mark.slow
def test_load_mace_off_model_requires_explicit_sel_and_uses_plain_elements(
    tmp_path: Path,
) -> None:
    """Loaded wrappers should keep explicit sel and ordinary element names only."""
    model = load_mace_off_model(model_name="off23_small", cache_dir=tmp_path, sel=64)
    assert isinstance(model, MaceModel)
    assert model.get_sel() == [64]
    assert model.get_type_map() == ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    assert all(not atom_type.startswith("m") for atom_type in model.get_type_map())
    assert "HW" not in model.get_type_map()
    assert "OW" not in model.get_type_map()


@pytest.mark.slow
def test_load_mace_off_model_matches_native_mace_predictions(tmp_path: Path) -> None:
    """Wrapped DeePMD-GNN predictions should match the native MACE checkpoint."""
    model_path = download_mace_off_model("off23_small", cache_dir=tmp_path)
    native_model = _load_mace_checkpoint(model_path, device="cpu")
    native_model.eval()
    wrapped_model = load_mace_off_model(model_path=model_path, sel=64, device="cpu")

    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2390, 0.9266, 0.0]]],
        dtype=torch.float64,
    ).view(1, 9)
    atype = torch.tensor([[3, 0, 0]], dtype=torch.int64)
    box = (torch.eye(3, dtype=torch.float64) * 20.0).view(1, 9)

    native_outputs = _native_mace_reference_outputs(
        native_model,
        coord=coord,
        atype=atype,
        box=box,
        sel=64,
        ntypes=wrapped_model.get_ntypes(),
    )
    wrapped_outputs = wrapped_model(coord=coord, atype=atype, box=box)

    torch.testing.assert_close(
        wrapped_outputs["energy"],
        native_outputs["energy"],
        rtol=1e-5,
        atol=5e-6,
    )
    torch.testing.assert_close(
        wrapped_outputs["force"],
        native_outputs["force"],
        rtol=1e-5,
        atol=5e-6,
    )
    torch.testing.assert_close(
        wrapped_outputs["virial"],
        native_outputs["virial"],
        rtol=1e-5,
        atol=5e-6,
    )


@pytest.mark.slow
def test_convert_mace_off_to_deepmd_scripts_model(tmp_path: Path) -> None:
    """The conservative wrapper should still serialize to TorchScript."""
    output_file = tmp_path / "mace_off23_small_dp.pt"
    result = convert_mace_off_to_deepmd(
        output_file=str(output_file),
        model_name="off23_small",
        cache_dir=tmp_path,
        sel=64,
    )
    assert result == output_file
    assert output_file.exists()
    loaded = torch.jit.load(str(output_file))
    assert loaded is not None
