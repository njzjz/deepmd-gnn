"""Test custom operations."""

import pytest
import torch

import deepmd_gnn.op  # noqa: F401
from deepmd_gnn.edge import _fake_dense_edge_index, dense_edge_index


def test_one_frame() -> None:
    """Test one frame."""
    nlist_ff = torch.tensor(
        [
            [1, 2, -1, -1],
            [2, 0, -1, -1],
            [0, 1, -1, -1],
        ],
        dtype=torch.int64,
        device="cpu",
    )
    extended_atype_ff = torch.tensor(
        [0, 1, 2],
        dtype=torch.int64,
        device="cpu",
    )
    mm_types = [1, 2]
    expected_edge_index = torch.tensor(
        [
            [1, 0],
            [2, 0],
            [0, 1],
            [0, 2],
        ],
        dtype=torch.int64,
        device="cpu",
    )

    edge_index = torch.ops.deepmd_gnn.edge_index(
        nlist_ff,
        extended_atype_ff,
        torch.tensor(mm_types, dtype=torch.int64, device="cpu"),
    )
    dense_edge_index_, edge_mask = dense_edge_index(
        nlist_ff,
        extended_atype_ff,
        mm_types,
    )

    assert torch.equal(edge_index, expected_edge_index)
    assert torch.equal(dense_edge_index_[edge_mask], expected_edge_index)


def test_two_frame() -> None:
    """Test two frames."""
    nlist = torch.tensor(
        [
            [
                [1, 2, -1, -1],
                [2, 0, -1, -1],
                [0, 1, -1, -1],
            ],
            [
                [1, 2, -1, -1],
                [2, 0, -1, -1],
                [0, 1, -1, -1],
            ],
        ],
        dtype=torch.int64,
        device="cpu",
    )
    extended_atype = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 2],
        ],
        dtype=torch.int64,
        device="cpu",
    )
    mm_types = [1, 2]
    mm_tensor = torch.tensor(mm_types, dtype=torch.int64, device="cpu")
    expected_edge_index = torch.tensor(
        [
            [1, 0],
            [2, 0],
            [0, 1],
            [0, 2],
            [4, 3],
            [5, 3],
            [3, 4],
            [3, 5],
        ],
        dtype=torch.int64,
        device="cpu",
    )

    edge_index = torch.ops.deepmd_gnn.edge_index(
        nlist,
        extended_atype,
        mm_tensor,
    )
    dense_edge_index_, edge_mask = dense_edge_index(
        nlist,
        extended_atype,
        mm_types,
    )
    legacy_edge_index = torch.ops.deepmd_mace.mace_edge_index(
        nlist,
        extended_atype,
        mm_tensor,
    )

    assert torch.equal(edge_index, expected_edge_index)
    assert torch.equal(dense_edge_index_[edge_mask], expected_edge_index)
    assert torch.equal(legacy_edge_index, expected_edge_index)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_cpu() -> None:
    """Test CUDA edge-index implementation against the CPU implementation."""
    nlist = torch.tensor(
        [
            [
                [1, 2, -1, -1],
                [2, 0, -1, -1],
                [0, 1, -1, -1],
            ],
            [
                [1, 2, -1, -1],
                [2, 0, -1, -1],
                [0, 1, -1, -1],
            ],
        ],
        dtype=torch.int64,
        device="cpu",
    )
    extended_atype = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 2],
        ],
        dtype=torch.int64,
        device="cpu",
    )
    mm_types = torch.tensor([1, 2], dtype=torch.int64, device="cpu")

    expected_edge_index = torch.ops.deepmd_gnn.edge_index(
        nlist,
        extended_atype,
        mm_types,
    )
    edge_index = torch.ops.deepmd_gnn.edge_index(
        nlist.cuda(),
        extended_atype.cuda(),
        mm_types,
    )
    legacy_edge_index = torch.ops.deepmd_mace.mace_edge_index(
        nlist.cuda(),
        extended_atype.cuda(),
        mm_types,
    )

    assert edge_index.is_cuda
    torch.testing.assert_close(edge_index.cpu(), expected_edge_index)
    assert legacy_edge_index.is_cuda
    torch.testing.assert_close(legacy_edge_index.cpu(), expected_edge_index)


def test_fake_dense_edge_index_handles_2d_nlist() -> None:
    """Fake dense edge op should preserve the flattened 2D nlist size."""
    nlist = torch.empty((3, 4), dtype=torch.int64)
    atype = torch.empty((3,), dtype=torch.int64)
    mm_types = torch.empty((2,), dtype=torch.int64)

    edge_index, edge_mask = _fake_dense_edge_index(nlist, atype, mm_types)

    assert edge_index.shape == (12, 2)
    assert edge_mask.shape == (12,)


def test_fake_dense_edge_index_rejects_invalid_rank() -> None:
    """Fake dense edge op should match the real op rank contract."""
    nlist = torch.empty((1, 2, 3, 4), dtype=torch.int64)
    atype = torch.empty((1, 2), dtype=torch.int64)
    mm_types = torch.empty((2,), dtype=torch.int64)

    with pytest.raises(ValueError, match="nlist must be 2D or 3D"):
        _fake_dense_edge_index(nlist, atype, mm_types)


@pytest.mark.parametrize(
    ("nlist", "atype", "mm_types", "message"),
    [
        (
            torch.zeros((3, 1), dtype=torch.int64),
            torch.zeros((2,), dtype=torch.int64),
            torch.empty((0,), dtype=torch.int64),
            "nloc must not exceed nall",
        ),
        (
            torch.zeros((2, 1), dtype=torch.int64),
            torch.zeros((2,), dtype=torch.int64),
            torch.empty((1, 1), dtype=torch.int64),
            "mm_tensor must be 1D",
        ),
    ],
)
def test_edge_index_rejects_invalid_shapes(
    nlist: torch.Tensor,
    atype: torch.Tensor,
    mm_types: torch.Tensor,
    message: str,
) -> None:
    """Native edge-index entry points should reject unsafe tensor shapes."""
    with pytest.raises((RuntimeError, ValueError), match=message):
        torch.ops.deepmd_gnn.edge_index(nlist, atype, mm_types)
    with pytest.raises((RuntimeError, ValueError), match=message):
        torch.ops.deepmd_gnn.dense_edge_index(nlist, atype, mm_types)


def test_out_of_range_neighbors_are_masked() -> None:
    """Non-negative neighbor indices beyond nall must never be dereferenced."""
    nlist = torch.tensor([[1, 2, -1], [0, -1, -1]], dtype=torch.int64)
    atype = torch.tensor([0, 1], dtype=torch.int64)
    mm_types = torch.empty((0,), dtype=torch.int64)
    expected = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)

    sparse = torch.ops.deepmd_gnn.edge_index(nlist, atype, mm_types)
    dense, mask = torch.ops.deepmd_gnn.dense_edge_index(nlist, atype, mm_types)

    assert torch.equal(sparse, expected)
    assert torch.equal(dense[mask], expected)
    assert mask.tolist() == [True, False, False, True, False, False]
