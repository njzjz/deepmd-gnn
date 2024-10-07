"""Test custom operations."""

import torch

import deepmd_gnn.op  # noqa: F401


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

    assert torch.equal(edge_index, expected_edge_index)


def test_two_frame() -> None:
    """Test one frame."""
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
        torch.tensor(mm_types, dtype=torch.int64, device="cpu"),
    )

    assert torch.equal(edge_index, expected_edge_index)
