"""Scoped compatibility for trusted e3nn package constants."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def trusted_e3nn_constants() -> Iterator[None]:
    """Allow the built-in slice type used by trusted e3nn constant files.

    Older e3nn releases serialize slice objects in their packaged Wigner
    constants. PyTorch's weights-only loader rejects those objects unless they
    are explicitly allowlisted. Keep that permission scoped to imports that may
    load e3nn's installed package data.
    """
    old_no_weights_only = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    # safe_globals removes its additions on exit even if the caller had already
    # allowed them. Add only missing entries to preserve nested/caller scopes.
    additions = [] if slice in torch.serialization.get_safe_globals() else [slice]
    try:
        with torch.serialization.safe_globals(additions):
            yield
    finally:
        # Older MACE releases set this variable during import. Restore the
        # caller's setting so the compatibility workaround stays local.
        if old_no_weights_only is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = old_no_weights_only
