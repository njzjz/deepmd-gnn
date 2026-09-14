# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native SevenNet checkpoint support for property descriptors."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from e3nn import o3

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sevenn.nn.sequential import AtomGraphSequential

ENERGY_MODULE_NAMES = (
    "reduce_input_to_hidden",
    "reduce_hidden_to_energy",
    "readout_FCN",
    "rescale_atomic_energy",
    "reduce_total_enegy",
    "force_output",
)

_INT_KEY_DICTS = ("_type_map",)


@contextmanager
def temporary_default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Temporarily set Torch's process-wide construction dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def json_safe_value(value: Any) -> Any:  # noqa: ANN401, PLR0911
    """Convert nested SevenNet config values into JSON-serializable objects."""
    if isinstance(value, dict):
        converted = {}
        for key, inner in value.items():
            json_key = str(int(key)) if isinstance(key, (int, np.integer)) else key
            converted[json_key] = json_safe_value(inner)
        return converted
    if isinstance(value, (list, tuple)):
        return [json_safe_value(inner) for inner in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def restore_sevenn_config(config: dict[str, Any]) -> dict[str, Any]:
    """Restore JSON-safe SevenNet config into constructor-ready Python objects."""
    restored = deepcopy(config)
    for name in _INT_KEY_DICTS:
        mapping = restored.get(name)
        if isinstance(mapping, dict):
            restored[name] = {int(key): int(value) for key, value in mapping.items()}
    return restored


def persistable_checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return JSON-safe metadata needed to rebuild a SevenNet feature backbone."""
    if "sevenn_config" in config:
        sevenn_config = json_safe_value(config["sevenn_config"])
        chemical_species = list(
            config.get("type_map") or sevenn_config.get("chemical_species") or [],
        )
        source_dtype = str(
            config.get("source_dtype") or sevenn_config.get("dtype") or "float32",
        )
    else:
        sevenn_config = json_safe_value(dict(config))
        chemical_species = list(sevenn_config.get("chemical_species") or [])
        source_dtype = str(sevenn_config.get("dtype") or "float32")
    if source_dtype not in {"float32", "float64"}:
        source_dtype = "float32"
    return {
        "type_map": chemical_species,
        "cutoff": float(sevenn_config["cutoff"]),
        "num_convolution_layer": int(sevenn_config["num_convolution_layer"]),
        "source_dtype": source_dtype,
        "sevenn_config": sevenn_config,
    }


def reject_unsupported_sevennet(config: dict[str, Any]) -> None:
    """Reject multi-fidelity, accelerator, and LAMMPS-only SevenNet variants."""
    if config.get("use_modality"):
        msg = (
            "Multi-fidelity / modal SevenNet checkpoints are unsupported; "
            "use a single-task model such as 7net-0, 7net-l3i5, or 7net-omat"
        )
        raise ValueError(msg)
    if config.get("_modal_map"):
        msg = "SevenNet checkpoints with a modal map are unsupported"
        raise ValueError(msg)
    cueq = config.get("cuequivariance_config") or {}
    if isinstance(cueq, dict) and cueq.get("use"):
        msg = "cuEquivariance SevenNet descriptors are unsupported"
        raise ValueError(msg)
    if config.get("use_flash_tp"):
        msg = "FlashTP SevenNet descriptors are unsupported"
        raise ValueError(msg)
    if config.get("use_oeq"):
        msg = "OpenEquivariance SevenNet descriptors are unsupported"
        raise ValueError(msg)
    if config.get("use_mliap"):
        msg = "MLIAP / LAMMPS SevenNet descriptors are unsupported"
        raise ValueError(msg)


def _infer_state_dtype(state_dict: dict[str, Any]) -> torch.dtype:
    for value in state_dict.values():
        if torch.is_tensor(value) and value.is_floating_point():
            return value.dtype
    return torch.float32


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    return torch.float32


def prepare_feature_backbone(model: AtomGraphSequential) -> AtomGraphSequential:
    """Drop energy/force heads so node features remain last-layer invariants."""
    for name in ENERGY_MODULE_NAMES:
        model.delete_module_by_key(name)
    model.eval_type_map = False
    model.eval_modal_map = False
    model.key_grad = None
    return model


def script_feature_backbone(model: AtomGraphSequential) -> torch.nn.Module:
    """Script the stripped backbone with e3nn so DeePMD can script the wrapper."""
    from e3nn.util.jit import script  # noqa: PLC0415

    model.eval()
    device = next(model.parameters()).device
    return script(model).to(device)


def last_feature_irreps(model: AtomGraphSequential) -> o3.Irreps:
    """Return irreps of the last interaction-block node features."""
    gates = [
        module
        for name, module in model.named_children()
        if name.endswith("equivariant_gate")
    ]
    if not gates:
        msg = "SevenNet backbone has no equivariant_gate layers"
        raise ValueError(msg)
    irreps_out = getattr(getattr(gates[-1], "gate", None), "irreps_out", None)
    if irreps_out is None:
        msg = "Final SevenNet gate does not expose output irreps"
        raise ValueError(msg)
    return o3.Irreps(irreps_out)


def scalar_even_indices(irreps: o3.Irreps) -> list[int]:
    """Return flattened indices belonging to invariant, even scalar irreps."""
    indices: list[int] = []
    offset = 0
    for multiplicity, irrep in irreps:
        width = multiplicity * irrep.dim
        if irrep.l == 0 and irrep.p == 1:
            indices.extend(range(offset, offset + width))
        offset += width
    return indices


def resolve_sevennet_checkpoint_path(model_path: str | Path) -> Path | None:
    """Resolve a local file or SevenNet pretrained keyword without rebuilding."""
    from sevenn.util import pretrained_name_to_path  # noqa: PLC0415

    path = Path(model_path)
    if path.is_file():
        return path
    try:
        return Path(pretrained_name_to_path(str(model_path)))
    except (ValueError, FileNotFoundError):
        return None


def _load_raw_sevennet_payload(model_path: str | Path) -> dict[str, Any]:
    """Read a trusted SevenNet pickle before SevenNet applies config patches."""
    resolved = resolve_sevennet_checkpoint_path(model_path)
    if resolved is None:
        msg = f"SevenNet checkpoint not found: {model_path}"
        raise FileNotFoundError(msg)
    payload = torch.load(str(resolved), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        msg = (
            "SevenNet checkpoint must be a dictionary containing config and "
            f"model_state_dict, got {type(payload)!r}"
        )
        raise TypeError(msg)
    return payload


def load_sevennet_checkpoint_config(model_path: str | Path) -> dict[str, Any]:
    """Load and persist architecture metadata from a trusted SevenNet checkpoint."""
    from sevenn.util import load_checkpoint  # noqa: PLC0415

    payload = _load_raw_sevennet_payload(model_path)
    reject_unsupported_sevennet(payload.get("config") or {})
    checkpoint = load_checkpoint(str(model_path))
    reject_unsupported_sevennet(checkpoint.config)
    return persistable_checkpoint_config(checkpoint.config)


def build_sevennet_feature_backbone(
    config: dict[str, Any],
    *,
    device: str | torch.device,
) -> AtomGraphSequential:
    """Rebuild a stripped SevenNet backbone from persisted constructor metadata."""
    from sevenn.model_build import build_E3_equivariant_model  # noqa: PLC0415

    persistable = persistable_checkpoint_config(config)
    sevenn_config = restore_sevenn_config(persistable["sevenn_config"])
    reject_unsupported_sevennet(sevenn_config)
    dtype = _dtype_from_name(persistable["source_dtype"])
    with temporary_default_dtype(dtype):
        model = build_E3_equivariant_model(sevenn_config)
    return prepare_feature_backbone(model).to(device)


def load_native_sevennet_feature_backbone(
    model_path: str | Path,
    *,
    device: str | torch.device,
) -> tuple[AtomGraphSequential, dict[str, Any]]:
    """Load a trusted SevenNet checkpoint and drop energy/force modules."""
    from sevenn.util import load_checkpoint  # noqa: PLC0415

    payload = _load_raw_sevennet_payload(model_path)
    reject_unsupported_sevennet(payload.get("config") or {})
    checkpoint = load_checkpoint(str(model_path))
    reject_unsupported_sevennet(checkpoint.config)
    persistable = persistable_checkpoint_config(checkpoint.config)
    dtype = _infer_state_dtype(checkpoint.model_state_dict)
    persistable["source_dtype"] = "float64" if dtype == torch.float64 else "float32"
    with temporary_default_dtype(dtype):
        model = checkpoint.build_model()
    model = prepare_feature_backbone(model).to(device)
    return model, persistable


def validate_sevennet_state_dict_load(load_result: object) -> None:
    """Reject missing or unexpected backbone weights."""
    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    if missing_keys or unexpected_keys:
        msg = (
            "Failed to load SevenNet checkpoint into DeePMD-GNN wrapper. "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
        raise RuntimeError(msg)
