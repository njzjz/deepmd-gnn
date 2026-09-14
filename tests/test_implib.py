"""Regression tests for the vendored implib vtable extraction helpers."""

from __future__ import annotations

import runpy
from pathlib import Path

IMPLIB = runpy.run_path(
    str(Path(__file__).parents[1] / "third_party" / "implib" / "implib-gen.py"),
)


def test_vtable_bytes_use_symbol_and_pointer_offsets(tmp_path: Path) -> None:
    """Read a non-section-start symbol and decode each pointer exactly once."""
    input_path = tmp_path / "fixture.so"
    input_path.write_bytes(bytes(range(64)))
    symbols = {
        "_ZTV7Example": {
            "Demangled Name": "vtable for Example",
            "Value": 0x1010,
            "Size": 16,
        },
    }
    sections = [{"Address": 0x1000, "Off": 8, "Size": 32}]

    symbol_bytes = IMPLIB["read_unrelocated_data"](
        input_path,
        symbols,
        sections,
    )
    assert symbol_bytes["_ZTV7Example"] == bytes(range(24, 40))

    relocated = IMPLIB["collect_relocated_data"](
        symbols,
        symbol_bytes,
        [],
        8,
        set(),
    )
    assert relocated["_ZTV7Example"] == [
        ("offset", int.from_bytes(bytes(range(24, 32)), byteorder="little")),
        ("offset", int.from_bytes(bytes(range(32, 40)), byteorder="little")),
    ]
