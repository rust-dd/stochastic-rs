"""Pytest configuration for the binding suite: skipped when the extension is not importable.

CI runs `maturin develop` first, so a skip there means the wheel was not built.
"""

from __future__ import annotations

import pytest

collect_ignore: list[str] = []

try:
    import stochastic_rs  # noqa: F401
except ImportError:  # pragma: no cover - exercised only in unbuilt envs
    pytest.skip(
        "stochastic_rs extension not built — run `maturin develop` first",
        allow_module_level=True,
    )
