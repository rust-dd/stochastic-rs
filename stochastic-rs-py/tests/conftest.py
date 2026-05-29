"""Pytest configuration for the `stochastic_rs` binding test suite.

If the compiled extension module is not importable (e.g. `maturin
develop` has not been run), the whole suite is skipped with a clear
message rather than erroring — the CI gate runs `maturin develop` first,
so a skip here means the wheel was not built.
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
