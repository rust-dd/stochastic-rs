"""`STOCHASTIC_RS_DEVICE` / `STOCHASTIC_RS_DEVICE_BATCH_BYTES` are parsed strictly; no GPU is needed, the
name reads the environment before the build is checked, and each case runs in a fresh interpreter."""

import os
import re
import subprocess
import sys

import pytest

_CHILD = """
import sys
import stochastic_rs as sr
kind, name = sys.argv[1], sys.argv[2]
try:
    if kind == "probe":
        sr.probe_device(name)
    else:
        sr.PyGbm(0.05, 0.2, 16, x0=1.0, t=1.0, seed=1, device=name).sample()
except Exception as error:
    print(f"{type(error).__name__}: {error}")
else:
    print("ok")
"""


def _outcome(kind: str, name: str, **variables: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("STOCHASTIC_RS_DEVICE")}
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, kind, name],
        env={**env, **variables},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_a_bare_gpu_name_refuses_a_malformed_ordinal_variable():
    outcome = _outcome("probe", "cuda", STOCHASTIC_RS_DEVICE="gpu1")
    assert outcome.startswith("ValueError:"), outcome
    assert re.search(r"STOCHASTIC_RS_DEVICE\b", outcome), outcome


def test_an_explicit_ordinal_still_refuses_a_malformed_budget_variable():
    outcome = _outcome("probe", "cuda:0", STOCHASTIC_RS_DEVICE_BATCH_BYTES="0")
    assert outcome.startswith("ValueError:"), outcome
    assert "BATCH_BYTES" in outcome, outcome


@pytest.mark.parametrize("kind", ["probe", "build"])
def test_the_cpu_reads_neither_variable(kind):
    outcome = _outcome(kind, "cpu", STOCHASTIC_RS_DEVICE="gpu1", STOCHASTIC_RS_DEVICE_BATCH_BYTES="0")
    assert outcome == "ok", outcome
