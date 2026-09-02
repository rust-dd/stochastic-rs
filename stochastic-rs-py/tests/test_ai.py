"""Surrogate (AI) binding coverage — only in `maturin develop --features ai`
builds; the published wheels leave candle out, so the module is skipped when
the classes are absent."""

from __future__ import annotations

import numpy as np
import pytest
import stochastic_rs as sr

if not hasattr(sr, "HestonNn"):
    pytest.skip("built without the ai feature", allow_module_level=True)


def _synthetic(params: np.ndarray, lb: np.ndarray, ub: np.ndarray, out: int) -> np.ndarray:
    x = (params - 0.5 * (lb + ub)) / (0.5 * (ub - lb))
    k = np.arange(out)
    surf = 0.2 + 0.03 * k / out
    surf = np.broadcast_to(surf, (params.shape[0], out)).copy()
    for j in range(params.shape[1]):
        surf += (0.08 + 0.02 * (j + 1)) * x[:, [j]] * np.sin((k + 1) * (j + 1) * 0.11)
    return surf


def test_heston_nn_trains_predicts_and_calibrates(tmp_path):
    model = sr.HestonNn()
    lb, ub = np.array(model.param_lb), np.array(model.param_ub)
    assert model.input_dim == 5 and model.output_dim == 88
    rng = np.random.default_rng(3)
    params = lb + rng.uniform(size=(256, 5)) * (ub - lb)
    surfaces = _synthetic(params, lb, ub, 88)
    report = model.train(params, surfaces, epochs=20, seed=1)
    assert len(report["val_rmse"]) == 20 and np.isfinite(report["val_rmse"][-1])
    theta = list(lb + 0.4 * (ub - lb))
    surface = model.predict_surface(theta)
    assert surface.shape == (88,)
    surface2, jacobian = model.predict_surface_with_jacobian(theta)
    assert np.allclose(surface, surface2, atol=1e-6) and jacobian.shape == (88, 5)
    fit = sr.calibrate_surrogate(model, surface)
    assert fit["converged"] and fit["in_bounds"] and fit["rmse"] < 1e-5
    assert np.allclose(fit["params"], theta, atol=1e-2 * (ub - lb))
    model.save(str(tmp_path / "heston_nn"))
    loaded = sr.HestonNn.load(str(tmp_path / "heston_nn"))
    assert np.allclose(loaded.predict_surface(theta), surface, atol=1e-5)
    with pytest.raises(ValueError):
        sr.calibrate_surrogate(model, surface[:10])
    with pytest.raises(ValueError):
        sr.calibrate_surrogate("not a model", surface)
