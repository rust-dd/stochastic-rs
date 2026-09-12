"""Regressions for stateful sampling, surface calibration and exposure failures."""

import math

import numpy as np
import pytest
import stochastic_rs as sr


@pytest.mark.parametrize("device", ["cpu", "accelerate", "metal", "cuda"])
@pytest.mark.parametrize("dtype", ["f32", "f64"])
@pytest.mark.parametrize("model", ["gbm", "fbm", "sabr", "cfou", "heston", "ho_lee", "cir_pp", "black_karasinski", "hull_white", "hull_white_2f", "bergomi", "bates_svj"])
def test_device_sampling_advances_and_replays(device, dtype, model):
    try:
        info = sr.probe_device(device)
    except (ValueError, RuntimeError) as error:
        pytest.skip(str(error))
    f64_only = {"ho_lee", "cir_pp", "black_karasinski", "hull_white", "hull_white_2f"}
    if dtype not in info["precisions"] or (model in f64_only and dtype != "f64"):
        pytest.skip("unsupported precision")

    def create():
        args = dict(seed=42, device=device, dtype=dtype)
        if model == "gbm":
            return sr.PyGbm(0.05, 0.2, 64, x0=100.0, t=1.0, **args)
        if model == "fbm":
            return sr.PyFbm(0.7, 64, t=1.0, **args)
        if model == "sabr":
            return sr.PySabr(0.3, 0.5, -0.4, 64, f0=100.0, v0=0.2, t=1.0, **args)
        if model == "cfou":
            return sr.PyCfou(0.7, 1.2, 3.0, 0.4, 64, t=1.0, **args)
        if model == "heston":
            return sr.PyHeston(2.0, 0.04, 0.3, -0.7, 0.05, 64, s0=100.0, v0=0.04, t=1.0, **args)
        if model == "bergomi":
            return sr.PyBergomi(0.3, 0.02, -0.7, 64, v0=0.04, s0=100.0, t=1.0, **args)
        if model == "bates_svj":
            return sr.PyBatesSvj(0.2, 0.0, 0.1, 2.0, 0.04, 0.3, -0.7, 64,
                                 r=0.02, r_f=0.0, s0=100.0, v0=0.04, t=1.0, **args)
        args.pop("dtype")
        if model == "cir_pp":
            return sr.PyCirPlusPlus(1.2, 0.04, 0.3, lambda t: 0.02, 64, x0=0.03, t=1.0, **args)
        if model == "black_karasinski":
            return sr.PyBlackKarasinski(lambda t: 0.03, 0.5, 0.1, 64, r0=0.03, t=1.0, **args)
        if model == "hull_white":
            return sr.PyHullWhite(lambda t: 0.03, 0.5, 0.01, 64, x0=0.03, t=1.0, **args)
        if model == "hull_white_2f":
            return sr.PyHullWhite2F(lambda t: 0.03, 0.5, 0.01, 0.02, -0.3, 0.4, 64,
                                   x0=0.03, t=1.0, **args)
        return sr.PyHoLee(0.01, 64, f_T=lambda t: 0.03, t=1.0, **args)

    first, replay = create(), create()
    previous = None
    for method, args in [("sample", ()), ("sample", ()), ("sample_par", (4,)), ("sample_par", (4,))]:
        got = np.asarray(getattr(first, method)(*args))
        same = np.asarray(getattr(replay, method)(*args))
        assert np.array_equal(got, same)
        if previous is not None and previous.shape == got.shape:
            assert not np.array_equal(got, previous)
        previous = got


def test_essvi_accepts_equal_consecutive_variances():
    ks = [-0.1, 0.0, 0.1]
    ws = [0.02 * (1.0 + math.sqrt((2.5 * k) ** 2 + 1.0)) for k in ks]
    surface = sr.EssviSurface.calibrate([1.0, 2.0], [(ks, ws, 0.04)] * 2)
    assert surface.is_calendar_spread_free()
    assert surface.is_butterfly_free()
    for k, w in zip(ks, ws):
        assert surface.total_variance(k, 2.0) == pytest.approx(w, abs=1e-8)


def test_essvi_infeasible_quotes_raise_value_error():
    first = ([-0.1, 0.0, 0.1], [0.05, 0.04, 0.05], 0.04)
    second = ([-0.1, 0.0, 0.1], [0.03, 0.02, 0.03], 0.02)
    with pytest.raises(ValueError, match="no admissible"):
        sr.EssviSurface.calibrate([1.0, 2.0], [first, second])


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_xva_rejects_nonfinite_mtm(invalid):
    with pytest.raises(ValueError, match="MtM values must be finite"):
        sr.ExposureProfile.from_mtm(np.array([[100.0], [invalid]]), [1.0])


def test_adi_prices_spot_beyond_eight_strikes():
    model = sr.HestonAdiPricer(100.0, 0.04, 10.0, 0.0, 2.0, 0.04, 0.3, -0.7, 1.0,
                               q=0.0, m1=40, m2=20, steps=20)
    call, put = model.call_put()
    assert call == pytest.approx(90.0, abs=1e-3)
    assert -1e-6 <= put < 1e-3
