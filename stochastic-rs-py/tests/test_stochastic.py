"""GBM process binding: seed determinism, path shape, the parallel multi-path sampler and sanity of
the simulated levels."""

from __future__ import annotations

import math

import numpy as np
import pytest
import stochastic_rs as sr


def test_gbm_seed_determinism():
    a = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=42).sample()
    b = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=42).sample()
    assert np.allclose(a, b)


def test_gbm_path_length():
    p = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=1).sample()
    assert p.shape[0] == 252


def test_gbm_starts_at_x0():
    p = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=1).sample()
    assert abs(float(p[0]) - 100.0) < 1e-6


def test_gbm_strictly_positive():
    p = sr.PyGbm(0.03, 0.4, 500, x0=50.0, t=2.0, seed=8).sample()
    assert np.all(p > 0.0)


def test_gbm_distinct_seeds_differ():
    a = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=1).sample()
    b = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=2).sample()
    assert not np.allclose(a, b)


def test_gbm_sample_par_determinism():
    a = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=42).sample_par(8)
    b = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=42).sample_par(8)
    assert np.allclose(a, b)
    assert a.shape == (8, 252)


def test_gbm_sample_par_shape():
    s = sr.PyGbm(0.05, 0.2, 128, x0=100.0, t=1.0, seed=3).sample_par(16)
    assert s.shape == (16, 128)


def test_gbm_paths_all_finite():
    s = sr.PyGbm(0.05, 0.2, 252, x0=100.0, t=1.0, seed=4).sample_par(32)
    assert np.all(np.isfinite(s))


def test_sobol_and_bridge_qmc():
    plain = sr.PySobolSeq(3000)
    pts = plain.sample(64)
    assert pts.shape == (64, 3000) and (pts >= 0).all() and (pts < 1).all()
    assert not plain.is_scrambled
    scrambled = sr.PySobolSeq(8, seed=5)
    a, b = scrambled.sample(1024), sr.PySobolSeq(8, seed=5).sample(1024)
    assert scrambled.is_scrambled and np.array_equal(a, b)
    qmc = sr.PyBrownianBridgeQmc(32, 2.0, seed=3)
    w = qmc.paths(2048)
    assert w.shape == (2048, 32) and abs(np.var(w[:, -1]) - 2.0) < 0.15
    dw = qmc.increments(2048)
    assert np.allclose(np.cumsum(dw, axis=1), w)


def test_multi_gbm_and_correlated_driver():
    rho = np.array([[1.0, 0.6, -0.2], [0.6, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    dw = sr.PyMcgns(rho, 100_000, t=1.0, seed=3).sample()
    assert dw.shape == (3, 100_000)
    assert np.allclose(np.corrcoef(dw), rho, atol=0.02)
    model = sr.PyMultiGbm([0.05, 0.03, 0.01], [0.2, 0.3, 0.15], rho, 64, [100.0, 50.0, 10.0], t=1.0, seed=7)
    one = model.sample()
    assert one.shape == (3, 64) and np.allclose(one[:, 0], [100.0, 50.0, 10.0])
    paths = model.sample_par(2000)
    assert len(paths) == 2000 and paths[0].shape == (3, 64)
    assert not np.array_equal(paths[0], paths[1])


def test_wishart_process():
    b = np.array([[-0.5, 0.1], [0.05, -0.3]])
    a = np.array([[0.3, 0.1], [0.0, 0.2]])
    x0 = np.array([[1.0, 0.2], [0.2, 0.5]])
    process = sr.PyWishart(2.5, b, a, x0, 5, t=1.0, seed=7)
    path = process.sample()
    assert path.shape == (5, 2, 2)
    assert np.allclose(path[0], x0) and np.allclose(path, np.swapaxes(path, 1, 2))
    paths = process.sample_par(4000)
    assert len(paths) == 4000 and paths[0].shape == (5, 2, 2)
    terminal = np.mean([p[-1] for p in paths], axis=0)
    assert np.abs(terminal - process.mean(1.0)).max() < 0.05
    v = -0.4 * np.array([[1.0, 0.3], [0.3, 1.0]])
    lt = process.laplace_transform(v, 1.0)
    assert 0.0 < lt < 1.0
    mc = np.mean([np.exp(np.trace(v @ p[-1])) for p in paths])
    assert abs(mc - lt) < 0.03


def test_cheyette_state_and_bonds():
    model = sr.PyCheyette(lambda t: 0.03, 0.5, lambda t, x: 0.01, 101, t=1.0, seed=3)
    x, y = model.sample()
    assert x.shape == (101,) and y.shape == (101,)
    assert x[0] == 0.0 and y[0] == 0.0
    assert np.all(np.diff(y) >= 0.0)
    closed = 0.01**2 * (1.0 - np.exp(-2.0 * 0.5 * 1.0)) / (2.0 * 0.5)
    assert abs(y[-1] - closed) / closed < 0.01
    xs, ys = model.sample_par(50)
    assert xs.shape == (50, 101) and ys.shape == (50, 101)
    assert abs(model.zero_bond(0.0, 2.0, 0.0, 0.0) - np.exp(-0.06)) < 1e-12
    assert abs(model.short_rate(0.5, 0.004) - 0.034) < 1e-15


def test_hjm_returns_the_short_rate_bond_price_and_forward_rate():
    flat = lambda t, m: 0.0
    vol = lambda t, m: 0.01
    model = sr.PyHjm(
        lambda t: 0.0, lambda t: 0.01, lambda t, m: 1.0, flat, vol, flat, vol, 16,
        r0=0.03, p0=1.0, f0=0.04, seed=3,
    )
    r, p, f = model.sample()
    assert r.shape == p.shape == f.shape == (16,)
    assert (r[0], p[0], f[0]) == (0.03, 1.0, 0.04)
    rs, ps, fs = model.sample_par(8)
    assert rs.shape == ps.shape == fs.shape == (8, 16)
    assert (rs[:, 0] == 0.03).all() and (fs[:, 0] == 0.04).all()


def test_device_classes_reproduce_the_euler_moments():
    paths = sr.PyGbm(0.05, 0.2, 253, x0=100.0, t=1.0, seed=7).sample_par(20_000)
    assert paths.shape == (20_000, 253) and np.allclose(paths[:, 0], 100.0)
    assert abs(paths[:, -1].mean() / (100.0 * math.exp(0.05)) - 1.0) < 0.02
    ou = sr.PyOu(2.0, 1.0, 0.5, 501, x0=1.0, t=2.0, seed=3).sample_par(10_000)
    assert abs(ou[:, -1].mean() - 1.0) < 0.05
    cir = sr.PyCir(1.5, 0.04, 0.3, 253, x0=0.09, t=1.0, seed=5).sample_par(5_000)
    assert cir.min() >= 0.0
    assert np.array_equal(paths, sr.PyGbm(0.05, 0.2, 253, x0=100.0, t=1.0, seed=7).sample_par(20_000))


def test_device_names_are_honoured_or_reported():
    # A device name is either served by a compiled back-end or refused with a
    # rebuild hint; never silently redirected to the CPU.
    with pytest.raises(ValueError):
        sr.PyGbm(0.05, 0.2, 10, x0=1.0, seed=7, device="tpu")
    for device in ("cuda", "metal"):
        try:
            gbm = sr.PyGbm(0.05, 0.2, 64, x0=100.0, t=1.0, seed=7, dtype="f32", device=device)
        except ValueError as err:
            assert "rebuild" in str(err)
            continue
        except RuntimeError:
            continue
        paths = gbm.sample_par(256)
        assert paths.shape == (256, 64) and paths.dtype == np.float32
        assert np.allclose(paths[:, 0], 100.0)
        assert abs(paths[:, -1].mean() / (100.0 * np.exp(0.05)) - 1.0) < 0.05


def test_probe_device_describes_the_host_and_reports_missing_devices():
    info = sr.probe_device("cpu")
    assert info["backend"] == "Cpu" and info["precisions"] == ["f32", "f64"] and info["ordinal"] is None
    assert "thread" in info["name"]
    with pytest.raises(ValueError):
        sr.probe_device("tpu")
    for device in ("cuda", "metal", "accelerate"):
        try:
            found = sr.probe_device(device)
        except ValueError as err:
            assert "rebuild" in str(err)
            continue
        except RuntimeError:
            continue
        assert found["backend"] in ("Cuda", "Metal", "Accelerate")
        assert all(p in ("f32", "f64") for p in found["precisions"])
    with pytest.raises(ValueError):
        sr.probe_device("cpu:x")


def test_device_argument_on_the_process_classes():
    base = sr.PyGbm(0.05, 0.2, 64, x0=100.0, t=1.0, seed=7)
    same = sr.PyGbm(0.05, 0.2, 64, x0=100.0, t=1.0, seed=7, device="cpu")
    assert np.array_equal(base.sample_par(4), same.sample_par(4))
    assert np.array_equal(sr.PyFgn(0.7, 64, seed=3).sample_par(2), sr.PyFgn(0.7, 64, seed=3, device="cpu").sample_par(2))
    with pytest.raises(ValueError):
        sr.PyGbm(0.05, 0.2, 64, x0=1.0, seed=7, device="tpu")
    # float64 on a single-precision device is refused whatever the build carries
    with pytest.raises(ValueError):
        sr.PyGbm(0.05, 0.2, 64, x0=1.0, seed=7, device="metal")
    with pytest.raises(ValueError):
        sr.PyFbm(0.7, 64, seed=7, device="metal")
    for device in ("cuda", "metal", "accelerate"):
        try:
            gbm = sr.PyGbm(0.05, 0.2, 64, x0=100.0, t=1.0, seed=7, dtype="f32", device=device)
            fgn = sr.PyFgn(0.7, 64, seed=7, dtype="f32", device=device)
        except ValueError as err:
            assert "rebuild" in str(err)
            continue
        except RuntimeError:
            continue
        paths = gbm.sample_par(8)
        assert paths.shape == (8, 64) and paths.dtype == np.float32
        assert np.allclose(paths[:, 0], 100.0)
        assert gbm.sample().shape == (64,)
        noise = fgn.sample_par(3)
        assert noise.shape == (3, 64) and noise.dtype == np.float32



def test_heston_slv_samples_under_a_callable_and_a_grid_leverage():
    kwargs = dict(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, mu=0.05, eta=0.6, n=65, s0=100.0, v0=0.04, t=1.0)
    by_callable = sr.PyHestonSlv(leverage=lambda t, s: 0.8 + 0.002 * s, seed=3, **kwargs)
    s, v = by_callable.sample()
    assert s.shape == (65,) and v.shape == (65,)
    assert s[0] == 100.0 and v[0] == 0.04 and np.all(v >= 0.0)
    assert not by_callable.device_ready()
    assert abs(by_callable.leverage(0.2, 100.0) - 1.0) < 1e-15
    spots = np.linspace(1.0, 400.0, 400)
    times = np.array([0.0, 1.0])
    values = 0.8 + 0.002 * spots[None, :].repeat(2, axis=0)
    by_grid = sr.PyHestonSlv(leverage=(spots, times, values), seed=3, **kwargs)
    s_grid, v_grid = by_grid.sample()
    assert np.array_equal(v_grid, v)
    assert np.allclose(s_grid, s, rtol=1e-9)
    ss, vs = by_grid.sample_par(8)
    assert ss.shape == (8, 65) and vs.shape == (8, 65)
    again, _ = sr.PyHestonSlv(leverage=(spots, times, values), seed=3, **kwargs).sample()
    assert np.array_equal(again, s_grid)
    with pytest.raises(ValueError):
        sr.PyHestonSlv(leverage=(spots, times, values[:1]), **kwargs)
    with pytest.raises(ValueError):
        sr.PyHestonSlv(leverage=lambda t, s: 1.0, **{**kwargs, "eta": 1.5})


def test_heston_slv_rejects_bad_inputs_with_python_errors():
    kwargs = dict(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, mu=0.05, eta=0.6, n=65, s0=100.0, v0=0.04)
    unit = lambda t, s: 1.0
    for bad in (
        {"kappa": math.nan},
        {"sigma": -0.1},
        {"mu": math.inf},
        {"s0": math.nan},
        {"v0": -0.01},
        {"t": 0.0},
        {"t": -1.0},
        {"t": math.nan},
    ):
        with pytest.raises(ValueError):
            sr.PyHestonSlv(leverage=unit, **{**kwargs, **bad})
    spots, times, ones = np.array([50.0, 150.0]), np.array([0.0, 1.0]), np.ones((2, 2))
    with pytest.raises(ValueError, match="float64"):
        sr.PyHestonSlv(leverage=(spots.astype(np.float32), times, ones), **kwargs)
    with pytest.raises(ValueError):
        sr.PyHestonSlv(leverage=(np.array([50.0, math.nan]), times, ones), **kwargs)
    with pytest.raises(ValueError):
        sr.PyHestonSlv(leverage=(spots, times, np.full((2, 2), math.nan)), **kwargs)
    with pytest.raises(TypeError):
        sr.PyHestonSlv(leverage=1.0, **kwargs)


@pytest.mark.parametrize("operation", ["leverage", "sample", "sample_par"])
def test_heston_slv_callback_failures_are_runtime_errors(operation):
    def broken(t, s):
        raise RuntimeError("leverage callback failed")

    process = sr.PyHestonSlv(2.0, 0.04, 0.3, -0.7, 0.05, 0.6, broken, 5, 1.0, 0.04, seed=7)
    with pytest.raises(RuntimeError, match="leverage callback failed"):
        if operation == "leverage":
            process.leverage(0.0, 100.0)
        elif operation == "sample":
            process.sample()
        else:
            process.sample_par(3)


def _heston(**kwargs):
    return sr.PyHeston(1.5, 0.04, 0.5, -0.7, 0.02, 53, s0=100.0, v0=0.04, t=1.0, **kwargs)


def test_heston_qe_prices_the_atm_call_within_monte_carlo_error():
    exact = sr.HestonFourier(0.04, 1.5, 0.04, 0.5, -0.7).price_call(
        100.0, 100.0, 0.02, 0.0, 1.0
    )
    errors = []
    for seed in (7, 2718, 42):
        s, v = _heston(seed=seed, scheme="qe").sample_par(100_000)
        payoff = np.exp(-0.02) * np.maximum(s[:, -1] - 100.0, 0.0)
        assert np.all(v >= 0.0)
        errors.append(abs(payoff.mean() - exact) / (payoff.std() / np.sqrt(payoff.size)))
    assert min(errors) < 4.0, errors


def test_heston_qe_is_seeded_and_validated():
    a = _heston(seed=3, scheme="qe").sample_par(64)
    b = _heston(seed=3, scheme="qe").sample_par(64)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], _heston(seed=3).sample_par(64)[0])
    for bad in ({"scheme": "milstein"}, {"scheme": "qe", "pow": "3/2"}, {"pow": "half"}):
        with pytest.raises(ValueError):
            _heston(**bad)
    with pytest.raises(ValueError):
        sr.PyHeston(0.0, 0.04, 0.5, -0.7, 0.02, 53, s0=100.0, v0=0.04, t=1.0, scheme="qe")


def test_the_verhulst_clamp_kwarg_defaults_to_clamping():
    def path(**kw):
        return sr.PyVerhulst(1.0, 2.0, 3.0, 128, x0=1.0, t=1.0, seed=9, **kw).sample()

    clamped = path(clamp=True)
    assert np.all((clamped >= 0.0) & (clamped <= 2.0))
    assert np.array_equal(path(), clamped)
    assert np.array_equal(path(clamp=None), clamped)
    assert not np.array_equal(path(clamp=False), clamped)


def test_the_boundary_kwarg_selects_the_variance_step():
    def variance(boundary):
        kwargs = {"s0": 100.0, "v0": 0.0, "t": 1.0, "seed": 3, "boundary": boundary}
        return sr.PyHeston(2.0, 0.04, 2.0, -0.7, 0.03, 65, **kwargs).sample()[1]

    absorbed, reflect = variance("absorption"), variance("reflect")
    truncated = variance("full_truncation")
    assert np.all(absorbed >= 0.0) and np.all(reflect >= 0.0) and np.all(truncated >= 0.0)
    split = int(np.argmax(absorbed != reflect))
    assert absorbed[split] != reflect[split]
    assert absorbed[split] == 0.0 and reflect[split] > 0.0
    assert truncated[split] == 0.0 and not np.array_equal(truncated, absorbed)
    assert np.array_equal(variance(None), truncated)
    cir = sr.PyCir(1.5, 0.04, 0.3, 64, x0=0.04, t=1.0, seed=5, boundary="full_truncation").sample()
    assert np.array_equal(cir, sr.PyCir(1.5, 0.04, 0.3, 64, x0=0.04, t=1.0, seed=5).sample())
    for build in (variance, lambda b: sr.PyCir(1.5, 0.04, 0.3, 64, boundary=b)):
        with pytest.raises(ValueError):
            build("floor")


def test_a_log_price_model_takes_exactly_one_drift():
    def gbm_log(**drift):
        return sr.PyGbmLog(sigma=0.2, n=64, s0=1.0, t=1.0, seed=4, **drift).sample()

    assert np.array_equal(gbm_log(mu=0.5), gbm_log(b=0.5))
    assert np.array_equal(gbm_log(mu=0.5), gbm_log(r=0.75, r_f=0.25))
    assert not np.array_equal(gbm_log(mu=0.5), gbm_log(mu=0.25))
    builds = [
        lambda **d: sr.PyGbmLog(sigma=0.2, n=64, s0=1.0, **d),
        lambda **d: sr.PyHestonLog(kappa=1.5, theta=0.04, xi=0.3, rho=-0.7, n=64, s0=1.0, v0=0.04, **d),
        lambda **d: sr.PyMjdLog(sigma=0.2, lambda_=0.5, nu=-0.1, omega=0.15, n=64, s0=1.0, **d),
        lambda **d: sr.PyBates(0.5, 0.0, 0.08, 2.0, 0.3, -0.5, lambda: 0.1, 64, s0=1.0, v0=0.0, **d),
        lambda **d: sr.PyBatesSvj(0.5, 0.0, 0.1, 0.08, 2.0, 0.3, -0.7, 64, s0=1.0, v0=0.0, **d),
    ]
    for build in builds:
        build(mu=0.05)
        for bad in ({}, {"mu": 0.05, "b": 0.01}, {"r": 0.05}, {"mu": 0.05, "r": 0.05, "r_f": 0.0}):
            with pytest.raises(ValueError, match="exactly one drift"):
                build(**bad)


def test_an_event_process_takes_exactly_one_of_n_and_t_max():
    builds = [
        lambda **h: sr.PyPoisson(2.0, **h),
        lambda **h: sr.PyHawkes(0.5, 0.3, 1.0, **h),
        lambda **h: sr.PyCustomJt(lambda: 0.1, **h),
        lambda **h: sr.PyCompoundCustom(lambda: 0.1, lambda: 0.1, **h),
        lambda **h: sr.PyCompoundPoisson(lambda: 0.1, 2.0, **h),
    ]
    for build in builds:
        out = build(n=10).sample()
        assert {len(row) for row in (out if isinstance(out, tuple) else (out,))} == {11}
        build(t_max=1.0)
        for bad in ({}, {"n": 10, "t_max": 1.0}):
            with pytest.raises(ValueError, match="exactly one of n and t_max"):
                build(**bad)
    assert len(sr.PyPoisson(3.0, n=10, seed=2).sample()) == 11
    assert np.all(sr.PyPoisson(3.0, t_max=2.0, seed=2).sample() <= 2.0)
    assert sr.PyPoisson(3.0, n=10).device_ready() and sr.PyHawkes(0.5, 0.3, 1.0, n=10).device_ready()
    assert not sr.PyPoisson(3.0, t_max=2.0).device_ready()
    assert sr.PyHawkes(0.5, 0.3, 1.0, t_max=2.0).device_fallback() is not None


def test_a_zero_rate_counts_no_arrival():
    assert list(sr.PyPoisson(0.0, t_max=2.0, seed=2).sample()) == [0.0]
    assert list(sr.PyPoisson(0.0, n=0, seed=2).sample()) == [0.0]
    with pytest.raises(ValueError, match=r"Events\(1\.\.\)"):
        sr.PyPoisson(0.0, n=3)
    with pytest.raises(ValueError, match="lambda >= T::zero"):
        sr.PyPoisson(-1.0, t_max=2.0)


def test_ho_lee_takes_exactly_one_of_theta_and_forward_curve():
    sr.PyHoLee(0.01, 64, theta=0.02)
    sr.PyHoLee(0.01, 64, forward_curve=lambda t: 0.03)
    for bad in ({}, {"theta": 0.02, "forward_curve": lambda t: 0.03}):
        with pytest.raises(ValueError, match="exactly one drift: theta or forward_curve"):
            sr.PyHoLee(0.01, 64, **bad)
    with pytest.raises(TypeError):
        sr.PyHoLee(0.01, 64, f_T=lambda t: 0.03)


def test_the_fractional_sheet_names_its_grid_rows_and_cols():
    sheet = sr.PyFbs(0.7, rows=9, cols=5, r=1.0, seed=3).sample()
    assert sheet.shape == (9, 5)
    with pytest.raises(TypeError):
        sr.PyFbs(0.7, m=9, n=5, r=1.0)


def test_the_rough_heston_spot_drift_and_correlation_are_checked_at_construction():
    for bad, match in (({"s0": 0.0}, "s0 must satisfy"), ({"mu": math.nan}, "mu must satisfy"),
                       ({"rho": 1.5}, "rho must satisfy")):
        with pytest.raises(ValueError, match=match):
            sr.PyRoughHeston(0.1, 0.04, 1.0, 0.3, 64, 0.2, **bad)


def test_the_bates_svj_jump_is_lognormal_and_compensated_by_its_mean():
    def bates_svj(nu, omega, lambda_=5.0):
        return sr.PyBatesSvj(lambda_, nu, omega, 0.0, 0.0, 0.0, 0.0, 33, mu=0.0, s0=100.0, v0=0.0,
                             t=1.0, seed=5)

    terminal = bates_svj(-0.1, 0.2).sample_par(20_000)[0][:, -1]
    assert abs(terminal.mean() - 100.0) < 5.0 * terminal.std(ddof=1) / math.sqrt(terminal.size)
    quarterly = sr.PyBatesSvj(1.0, -1.0, 0.5, 0.5, 2.0, 0.5, -0.7, 5, mu=0.05, s0=100.0, v0=0.25,
                              t=1.0, seed=11)
    assert np.all(quarterly.sample_par(20_000)[0] > 0.0)


def test_the_bates_svj_takes_its_jumps_to_a_device():
    def bates_svj(**device):
        return sr.PyBatesSvj(3.0, -0.05, 0.1, 0.08, 2.0, 0.3, -0.7, 64, mu=0.02, s0=100.0, v0=0.04,
                             t=1.0, seed=7, dtype="f32", **device)

    assert bates_svj().device_ready() and bates_svj().device_fallback() is None
    host = bates_svj().sample_par(256)[0]
    for device in ("cuda", "metal"):
        try:
            on_device = bates_svj(device=device)
        except (ValueError, RuntimeError):
            continue
        spots = on_device.sample_par(256)[0]
        assert spots.shape == host.shape and np.all(spots > 0.0)
        assert not np.array_equal(spots, host)


def test_the_inverse_gaussian_process_is_the_unit_subordinator():
    assert not hasattr(sr, "PyIg")
    path = sr.PyIGSubordinator(2.0, 1.0, 65, seed=3).sample()
    assert path.shape == (65,) and path[0] == 0.0 and np.all(np.diff(path) > 0.0)


def test_white_noise_takes_its_location_as_mu_and_its_scale_as_sigma():
    flat = sr.PyWn(64, mu=0.5, sigma=0.0, seed=1).sample()
    assert np.all(flat == 0.5)
    assert np.array_equal(sr.PyWn(64, sigma=2.0, seed=1).sample(), 2.0 * sr.PyWn(64, seed=1).sample())
    for old in ({"std_dev": 1.0}, {"mean": 0.5}):
        with pytest.raises(TypeError):
            sr.PyWn(64, **old)
