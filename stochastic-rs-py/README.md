[![PyPI](https://img.shields.io/pypi/v/stochastic-rs?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/stochastic-rs/)
[![Crates.io](https://img.shields.io/crates/v/stochastic-rs?style=flat-square)](https://crates.io/crates/stochastic-rs)
![License](https://img.shields.io/pypi/l/stochastic-rs?style=flat-square)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21553307.svg)](https://doi.org/10.5281/zenodo.21553307)

# stochastic-rs for Python

**Quantitative finance for Python, powered by Rust.** stochastic-rs is an
open-source quantitative-finance library: it simulates 132 stochastic
processes, prices and calibrates option models, builds volatility surfaces
and estimates model parameters from data. The Python module has 308
entries, takes and returns NumPy arrays, and ships prebuilt wheels for
Linux, macOS and Windows — the same code the
[Rust crate](https://crates.io/crates/stochastic-rs) runs.

📖 Documentation: **[stochastic.rust-dd.com](https://stochastic.rust-dd.com/docs/python)**

## Install

```bash
pip install stochastic-rs
```

Python 3.11 or newer. NumPy is the only dependency.

## Quickstart

```python
import stochastic_rs as srs

# Ornstein-Uhlenbeck path: PyOu(theta, mu, sigma, n, x0=None, t=None, seed=None)
path = srs.PyOu(2.0, 0.0, 1.0, 1000, x0=0.0, t=1.0, seed=42).sample()
print(path.shape)                    # (1000,)

# Heston (1993) European call and put in closed form
pricer = srs.HestonPricer(
    s=100, v0=0.04, k=100, r=0.03, kappa=2.0, theta=0.04, sigma=0.3,
    rho=-0.5, tau=1.0, q=0.0,
)
call, put = pricer.call_put()

# Read the Hurst exponent back from fractional Gaussian noise
noise = srs.PyFgn(0.3, 4096, t=1.0, seed=7).sample()
est = srs.RescaledRange(take_differences=False).estimate(noise)
print(round(est.hurst, 2))           # 0.38, for a true H of 0.3
```

Processes and distributions carry a `Py` prefix (`PyHeston`, `PyGbm`,
`PyNormal`); pricers, calibrators and estimators do not (`HestonPricer`,
`SabrCalibrator`, `FukasawaHurst`). Every process has `sample()` for one
path and `sample_par(m)` for `m` paths in parallel.

## What is inside

| Area | Examples |
|---|---|
| Stochastic processes | `PyGbm`, `PyHeston`, `PyRoughBergomi`, `PyFbm`, `PyCir`, `PyHullWhite`, `PyMerton`, `PyHawkes` |
| Option pricing | `BSMPricer`, `HestonPricer`, `SabrPricer`, `Merton1976Pricer`, Fourier pricers (`HestonFourier`, `CGMYFourier`, …) |
| Calibration | `HestonCalibrator`, `SabrCalibrator`, `SviCalibrator`, `SsviCalibrator`, `RBergomiCalibrator` |
| Estimation | `FukasawaHurst`, `RescaledRange`, `GarchFit`, `HestonMLE` |
| Distributions | `PyNormal`, `PyAlphaStable`, `PyNig`, `PyVarianceGamma` |
| Copulas | `Clayton`, `Gumbel`, `Frank`, `fit_vine` |

The [Python bindings page](https://stochastic.rust-dd.com/docs/python) lists
every entry. The tutorials walk through whole workflows with runnable Python:
[the Heston model](https://stochastic.rust-dd.com/docs/tutorials/heston) (simulate, price, calibrate),
[the Hurst exponent](https://stochastic.rust-dd.com/docs/tutorials/hurst-exponent),
[SVI and SSVI volatility surfaces](https://stochastic.rust-dd.com/docs/tutorials/svi-volatility-surface) and
[GPU paths on a free Colab GPU](https://stochastic.rust-dd.com/docs/tutorials/gpu-paths-on-colab).

## GPU sampling

The wheels run on the CPU. Device-capable classes take a `device=` argument
(`"cuda"`, `"metal"`), which needs a source build with that back-end:

```bash
pip install "maturin>=1.9.4"
maturin develop --release --features metal   # or: --features cuda
```

`srs.probe_device("metal")` reports whether the build can open a device. See
[GPU support](https://stochastic.rust-dd.com/docs/concepts/gpu-support) for
what runs where.

## Links

- Documentation: [stochastic.rust-dd.com](https://stochastic.rust-dd.com)
- Comparison with QuantLib and RustQuant: [stochastic.rust-dd.com/docs/comparison](https://stochastic.rust-dd.com/docs/comparison)
- Source: [github.com/rust-dd/stochastic-rs](https://github.com/rust-dd/stochastic-rs)
- Rust crate: [crates.io/crates/stochastic-rs](https://crates.io/crates/stochastic-rs)
- Cite: [doi.org/10.5281/zenodo.21553307](https://doi.org/10.5281/zenodo.21553307)

## License

MIT
