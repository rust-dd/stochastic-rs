![Build Workflow](https://github.com/rust-dd/stochastic-rs/actions/workflows/rust.yml/badge.svg)
[![Crates.io](https://img.shields.io/crates/v/stochastic-rs?style=flat-square)](https://crates.io/crates/stochastic-rs)
[![docs.rs](https://img.shields.io/docsrs/stochastic-rs?style=flat-square)](https://docs.rs/stochastic-rs)
[![Downloads](https://img.shields.io/crates/d/stochastic-rs?style=flat-square)](https://crates.io/crates/stochastic-rs)
[![PyPI](https://img.shields.io/pypi/v/stochastic-rs?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/stochastic-rs/)
![License](https://img.shields.io/crates/l/stochastic-rs?style=flat-square)
[![codecov](https://codecov.io/gh/rust-dd/stochastic-rs/graph/badge.svg)](https://codecov.io/gh/rust-dd/stochastic-rs)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21553307.svg)](https://doi.org/10.5281/zenodo.21553307)

# stochastic-rs

**Quantitative finance in Rust**: stochastic process simulation, option
pricing and calibration, volatility surfaces, fixed income and credit, risk,
statistics, copulas and neural volatility surrogates. Generic over `f32` /
`f64`, SIMD on the CPU, CUDA / Metal / CubeCL back-ends where they pay off, and
Python bindings via PyO3 that ship the same surface as the Rust crates.

## Documentation

📖 **[stochastic.rust-dd.com](https://stochastic.rust-dd.com)** is the reference; this
README only gets you installed and running.

- [Getting started](https://stochastic.rust-dd.com/docs/getting-started/quickstart) — Rust and Python installation, first program
- [Concepts](https://stochastic.rust-dd.com/docs/concepts/traits) — the traits (`ProcessExt`, `DistributionExt`, `ModelPricer`), seeding, feature flags, [design philosophy](https://stochastic.rust-dd.com/docs/concepts/design-philosophy)
- [Processes](https://stochastic.rust-dd.com/docs/processes) · [Distributions](https://stochastic.rust-dd.com/docs/distributions) · [Copulas](https://stochastic.rust-dd.com/docs/copulas) · [Statistics](https://stochastic.rust-dd.com/docs/stats) · [Quant](https://stochastic.rust-dd.com/docs/quant) · [AI](https://stochastic.rust-dd.com/docs/ai) — the catalogues with selection guides
- [GPU support](https://stochastic.rust-dd.com/docs/concepts/gpu-support) — what runs on which device today, precision, an executed T4 run
- [Python](https://stochastic.rust-dd.com/docs/python) — the bindings, `device=`, NumPy interop
- [Benchmarks](https://stochastic.rust-dd.com/docs/benchmarks) · [Migrating to v3](https://stochastic.rust-dd.com/docs/migration) · [Tutorials](https://stochastic.rust-dd.com/docs/tutorials)

## What is inside

One workspace, one umbrella crate (`stochastic-rs`) that re-exports the sub-crates:

| Crate | Contents |
|---|---|
| `stochastic-rs-core` | the SIMD RNG and seed sources (`Deterministic`, `Unseeded`) |
| `stochastic-rs-distributions` | SIMD samplers with closed-form pdf / cdf / characteristic function / moments, special functions |
| `stochastic-rs-stochastic` | 131 processes behind one `ProcessExt` trait: diffusion, jump, stochastic and rough volatility, short rate, HJM / LMM, fractional noise, Volterra |
| `stochastic-rs-copulas` | 15 bivariate and 8 multivariate copulas, vine fitting, goodness of fit |
| `stochastic-rs-stats` | Hurst and diffusion estimators, unit-root and cointegration tests, realised volatility, filters, extreme values, risk measures |
| `stochastic-rs-quant` | closed-form, Fourier, PDE, lattice and Monte Carlo pricers, calibrators, vol surfaces, curves, credit, XVA, market microstructure |
| `stochastic-rs-ai` | neural volatility surrogates and surrogate calibration (`ai` feature) |
| `stochastic-rs-py` | the Python module: every distribution, process, pricer, copula and estimator, NumPy in and out |

## Installation

```toml
[dependencies]
stochastic-rs = "3.0.0-rc.1"
```

Device back-ends and other optional parts are cargo features (`cuda`,
`metal`, `cubecl-cuda` / `cubecl-wgpu`, `accelerate`, `ai`, `dual-stream-rng`);
the [installation guide](https://stochastic.rust-dd.com/docs/getting-started/installation-rust)
and the [feature flags](https://stochastic.rust-dd.com/docs/concepts/feature-flags)
page list them with what each pulls in. Sub-crates can be depended on directly
for lean builds.

```bash
pip install stochastic-rs
```

The wheels are CPU-only and carry the whole surface on Linux, macOS and Windows
(linear algebra is pure Rust). A source build with a device back-end:
`maturin develop --release --features metal` (or `cuda`) in a checkout.

## Quickstart

```rust
use stochastic_rs::prelude::*;
use stochastic_rs::simd_rng::Unseeded;
use stochastic_rs::stochastic::diffusion::ou::Ou;
use stochastic_rs::quant::pricing::heston::HestonPricer;

fn main() {
    // Mean-reverting Ornstein-Uhlenbeck path: Ou::new(theta, mu, sigma, n, x0, t, seed)
    let ou = Ou::<f64>::new(2.0, 0.0, 1.0, 1_000, Some(0.0), Some(1.0), Unseeded);
    let path = ou.sample();
    println!("OU path points: {}", path.len());

    // Heston (1993) European option, closed form. The model holds only its own
    // parameters; the pricing query is passed to the call, so one model can
    // price a whole strike/maturity grid.
    // HestonPricer::new args: v0, rho, kappa, theta, sigma, lambda
    let pricer = HestonPricer::new(0.04, -0.5, 2.0, 0.04, 0.3, Some(0.0));
    // price_call/price_put args: s, k, r, q, tau
    let call = pricer.price_call(100.0, 100.0, 0.03, 0.0, 1.0);
    let put = pricer.price_put(100.0, 100.0, 0.03, 0.0, 1.0);
    println!("call={call:.4}, put={put:.4}");
}
```

```python
import stochastic_rs as srs

# Mean-reverting OU path: PyOu(theta, mu, sigma, n, x0=None, t=None, seed=None, dtype=None, device=None)
path = srs.PyOu(2.0, 0.0, 1.0, 1000, x0=0.0, t=1.0, seed=42).sample()   # numpy.ndarray, shape (1000,)

# Heston European option, closed form
pricer = srs.HestonPricer(
    s=100, v0=0.04, k=100, r=0.03, kappa=2.0, theta=0.04, sigma=0.3,
    rho=-0.5, tau=1.0, q=0.0,
)
call, put = pricer.call_put()
```

A process samples on a device by re-typing it: `Gbm::new(...).on::<Metal>()`
(`Cuda`, `CubeclCuda`, `CubeclWgpu`, `Accelerate`), with `handle.probe()` to check the device
first; from Python, `device="metal"` on the device-capable classes. The
[GPU support](https://stochastic.rust-dd.com/docs/concepts/gpu-support) page has
the support matrix, and [`notebooks/`](notebooks/) a Colab notebook that runs the
CUDA back-end on a free T4.

## Benchmarks

Criterion suites live under `benches/`; the
[benchmarks page](https://stochastic.rust-dd.com/docs/benchmarks) carries the
numbers: the SIMD Normal sampler against `rand_distr`, fractional Gaussian noise
on CPU, Accelerate, Metal, CubeCL and cuFFT, and the per-release speedups.

## Citing

The concept DOI [10.5281/zenodo.21553307](https://doi.org/10.5281/zenodo.21553307)
always resolves to the latest release; [`CITATION.cff`](CITATION.cff) carries the
version DOI of the current one.

## Contributing

Bug reports, suggestions and pull requests are welcome on GitHub. The
[contributing page](https://stochastic.rust-dd.com/docs/contributing) has the
development rules; per-feature recipes (`add-diffusion-process`,
`adding-distribution`, `calibration-pattern`, …) live under
[`.claude/skills/`](.claude/skills/).

## License

MIT — see [LICENSE](https://github.com/rust-dd/stochastic-rs/blob/main/LICENSE).
