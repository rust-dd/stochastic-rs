![Build Workflow](https://github.com/dancixx/stochastic-rs/actions/workflows/rust.yml/badge.svg)
[![Crates.io](https://img.shields.io/crates/v/stochastic-rs?style=flat-square)](https://crates.io/crates/stochastic-rs)
![License](https://img.shields.io/crates/l/stochastic-rs?style=flat-square)
[![codecov](https://codecov.io/gh/dancixx/stochastic-rs/graph/badge.svg?token=SCSp3z7BQJ)](https://codecov.io/gh/dancixx/stochastic-rs)

# stochastic-rs

A high-performance Rust library for stochastic process simulation,
quantitative finance, statistics, copulas, distributions, and
neural-network volatility surrogates. Generic over `f32` / `f64`, with
SIMD acceleration on CPU and CUDA / Metal / Accelerate / cubecl backends
where they pay off, and first-class Python bindings via PyO3.

## Documentation

The full docs site lives under [`website/`](website/) (Fumadocs +
Next.js). Local preview:

```bash
cd website
bun install
bun run dev          # http://localhost:3000
```

Highlights:

- **120+ stochastic processes** — diffusion, jump, fractional / rough,
  short-rate, HJM, LMM, fBM, Hawkes, Lévy. Generic-precision
  `ProcessExt<T>` impl, SIMD on CPU, optional CUDA / Metal for FGN /
  fBM.
- **Pricing & calibration** — closed-form (BSM, Bachelier, Black76,
  Bjerksund-Stensland, …), Fourier (Heston / Bates / Merton-jump / Kou
  / VG / CGMY / HKDE / double-Heston), Monte Carlo (basket, rainbow,
  cliquet, autocallable, spread), finite difference, Bermudan LSM,
  Heston SLV. Heston / SABR / SVJ / Lévy / rough Bergomi / double-Heston
  / Hull-White swaption-grid calibrators.
- **Statistics & risk** — Hurst (Fukasawa), MLE for 1-D diffusions
  with 6 transition densities, ADF / KPSS / Phillips-Perron, realised
  variance with BNHLS bandwidth, HMM, changepoint, particle filter, UKF.
  VaR / CVaR / drawdown, Sharpe / Sortino / IR / Calmar.
- **Fixed income & credit** — yield-curve bootstrapping, Nelson-Siegel /
  Svensson, multi-curve, IRS / inflation swaps, Vasicek / CIR /
  Hull-White / G2++ short-rate engines, Merton structural model,
  reduced-form survival curves, CDS pricing, JLT migration matrices.
- **Microstructure** — Almgren-Chriss, Kyle (1985), Bouchaud propagator,
  full price-time priority order book.
- **Distributions & copulas** — 19 SIMD distributions with
  closed-form pdf / cdf / cf / moments. Clayton / Frank / Gumbel /
  Independence bivariate; Gaussian / vine multivariate.
- **Python bindings** — 210 entries (198 PyO3 classes + 12 functions)
  spanning every sub-crate except AI surrogates. Numpy-in / numpy-out.

## Installation

### Rust

```toml
[dependencies]
stochastic-rs = "2.0.0"
```

```rust
use stochastic_rs::prelude::*;
use stochastic_rs::stochastic::diffusion::gbm::Gbm;
use stochastic_rs::quant::pricing::heston::HestonPricer;
```

For per-sub-crate (lean) builds, OpenBLAS / CUDA / Metal / cubecl /
Accelerate feature flags, native CPU optimisation, and SIMD details,
see the [installation guide](website/content/docs/getting-started/installation-rust.mdx)
in the docs site.

### Python

```bash
pip install stochastic-rs
```

Source build (requires the Rust toolchain):

```bash
pip install maturin
maturin develop --release --manifest-path stochastic-rs-py/Cargo.toml
```

Linux (x86_64 / aarch64) and macOS (arm64 / x86_64) wheels ship with
the `openblas` feature on. The Windows wheel omits the 15
BLAS-backed classes; everything else (≈195 classes / 12 functions)
works identically. See the
[Python bindings page](website/content/docs/python.mdx) for the parity
table and the source-build path with vcpkg.

## Quickstart

```rust
use stochastic_rs::prelude::*;
use stochastic_rs::stochastic::diffusion::ou::Ou;
use stochastic_rs::quant::pricing::heston::HestonPricer;
use stochastic_rs::quant::types::OptionType;

fn main() {
    // Mean-reverting OU path
    let p = Ou::<f64>::new(2.0, 0.0, 1.0, 1_000, Some(0.0), Some(1.0));
    let path = p.sample();

    // Heston European call with first- and second-order Greeks
    let pricer = HestonPricer::<f64>::new(
        100.0, 100.0, 1.0, 0.03, 0.0,
        0.04, 2.0, 0.04, 0.3, -0.5,
    );
    let price = pricer.price(OptionType::Call);
    let greeks = pricer.greeks(OptionType::Call);
    println!("call={:.4}, delta={:.4}, vega={:.4}", price, greeks.delta, greeks.vega);
}
```

```python
import stochastic_rs as srs

# Mean-reverting OU path
p = srs.Ou(theta=2.0, mu=0.0, sigma=1.0, n=1000, x0=0.0, t=1.0)
path = p.sample()                       # numpy.ndarray, shape (1000,)

# Heston European call
pricer = srs.HestonPricer(
    s0=100, k=100, tau=1.0, r=0.03, q=0.0,
    v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5,
)
print("call =", pricer.price("call"))
g = pricer.greeks("call")
print(f"delta={g.delta:.4f}, vega={g.vega:.4f}")
```

More end-to-end recipes (Heston calibration, fBM Hurst estimation,
vol-surface from quotes, Python interop) live in the
[tutorials section](website/content/docs/tutorials.mdx).

## Benchmarks

### FGN — CPU vs CUDA native (`f32`, H = 0.7)

```bash
cargo bench --features cuda-native --bench fgn_cuda_native
```

Single path:

| n      | CPU `sample` | CUDA `sample_cuda_native(1)` | Speedup    |
|-------:|-------------:|------------------------------:|-----------:|
|  1,024 |       8.1 µs |                          46 µs|       0.18× |
|  4,096 |        35 µs |                          84 µs|       0.42× |
| 16,384 |       147 µs |                         110 µs| **1.3×**    |
| 65,536 |       850 µs |                         227 µs| **3.7×**    |

Batch:

| n, m         | CPU `sample_par` | CUDA `sample_cuda_native` | Speedup  |
|--------------|------------------:|---------------------------:|---------:|
|   4,096, 32  |          147 µs   |                     117 µs |  **1.3×** |
|   4,096, 512 |         1.78 ms   |                    2.37 ms |   0.75×   |
|  65,536, 128 |         12.6 ms   |                    10.5 ms |  **1.2×** |
|  65,536, 1 k |          102 ms   |                      93 ms |  **1.1×** |

CUDA wins for large `n` (≥ 16 k); CPU rayon dominates for medium `n`
because of the GPU launch / transfer overhead.

### Distribution sampling — multicore (`cargo bench --bench dist_multicore`)

`sample_matrix`, 1-thread vs 14-thread rayon. `f64` continuous, integer
discrete. Most distributions: `1024 × 1024`; heavy discrete: `512 × 512`.

| Distribution        | 1T (ms) | MT (ms) | Speedup |
|---------------------|--------:|--------:|--------:|
| Normal              |   1.78  |   0.34  |  5.28×  |
| Cauchy              |   6.23  |   0.90  |  6.96×  |
| LogNormal           |   5.07  |   0.81  |  6.25×  |
| Gamma               |   5.20  |   0.72  |  7.19×  |
| StudentT            |   7.89  |   1.89  |  4.18×  |
| Beta                |  11.85  |   1.68  |  7.04×  |
| Weibull             |  13.17  |   1.73  |  7.59×  |
| AlphaStable         |  42.52  |   5.36  |  7.94×  |
| Poisson             |   2.28  |   0.42  |  5.40×  |
| Hypergeo (512²)     |  20.99  |   2.76  |  7.60×  |

(Full table — 18 distributions — on the
[benchmarks page](website/content/docs/benchmarks.mdx).)

`Normal` single-thread `fill_slice` vs the upstream `rand_distr` baseline:

- vs `rand_distr + SimdRng` — ≈ **1.21×** to **1.35×**
- vs `rand_distr + rand::rng()` — ≈ **4.09×** to **4.61×**

## Contributing

Contributions are welcome — bug reports, feature suggestions, or PRs.
Open an issue or start a discussion on GitHub. Per-feature recipes
(`add-diffusion-process`, `adding-distribution`, `calibration-pattern`,
`docs-writing`, …) live under [`.claude/skills/`](.claude/skills/).

## License

MIT — see [LICENSE](https://github.com/dancixx/stochastic-rs/blob/main/LICENSE).
