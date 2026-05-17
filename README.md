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

📖 **[stochastic.rust-dd.com](https://stochastic.rust-dd.com)** —
full docs site (Fumadocs + Next.js, deployed on Vercel).

Local preview from source under [`website/`](website/):

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
see the [installation guide](https://stochastic.rust-dd.com/docs/getting-started/installation-rust)
on the docs site.

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
[Python bindings page](https://stochastic.rust-dd.com/docs/python) for the parity
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
[tutorials section](https://stochastic.rust-dd.com/docs/tutorials).

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

### Distribution sampling — `Normal` vs upstream `rand_distr`

Single-thread `fill_slice`, median of 7 runs (`cargo bench --bench
dist_multicore`). Comparison column:

- **`rand_distr + SimdRng`** — `rand_distr::Normal` consuming our `SimdRng`
  (same uniform stream, only the Normal algorithm differs).
- **`rand_distr + rand::rng()`** — the out-of-box upstream pipeline.

|     n  | `SimdNormal` (µs) | `rand_distr + SimdRng` (µs) | speedup | `rand_distr + rand::rng()` (µs) | speedup |
|-------:|------------------:|-----------------------------:|--------:|---------------------------------:|--------:|
|      4 |             0.008 |                        0.013 |  1.73×  |                            0.032 |  4.22×  |
|      8 |             0.014 |                        0.026 |  1.78×  |                            0.065 |  4.52×  |
|     16 |             0.029 |                        0.051 |  1.79×  |                            0.128 |  4.47×  |
|     64 |             0.109 |                        0.208 |  1.90×  |                            0.508 |  4.64×  |
|    256 |             0.432 |                        0.840 |  1.94×  |                            2.029 |  4.70×  |
|  4 096 |             6.975 |                       13.176 |  1.89×  |                           32.382 |  4.64×  |
| 65 536 |           113.458 |                      212.406 |  1.87×  |                          520.219 |  4.59×  |

### Single-sample speedup vs prior release

Criterion `dist.sample(rng)` loop, vs the `wide 1.3.0` baseline
(`cargo bench --bench distributions -- --baseline before`):

|       distribution | f32 / large       | f64 / large       | f64 / small       |
|-------------------:|------------------:|------------------:|------------------:|
|     `Uniform/simd` | **−57%** (≈ 2.3×) | **−77%** (≈ 4.4×) | **−58%** (≈ 2.4×) |
|      `Normal/simd` | **−51%** (≈ 2.0×) | **−75%** (≈ 4.0×) | **−63%** (≈ 2.7×) |
|    `Exp/simd N=64` | −3% (n.s.)        | **−73%** (≈ 3.7×) | —                 |
|   `LogNormal/simd` | **−71%** (≈ 3.4×) | **−70%** (≈ 3.4×) | **−66%** (≈ 2.9×) |

Driven by SIMD `u64→f64` / `u32→f32` magic-number conversion in `SimdRng`
(direct-write `fill_uniform_f64` / `fill_uniform_f32` APIs that skip the
`[f64; 8]` return-by-value round-trip), fused Exp(λ) scaling inside
`fill_exp_scaled`, and an 8-at-a-time main loop in `fill_ziggurat` so
`copy_from_slice` inlines to `stp` stores instead of a `memcpy` call.

#### Opt-in: dual-stream RNG (`dual-stream-rng` feature)

```toml
[dependencies]
stochastic-rs = { version = "2.1", features = ["dual-stream-rng"] }
```

Unlocks `SimdRngDual` (two parallel xoshiro engines) and `SimdNormalDual`
(Ziggurat unrolled 2× over the dual streams). Measured against the
single-stream `SimdNormal::fill_slice` on Apple Silicon
(`cargo bench --bench dual_stream_compare --features dual-stream-rng`):

|     n  | single (`SimdNormal`) | dual (`SimdNormalDual`) |   Δ   |
|-------:|----------------------:|-------------------------:|------:|
|     64 |              111.6 ns |                 105.5 ns | −5.5% |
|    256 |              444.8 ns |                 418.3 ns | −6.0% |
|   4 096 |              7.43 µs |                  6.60 µs |−11.2% |
|  65 536 |             113.9 µs |                 106.6 µs | −6.4% |
| 1 048 576 |            1.83 ms |                  1.70 ms | −6.8% |

The win comes from hiding the 16 scalar `kn` / `wn` table-lookup latencies
behind the second engine's `xoshiro` state update on a modern out-of-order
core. Uniform fills are not bottlenecked on the engine so they see no
speedup. Trade-off: `SimdRngDual::from_seed` does **not** reproduce
`SimdRng::from_seed`'s bit-exact sequence (statistical properties are
identical and KS-validated).

## Contributing

Contributions are welcome — bug reports, feature suggestions, or PRs.
Open an issue or start a discussion on GitHub. Per-feature recipes
(`add-diffusion-process`, `adding-distribution`, `calibration-pattern`,
`docs-writing`, …) live under [`.claude/skills/`](.claude/skills/).

## License

MIT — see [LICENSE](https://github.com/dancixx/stochastic-rs/blob/main/LICENSE).
