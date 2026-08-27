[![Crates.io](https://img.shields.io/crates/v/stochastic-rs-stochastic?style=flat-square)](https://crates.io/crates/stochastic-rs-stochastic)
[![docs.rs](https://img.shields.io/docsrs/stochastic-rs-stochastic?style=flat-square)](https://docs.rs/stochastic-rs-stochastic)
![License](https://img.shields.io/crates/l/stochastic-rs-stochastic?style=flat-square)

# stochastic-rs-stochastic

**Stochastic process simulation — 120+ processes**

The simulation engine: 120+ processes behind one `ProcessExt<T>` trait,
generic over `f32` / `f64`, with optional GPU backends for the fractional
family.

## What is in it

- **Diffusions** — GBM, OU, CIR, CEV, CKLS, Jacobi, Ait-Sahalia, Pearson,
  Verhulst, Kimura, three-halves, and their fractional counterparts.
- **Jump processes** — Merton, Kou, Bates, CGMY, KoBoL, variance gamma,
  NIG, bilateral gamma, Hawkes and multivariate Hawkes, compound Poisson.
- **Stochastic volatility** — Heston (several schemes), double Heston,
  multifactor Heston, SABR, multifactor SABR, Bergomi, rough Bergomi,
  fractional Heston, BNS, SVCGMY, HKDE.
- **Rough / fractional** — fBM, fGN, Riemann-Liouville variants, Volterra
  kernels, and a Markovian lift for rough volatility.
- **Interest rates** — Vasicek, Hull-White (1F/2F), Ho-Lee, CIR (1F/2F),
  HJM, LMM with drift coupling, Duffie-Kan, Wu-Zhang, ADG.
- **Monte Carlo** — antithetic, control variates, stratified sampling,
  importance sampling, Halton and Sobol sequences, Longstaff-Schwartz, MLMC.
- **Backends** — CPU SIMD by default; CUDA, Metal, Accelerate and cubecl
  for fractional Gaussian noise via `.on::<B>()`.

## Usage

```rust
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::diffusion::ou::Ou;
use stochastic_rs_stochastic::traits::ProcessExt;

let ou = Ou::<f64>::new(2.0, 0.0, 1.0, 1_000, Some(0.0), Some(1.0), Unseeded);
let path = ou.sample();          // Array1<f64>, length 1000
let paths = ou.sample_par(512);  // 512 paths across rayon workers
```

## Part of stochastic-rs

This crate is one of the sub-crates of
[**stochastic-rs**](https://github.com/rust-dd/stochastic-rs). Most users
should depend on the umbrella crate, which re-exports everything:

```toml
[dependencies]
stochastic-rs = "3.0.0-beta.1"
```

Depend on `stochastic-rs-stochastic` directly only when you want this slice and nothing else.

- Documentation: [stochastic.rust-dd.com](https://stochastic.rust-dd.com)
- API reference: [docs.rs/stochastic-rs-stochastic](https://docs.rs/stochastic-rs-stochastic)

## License

MIT
