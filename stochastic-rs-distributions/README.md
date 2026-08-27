[![Crates.io](https://img.shields.io/crates/v/stochastic-rs-distributions?style=flat-square)](https://crates.io/crates/stochastic-rs-distributions)
[![docs.rs](https://img.shields.io/docsrs/stochastic-rs-distributions?style=flat-square)](https://docs.rs/stochastic-rs-distributions)
![License](https://img.shields.io/crates/l/stochastic-rs-distributions?style=flat-square)

# stochastic-rs-distributions

**Probability distributions with SIMD bulk sampling**

Nineteen distributions, generic over `f32` / `f64`, with closed-form
analytics and SIMD-accelerated bulk generation.

## What is in it

- **`Simd*` distributions** — `SimdNormal`, `SimdExp`, `SimdGamma`,
  `SimdPoisson`, `SimdBeta`, `SimdStudentT`, `SimdAlphaStable`,
  `SimdInverseGauss`, `SimdNormalInverseGauss`, `SimdGev`, `SimdGed`,
  `SimdWeibull`, `SimdPareto`, `SimdCauchy`, `SimdChiSquared`,
  `SimdBinomial`, `SimdGeometric`, `SimdHypergeometric`, `SimdSkellam`,
  plus `SimdDirichlet` and `SimdWishart`. Ziggurat, rejection, inversion or
  transformation sampling depending on the family.
- **`DistributionExt`** — closed-form pdf, cdf, characteristic function and
  moments. 18 of 19 families implement it in closed form; the remaining
  gaps raise `unimplemented!` by name rather than returning a silent zero.
- **`scalar` module** — `ScalarNormal` and `ScalarExp`: stateless,
  `Copy + Send + Sync` samplers that draw from the caller's RNG. Use these
  wherever a process requires `D: Distribution<T> + Send + Sync`; the
  `Simd*` types are `!Sync` by construction.
- **`FloatExt` / `SimdFloatExt`** — the numeric trait bounds the whole
  workspace is generic over.

## Usage

```rust
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::normal::SimdNormal;

let dist = SimdNormal::<f64>::new(0.0, 1.0, &Deterministic::new(42));
let mut xs = vec![0.0; 10_000];
dist.fill_slice(&mut xs);            // amortised SIMD fill
```

For a single draw from a shared RNG in a `Sync` context:

```rust
use rand_distr::Distribution;
use stochastic_rs_distributions::scalar::ScalarNormal;

let d = ScalarNormal::<f64>::new(0.0, 1.0);
let z = d.sample(&mut rng);
```

## Part of stochastic-rs

This crate is one of the sub-crates of
[**stochastic-rs**](https://github.com/rust-dd/stochastic-rs). Most users
should depend on the umbrella crate, which re-exports everything:

```toml
[dependencies]
stochastic-rs = "3.0.0-beta.2"
```

Depend on `stochastic-rs-distributions` directly only when you want this slice and nothing else.

- Documentation: [stochastic.rust-dd.com](https://stochastic.rust-dd.com)
- API reference: [docs.rs/stochastic-rs-distributions](https://docs.rs/stochastic-rs-distributions)

## License

MIT
