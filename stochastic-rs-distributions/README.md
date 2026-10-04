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
  moments. Every method returns `Option`: `Some` where the family has a
  closed form, `None` where it has none — never a silent zero.
- **`FloatExt` / `SimdFloatExt`** — the numeric trait bounds the whole
  workspace is generic over.

## Usage

```rust
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

let mut dist = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(42));
let mut xs = vec![0.0; 10_000];
dist.fill_slice(&mut xs);            // amortised SIMD fill
```

For a single draw from a shared generator:

```rust
use rand::distr::Distribution;
use stochastic_rs_distributions::normal::SimdNormal;

let d = SimdNormal::<f64>::new(0.0, 1.0);
let z = d.sample(&mut rng);
```

## Part of stochastic-rs

This crate is one of the sub-crates of
[**stochastic-rs**](https://github.com/rust-dd/stochastic-rs). Most users
should depend on the umbrella crate, which re-exports everything:

```toml
[dependencies]
stochastic-rs = "3.0.0-rc.4"
```

Depend on `stochastic-rs-distributions` directly only when you want this slice and nothing else.

- Documentation: [stochastic.rust-dd.com](https://stochastic.rust-dd.com)
- Tutorials: [stochastic.rust-dd.com/docs/tutorials](https://stochastic.rust-dd.com/docs/tutorials)
- API reference: [docs.rs/stochastic-rs-distributions](https://docs.rs/stochastic-rs-distributions)

## License

MIT
