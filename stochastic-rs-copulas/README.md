[![Crates.io](https://img.shields.io/crates/v/stochastic-rs-copulas?style=flat-square)](https://crates.io/crates/stochastic-rs-copulas)
[![docs.rs](https://img.shields.io/docsrs/stochastic-rs-copulas?style=flat-square)](https://docs.rs/stochastic-rs-copulas)
![License](https://img.shields.io/crates/l/stochastic-rs-copulas?style=flat-square)

# stochastic-rs-copulas

**Bivariate, multivariate, and empirical copulas**

Dependence modelling: Archimedean and extreme-value families in two
dimensions, vine and nested constructions in higher dimensions.

## What is in it

- **Bivariate (`BivariateExt`)** — Clayton, Frank, Gumbel, Joe, AMH,
  Plackett, FGM, Galambos, Hüsler-Reiss, Marshall-Olkin, Student-t and
  independence. Each ships cdf, pdf, conditional inverse, Kendall's tau and
  a sampler.
- **Multivariate (`MultivariateExt`)** — Gaussian, Student-t, C-vine,
  D-vine, R-vine and nested Archimedean copulas, in the default build
  (linear algebra via the pure-Rust `faer`); the nested Archimedean copula
  learns its tree with `fit`.
- **Empirical** — pseudo-observations and the empirical copula.
- **Process coupling** — drive two stochastic processes through a copula.

Every sampler takes its seed: `sample(n, &Deterministic::new(42))` replays, `sample(n, &Unseeded)` draws fresh entropy.

## Usage

```rust
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::simd_rng::Deterministic;
use stochastic_rs_copulas::simd_rng::Unseeded;
use stochastic_rs_copulas::traits::BivariateExt;

let c = Clayton::new(2.0);
let u = c.sample(10_000, &Unseeded).expect("theta is set");                // Array2<f64>, shape (10000, 2)
let v = c.sample(10_000, &Deterministic::new(42)).expect("theta is set"); // reproducible

// Or set Kendall's tau and invert it for theta.
let from_tau = Clayton::from_tau(0.5).unwrap();
```

Multivariate constructions take a tree of `PairCopula` variants:

```rust
use stochastic_rs_copulas::multivariate::cvine::CVine;
use stochastic_rs_copulas::multivariate::dvine::PairCopula;
use stochastic_rs_copulas::simd_rng::Deterministic;
use stochastic_rs_copulas::traits::MultivariateExt;

let cv = CVine::new(2, vec![vec![PairCopula::Clayton { theta: 2.0 }]]).expect("a valid vine");
let draws = cv.sample(10_000, &Deterministic::new(42)).expect("a valid vine");
```

`fit` on a vine learns its structure and pair families from pseudo-observations (Dißmann et al. 2013); `multivariate::fit::fit_vine` exposes the candidate families and the criterion.

## Part of stochastic-rs

This crate is one of the sub-crates of
[**stochastic-rs**](https://github.com/rust-dd/stochastic-rs). Most users
should depend on the umbrella crate, which re-exports everything:

```toml
[dependencies]
stochastic-rs = "3.0.0-rc.4"
```

Depend on `stochastic-rs-copulas` directly only when you want this slice and nothing else.

- Documentation: [stochastic.rust-dd.com](https://stochastic.rust-dd.com)
- Tutorials: [stochastic.rust-dd.com/docs/tutorials](https://stochastic.rust-dd.com/docs/tutorials)
- API reference: [docs.rs/stochastic-rs-copulas](https://docs.rs/stochastic-rs-copulas)

## License

MIT
