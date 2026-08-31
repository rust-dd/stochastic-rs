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
  (linear algebra via the pure-Rust `faer`).
- **Empirical** — pseudo-observations and the empirical copula.
- **Process coupling** — drive two stochastic processes through a copula.

Every multivariate sampler has a `sample_with_seed(n, seed)` counterpart
when the draw needs to be reproducible, matching the bivariate one below.

## Usage

```rust
use stochastic_rs_copulas::bivariate::clayton::Clayton;
use stochastic_rs_copulas::traits::BivariateExt;

let mut c = Clayton {
    theta: Some(2.0),
    ..Clayton::new()
};
let u = c.sample(10_000)?;               // Array2<f64>, shape (10000, 2)
let v = c.sample_with_seed(10_000, 42)?; // reproducible
```

Multivariate constructions take a tree of `PairCopula` variants:

```rust
use stochastic_rs_copulas::multivariate::cvine::CVine;
use stochastic_rs_copulas::multivariate::dvine::PairCopula;
use stochastic_rs_copulas::traits::MultivariateExt;

let cv = CVine::new(2, vec![vec![PairCopula::Clayton { theta: 2.0 }]])?;
let draws = cv.sample_with_seed(10_000, 42)?;
```

## Part of stochastic-rs

This crate is one of the sub-crates of
[**stochastic-rs**](https://github.com/rust-dd/stochastic-rs). Most users
should depend on the umbrella crate, which re-exports everything:

```toml
[dependencies]
stochastic-rs = "3.0.0-beta.2"
```

Depend on `stochastic-rs-copulas` directly only when you want this slice and nothing else.

- Documentation: [stochastic.rust-dd.com](https://stochastic.rust-dd.com)
- API reference: [docs.rs/stochastic-rs-copulas](https://docs.rs/stochastic-rs-copulas)

## License

MIT
