//! # Bivariate
//!
//! $$
//! F_{X_1,\dots,X_d}(x)=C\left(F_1(x_1),\dots,F_d(x_d)\right)
//! $$
//!
//! Fifteen families behind [`BivariateExt`]; in every one `partial_derivative` is
//! $\partial_v C$, the conditional cdf of `U` given `V = v`, and `percent_point` inverts it in `u`.
//! Choosing a family: <https://stochastic.rust-dd.com/docs/copulas#choosing-a-copula>

pub use crate::traits::BivariateExt;

pub mod amh;
pub mod bb1;
pub mod bb7;
pub mod clayton;
pub(crate) mod conditional;
pub mod fgm;
pub mod frank;
pub mod galambos;
pub mod gaussian;
pub mod gumbel;
pub mod husler_reiss;
pub mod independence;
pub(crate) mod integral;
pub mod joe;
pub mod marshall_olkin;
pub mod plackett;
pub mod t_copula;
pub(crate) mod two_parameter;

/// One variant per bivariate copula of the crate; `BivariateExt::family` returns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BivariateFamily {
  Amh,
  Bb1,
  Bb7,
  Clayton,
  Fgm,
  Frank,
  Galambos,
  Gaussian,
  Gumbel,
  HuslerReiss,
  Independence,
  Joe,
  MarshallOlkin,
  Plackett,
  TCopula,
}

impl BivariateFamily {
  /// Every family, in the enum's order.
  pub const ALL: &[Self] = &[
    Self::Amh,
    Self::Bb1,
    Self::Bb7,
    Self::Clayton,
    Self::Fgm,
    Self::Frank,
    Self::Galambos,
    Self::Gaussian,
    Self::Gumbel,
    Self::HuslerReiss,
    Self::Independence,
    Self::Joe,
    Self::MarshallOlkin,
    Self::Plackett,
    Self::TCopula,
  ];

  /// The families the vine fitter can estimate, in the order it tries them.
  pub const VINE_CANDIDATES: &[Self] = &[
    Self::Independence,
    Self::Gaussian,
    Self::TCopula,
    Self::Clayton,
    Self::Frank,
    Self::Bb1,
    Self::Bb7,
  ];

  /// Free parameters of the family; Marshall-Olkin counts its shock rates (α, β), although its default fit path
  /// estimates one symmetric θ.
  pub fn parameter_count(self) -> usize {
    match self {
      Self::Independence => 0,
      Self::Amh
      | Self::Clayton
      | Self::Fgm
      | Self::Frank
      | Self::Galambos
      | Self::Gaussian
      | Self::Gumbel
      | Self::HuslerReiss
      | Self::Joe
      | Self::Plackett => 1,
      Self::Bb1 | Self::Bb7 | Self::MarshallOlkin | Self::TCopula => 2,
    }
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

  use crate::bivariate::clayton::Clayton;
  use crate::bivariate::frank::Frank;
  use crate::bivariate::gumbel::Gumbel;
  use crate::traits::BivariateExt;

  /// The trait-default `sample` replays bit-for-bit for one `Deterministic` seed, on one object and on two built alike.
  /// Frank and Gumbel sit off the independence boundary their `percent_point` shortcuts around, so the general path runs.
  #[test]
  fn every_bivariate_sampler_is_seedable_and_deterministic() {
    let clayton = Clayton::from_tau(0.5).unwrap();
    assert_eq!(
      clayton.sample(64, &Deterministic::new(42)).unwrap(),
      clayton.sample(64, &Deterministic::new(42)).unwrap(),
      "Clayton replay"
    );
    let clayton_2 = Clayton::from_tau(0.5).unwrap();
    assert_eq!(
      clayton.sample(64, &Deterministic::new(42)).unwrap(),
      clayton_2.sample(64, &Deterministic::new(42)).unwrap(),
      "Clayton cross-object"
    );

    let frank = Frank::from_tau(0.5).unwrap();
    assert_eq!(
      frank.sample(64, &Deterministic::new(42)).unwrap(),
      frank.sample(64, &Deterministic::new(42)).unwrap(),
      "Frank replay"
    );
    let frank_2 = Frank::from_tau(0.5).unwrap();
    assert_eq!(
      frank.sample(64, &Deterministic::new(42)).unwrap(),
      frank_2.sample(64, &Deterministic::new(42)).unwrap(),
      "Frank cross-object"
    );

    // theta = 1 / (1 - 0.5) = 2.0 — well past the theta == 1.0 independence
    // boundary that `Gumbel::percent_point` shortcuts around.
    let gumbel = Gumbel::from_tau(0.5).unwrap();
    assert_eq!(
      gumbel.sample(64, &Deterministic::new(42)).unwrap(),
      gumbel.sample(64, &Deterministic::new(42)).unwrap(),
      "Gumbel replay"
    );
    let gumbel_2 = Gumbel::from_tau(0.5).unwrap();
    assert_eq!(
      gumbel.sample(64, &Deterministic::new(42)).unwrap(),
      gumbel_2.sample(64, &Deterministic::new(42)).unwrap(),
      "Gumbel cross-object"
    );
  }
}

#[cfg(test)]
mod catalog_tests {
  use std::collections::HashSet;

  use super::BivariateFamily;

  /// Guards `copulas.mdx`'s catalog table and `CLAUDE.md`'s count: the match has no wildcard arm on purpose, so a new
  /// variant is a compile error here until its arm, its docs row and the counts are added.
  #[test]
  fn bivariate_catalog_matches_docs_count() {
    let count = BivariateFamily::ALL
      .iter()
      .map(|t| match t {
        BivariateFamily::Amh
        | BivariateFamily::Bb1
        | BivariateFamily::Bb7
        | BivariateFamily::Clayton
        | BivariateFamily::Fgm
        | BivariateFamily::Frank
        | BivariateFamily::Galambos
        | BivariateFamily::Gaussian
        | BivariateFamily::Gumbel
        | BivariateFamily::HuslerReiss
        | BivariateFamily::Independence
        | BivariateFamily::Joe
        | BivariateFamily::MarshallOlkin
        | BivariateFamily::Plackett
        | BivariateFamily::TCopula => 1,
      })
      .sum::<usize>();

    assert_eq!(count, 15, "copulas.mdx's bivariate table says 15");
    let distinct = BivariateFamily::ALL.iter().collect::<HashSet<_>>().len();
    assert_eq!(distinct, count, "ALL repeats a family");
  }
}
