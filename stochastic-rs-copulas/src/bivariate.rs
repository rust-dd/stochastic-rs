//! # Bivariate
//!
//! $$
//! F_{X_1,\dots,X_d}(x)=C\left(F_1(x_1),\dots,F_d(x_d)\right)
//! $$
//!
pub use crate::traits::BivariateExt;

pub mod amh;
pub mod clayton;
pub mod fgm;
pub mod frank;
pub mod galambos;
pub mod gaussian;
pub mod gumbel;
pub mod husler_reiss;
pub mod independence;
pub mod joe;
pub mod marshall_olkin;
pub mod plackett;
pub mod t_copula;

#[derive(Debug, Clone, Copy)]
pub enum CopulaType {
  Amh,
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

#[cfg(test)]
mod tests {
  use crate::bivariate::clayton::Clayton;
  use crate::bivariate::frank::Frank;
  use crate::traits::BivariateExt;

  /// `BivariateExt::sample_with_seed` is a trait-default method — no
  /// bivariate family overrides it — so pin its determinism contract
  /// directly here, mirroring
  /// `multivariate::tests::every_multivariate_sampler_is_seedable_and_deterministic`.
  /// Same seed on the same object must replay bit-for-bit, and two
  /// independently-constructed, identically-seeded objects must agree.
  #[test]
  fn every_bivariate_sampler_is_seedable_and_deterministic() {
    let mut clayton = Clayton::new();
    clayton.set_tau(0.5);
    clayton._compute_theta();
    assert_eq!(
      clayton.sample_with_seed(64, 42).unwrap(),
      clayton.sample_with_seed(64, 42).unwrap(),
      "Clayton replay"
    );
    let mut clayton_2 = Clayton::new();
    clayton_2.set_tau(0.5);
    clayton_2._compute_theta();
    assert_eq!(
      clayton.sample_with_seed(64, 42).unwrap(),
      clayton_2.sample_with_seed(64, 42).unwrap(),
      "Clayton cross-object"
    );

    let mut frank = Frank::new(None, Some(0.5));
    frank._compute_theta();
    assert_eq!(
      frank.sample_with_seed(64, 42).unwrap(),
      frank.sample_with_seed(64, 42).unwrap(),
      "Frank replay"
    );
    let mut frank_2 = Frank::new(None, Some(0.5));
    frank_2._compute_theta();
    assert_eq!(
      frank.sample_with_seed(64, 42).unwrap(),
      frank_2.sample_with_seed(64, 42).unwrap(),
      "Frank cross-object"
    );
  }
}
