//! # Multivariate
//!
//! $$
//! F_{X_1,\dots,X_d}(x)=C\left(F_1(x_1),\dots,F_d(x_d)\right)
//! $$
//!
pub use crate::traits::MultivariateExt;

pub(crate) mod correlation_matrix;
pub mod cvine;
pub mod dvine;
pub mod fit;
pub mod gaussian;
pub(crate) mod genz;
pub(crate) mod linalg;
pub mod nac;
pub(crate) mod qmc;
mod rosenblatt;
pub mod rvine;
pub mod t;
mod vine_order;

/// `values` with NaN on every row of `x` that has a coordinate at 0 or 1, where a copula has no density.
pub(crate) fn on_open_cube(
  x: ndarray::ArrayView2<f64>,
  mut values: ndarray::Array1<f64>,
) -> ndarray::Array1<f64> {
  for (value, row) in values.iter_mut().zip(x.rows()) {
    if row.iter().any(|&u| u == 0.0 || u == 1.0) {
      *value = f64::NAN;
    }
  }
  values
}

/// One variant per multivariate copula of the crate; `MultivariateExt::family` returns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MultivariateFamily {
  CVine,
  DVine,
  Gaussian,
  NestedArchimedean,
  RVine,
  TMultivariate,
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;

  use crate::bivariate::BivariateFamily;
  use crate::multivariate::cvine::CVine;
  use crate::multivariate::dvine::DVine;
  use crate::multivariate::gaussian::GaussianMultivariate;
  use crate::multivariate::nac::NacNode;
  use crate::multivariate::nac::NestedArchimedean;
  use crate::multivariate::rvine::RVine;
  use crate::multivariate::t::TMultivariate;
  use crate::traits::MultivariateExt;

  /// Every `MultivariateExt` implementor replays the same matrix for the same `Deterministic` seed.
  #[test]
  fn every_multivariate_sampler_is_seedable_and_deterministic() {
    let corr = array![[1.0, 0.3], [0.3, 1.0]];

    let gaussian = GaussianMultivariate::new(corr.clone());
    assert_eq!(
      gaussian.sample(64, &Deterministic::new(42)).unwrap(),
      gaussian.sample(64, &Deterministic::new(42)).unwrap(),
      "GaussianMultivariate"
    );

    let t = TMultivariate::new(corr, 5.0);
    assert_eq!(
      t.sample(64, &Deterministic::new(42)).unwrap(),
      t.sample(64, &Deterministic::new(42)).unwrap(),
      "TMultivariate"
    );

    let cvine = CVine::independence(3).unwrap();
    assert_eq!(
      cvine.sample(64, &Deterministic::new(42)).unwrap(),
      cvine.sample(64, &Deterministic::new(42)).unwrap(),
      "CVine"
    );

    let dvine = DVine::independence(3).unwrap();
    assert_eq!(
      dvine.sample(64, &Deterministic::new(42)).unwrap(),
      dvine.sample(64, &Deterministic::new(42)).unwrap(),
      "DVine"
    );

    // The Clayton tree exercises the root-frailty draw, whose sub-seed comes off the caller's stream.
    let root = NacNode::leaf_group(2.0, vec![0, 1, 2]);
    let nac = NestedArchimedean::new(BivariateFamily::Clayton, root, 3).unwrap();
    assert_eq!(
      nac.sample(64, &Deterministic::new(42)).unwrap(),
      nac.sample(64, &Deterministic::new(42)).unwrap(),
      "NestedArchimedean"
    );

    let rvine = RVine::independence(3).unwrap();
    assert_eq!(
      rvine.sample(64, &Deterministic::new(42)).unwrap(),
      rvine.sample(64, &Deterministic::new(42)).unwrap(),
      "RVine"
    );
  }
}

#[cfg(test)]
mod catalog_tests {
  use super::MultivariateFamily;

  /// Guards `copulas.mdx`'s catalog table: the match has no wildcard arm on purpose, so a new variant is a compile
  /// error here until its arm, its docs row and the count are added.
  #[test]
  fn multivariate_catalog_matches_docs_count() {
    let count = [
      MultivariateFamily::CVine,
      MultivariateFamily::DVine,
      MultivariateFamily::Gaussian,
      MultivariateFamily::NestedArchimedean,
      MultivariateFamily::RVine,
      MultivariateFamily::TMultivariate,
    ]
    .into_iter()
    .map(|t| match t {
      MultivariateFamily::CVine
      | MultivariateFamily::DVine
      | MultivariateFamily::Gaussian
      | MultivariateFamily::NestedArchimedean
      | MultivariateFamily::RVine
      | MultivariateFamily::TMultivariate => 1,
    })
    .sum::<usize>();

    assert_eq!(count, 6, "copulas.mdx's multivariate table says 6");
  }
}
