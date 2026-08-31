//! # Multivariate
//!
//! $$
//! F_{X_1,\dots,X_d}(x)=C\left(F_1(x_1),\dots,F_d(x_d)\right)
//! $$
//!
pub use crate::traits::MultivariateExt;

pub mod cvine;
pub mod dvine;
pub mod gaussian;
pub(crate) mod linalg;
pub mod nac;
pub mod rvine;
pub mod t;
pub mod tree;
pub mod vine;

pub enum CopulaType {
  CVine,
  DVine,
  Gaussian,
  NestedArchimedean,
  RVine,
  TMultivariate,
  Tree,
  Vine,
}

#[cfg(test)]
mod tests {
  use ndarray::array;

  use crate::multivariate::cvine::CVine;
  use crate::multivariate::dvine::DVine;
  use crate::multivariate::gaussian::GaussianMultivariate;
  use crate::multivariate::nac::NacFamily;
  use crate::multivariate::nac::NacNode;
  use crate::multivariate::nac::NestedArchimedean;
  use crate::multivariate::rvine::RVine;
  use crate::multivariate::t::TMultivariate;
  use crate::multivariate::tree::TreeMultivariate;
  use crate::multivariate::vine::VineMultivariate;
  use crate::traits::MultivariateExt;

  /// Every `MultivariateExt` implementor must expose a `sample_with_seed`
  /// that reproduces the same matrix for the same seed. Before this fix,
  /// `GaussianMultivariate`/`TMultivariate`/`VineMultivariate`/
  /// `TreeMultivariate` had no seeded path at all (hardcoded `Unseeded`),
  /// and `CVine`/`DVine`/`NestedArchimedean` exposed it only under an
  /// inconsistent inherent `sample_seeded` name outside the trait.
  #[test]
  fn every_multivariate_sampler_is_seedable_and_deterministic() {
    let corr = array![[1.0, 0.3], [0.3, 1.0]];

    let gaussian = GaussianMultivariate::new_with_corr(corr.clone()).unwrap();
    assert_eq!(
      gaussian.sample_with_seed(64, 42).unwrap(),
      gaussian.sample_with_seed(64, 42).unwrap(),
      "GaussianMultivariate"
    );

    let t = TMultivariate::new_with(corr.clone(), 5.0).unwrap();
    assert_eq!(
      t.sample_with_seed(64, 42).unwrap(),
      t.sample_with_seed(64, 42).unwrap(),
      "TMultivariate"
    );

    let vine = VineMultivariate::new_with_corr(corr.clone()).unwrap();
    assert_eq!(
      vine.sample_with_seed(64, 42).unwrap(),
      vine.sample_with_seed(64, 42).unwrap(),
      "VineMultivariate"
    );

    let tree = TreeMultivariate::new_with_corr(corr).unwrap();
    assert_eq!(
      tree.sample_with_seed(64, 42).unwrap(),
      tree.sample_with_seed(64, 42).unwrap(),
      "TreeMultivariate"
    );

    let cvine = CVine::independence(3).unwrap();
    assert_eq!(
      cvine.sample_with_seed(64, 42).unwrap(),
      cvine.sample_with_seed(64, 42).unwrap(),
      "CVine"
    );

    let dvine = DVine::independence(3).unwrap();
    assert_eq!(
      dvine.sample_with_seed(64, 42).unwrap(),
      dvine.sample_with_seed(64, 42).unwrap(),
      "DVine"
    );

    // Clayton family specifically exercises the root-frailty draw, which
    // used to be hardcoded to `Unseeded` regardless of the caller's seed.
    let root = NacNode::leaf_group(2.0, vec![0, 1, 2]);
    let nac = NestedArchimedean::new(NacFamily::Clayton, root, 3).unwrap();
    assert_eq!(
      nac.sample_with_seed(64, 42).unwrap(),
      nac.sample_with_seed(64, 42).unwrap(),
      "NestedArchimedean"
    );

    let rvine = RVine::independence(3).unwrap();
    assert_eq!(
      rvine.sample_with_seed(64, 42).unwrap(),
      rvine.sample_with_seed(64, 42).unwrap(),
      "RVine"
    );
  }
}
