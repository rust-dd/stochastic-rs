//! # Regular vine (R-vine)
//!
//! A regular vine is trees $T_1,\dots,T_{d-1}$ on $d$ variables, edges of $T_{k+1}$ sharing a node
//! of $T_k$; [`RVine`] covers the path ([`DVine`]) and star ([`CVine`]) cases only.
//!
//! References:
//! - Bedford, T., Cooke, R.M. (2002), "Vines — a new graphical model for
//!   dependent random variables", *Annals of Statistics* 30, 1031-1068.
//! - Joe, H., Kurowicka, D. (2011), "Dependence Modeling: Vine Copula
//!   Handbook", World Scientific.
//! - Dißmann, J., Brechmann, E.C., Czado, C., Kurowicka, D. (2013),
//!   "Selecting and estimating regular vine copulae and application to
//!   financial returns", *Computational Statistics & Data Analysis* 59,
//!   52-69.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView2;
use stochastic_rs_core::simd_rng::SeedExt;

use super::MultivariateFamily;
use super::cvine::CVine;
use super::dvine::DVine;
use super::fit::VineStructure;
use crate::error::CopulaError;
use crate::traits::MultivariateExt;

/// A D- or C-vine behind one type; every `MultivariateExt` method delegates to the wrapped vine.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RVine {
  /// Path-shaped trees — see [`super::dvine::DVine`].
  D(DVine),
  /// Star-shaped trees with a root variable per tree — see
  /// [`super::cvine::CVine`].
  C(CVine),
}

impl RVine {
  /// Construct an R-vine from a [`DVine`] (path-shaped tree structure).
  pub fn from_dvine(dvine: DVine) -> Self {
    RVine::D(dvine)
  }

  /// Construct an R-vine from a [`CVine`] (star-shaped tree per level).
  pub fn from_cvine(cvine: CVine) -> Self {
    RVine::C(cvine)
  }

  /// All-independence vine of `dim` marginals, D-backed: independence is structure-invariant, so
  /// D or C is a convention.
  pub fn independence(dim: usize) -> Result<Self, CopulaError> {
    DVine::independence(dim).map(RVine::D)
  }

  /// Dimensionality (number of marginals).
  pub fn dim(&self) -> usize {
    match self {
      RVine::D(d) => d.dim(),
      RVine::C(c) => c.dim(),
    }
  }

  /// Which of the two tree shapes the vine has.
  pub fn structure(&self) -> VineStructure {
    match self {
      RVine::D(_) => VineStructure::DVine,
      RVine::C(_) => VineStructure::CVine,
    }
  }

  /// Data column of each vine variable.
  pub fn order(&self) -> &[usize] {
    match self {
      RVine::D(d) => d.order(),
      RVine::C(c) => c.order(),
    }
  }
}

impl MultivariateExt for RVine {
  fn family(&self) -> MultivariateFamily {
    MultivariateFamily::RVine
  }

  fn sample<S: SeedExt>(&self, n: usize, seed: &S) -> Result<Array2<f64>, CopulaError> {
    match self {
      RVine::D(d) => d.sample(n, seed),
      RVine::C(c) => c.sample(n, seed),
    }
  }

  fn fit(&mut self, x: ArrayView2<f64>) -> Result<(), CopulaError> {
    match self {
      RVine::D(d) => d.fit(x),
      RVine::C(c) => c.fit(x),
    }
  }

  fn check_fit(&self, x: ArrayView2<f64>) -> Result<(), CopulaError> {
    match self {
      RVine::D(d) => d.check_fit(x),
      RVine::C(c) => c.check_fit(x),
    }
  }

  fn pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    match self {
      RVine::D(d) => d.pdf(x),
      RVine::C(c) => c.pdf(x),
    }
  }

  fn log_pdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    match self {
      RVine::D(d) => d.log_pdf(x),
      RVine::C(c) => c.log_pdf(x),
    }
  }

  fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    match self {
      RVine::D(d) => d.cdf(x),
      RVine::C(c) => c.cdf(x),
    }
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;

  use super::*;
  use crate::multivariate::dvine::PairCopula;

  /// An R-vine wrapping a DVine must produce identical density to the
  /// underlying DVine on the same inputs.
  #[test]
  fn rvine_dvine_delegates_density() {
    let tree = vec![
      vec![
        PairCopula::Gaussian { rho: 0.4 },
        PairCopula::Gaussian { rho: 0.3 },
      ],
      vec![PairCopula::Clayton { theta: 1.5 }],
    ];
    let dv = DVine::new(3, tree).unwrap();
    let rv = RVine::from_dvine(dv.clone());
    let q = array![[0.3, 0.5, 0.7], [0.1, 0.5, 0.9]];
    let lp_dv = dv.log_pdf(q.view()).unwrap();
    let lp_rv = rv.log_pdf(q.view()).unwrap();
    for i in 0..lp_dv.len() {
      assert!(
        (lp_dv[i] - lp_rv[i]).abs() < 1e-12,
        "DVine wrapper diverges from underlying DVine at row {i}: {} vs {}",
        lp_dv[i],
        lp_rv[i]
      );
    }
    assert_eq!(rv.structure(), VineStructure::DVine);
    assert_eq!(rv.dim(), 3);
  }

  /// Same property for the C-vine wrapper.
  #[test]
  fn rvine_cvine_delegates_density() {
    let tree = vec![
      vec![
        PairCopula::Gaussian { rho: 0.5 },
        PairCopula::Gaussian { rho: 0.5 },
      ],
      vec![PairCopula::Independence],
    ];
    let cv = CVine::new(3, tree).unwrap();
    let rv = RVine::from_cvine(cv.clone());
    let q = array![[0.2, 0.4, 0.6], [0.4, 0.6, 0.8]];
    let lp_cv = cv.log_pdf(q.view()).unwrap();
    let lp_rv = rv.log_pdf(q.view()).unwrap();
    for i in 0..lp_cv.len() {
      assert!(
        (lp_cv[i] - lp_rv[i]).abs() < 1e-12,
        "CVine wrapper diverges from underlying CVine at row {i}: {} vs {}",
        lp_cv[i],
        lp_rv[i]
      );
    }
    assert_eq!(rv.structure(), VineStructure::CVine);
  }

  #[test]
  fn rvine_fit_keeps_its_structure() {
    let u = DVine::new(
      3,
      vec![
        vec![PairCopula::Gaussian { rho: 0.5 }; 2],
        vec![PairCopula::Independence],
      ],
    )
    .unwrap()
    .sample(2000, &Deterministic::new(3))
    .unwrap();
    let mut d = RVine::independence(3).unwrap();
    d.fit(u.view()).unwrap();
    assert_eq!(d.structure(), VineStructure::DVine);
    let mut c = RVine::from_cvine(CVine::independence(3).unwrap());
    c.fit(u.view()).unwrap();
    assert_eq!(c.structure(), VineStructure::CVine);
    assert_eq!(c.order().len(), 3);
  }

  /// Sampling delegates to the wrapped vine and preserves uniform
  /// marginals for the independence case.
  #[test]
  fn rvine_independence_sampling_uniform_marginals() {
    let rv = RVine::independence(3).unwrap();
    let s = rv.sample(10_000, &Deterministic::new(7)).unwrap();
    for j in 0..3 {
      let col = s.column(j);
      let m: f64 = col.iter().sum::<f64>() / col.len() as f64;
      assert!(
        (m - 0.5).abs() < 0.02,
        "marginal {j} mean = {m}, expected ~0.5"
      );
    }
    assert_eq!(rv.structure(), VineStructure::DVine);
  }
}
