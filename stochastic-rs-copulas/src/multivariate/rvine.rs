//! # Regular vine (R-vine) — D-vine / C-vine wrapper
//!
//! The regular vine is a graph-theoretic generalisation of D-vines and
//! C-vines (Bedford-Cooke 2002, Joe 2011, Dißmann 2013): an R-vine is a
//! nested set of trees $T_1, \dots, T_{d-1}$ on $d$ variables satisfying
//! the **proximity condition** (two edges of $T_{k+1}$ must share a node
//! of $T_k$).
//!
//! Generic free-form R-vine evaluation requires either the Joe-Kurowicka
//! 2011 lower-triangular **structure matrix** representation or an
//! explicit edge-list with the proximity-condition recursion that walks
//! the conditional CDFs up the tree. That, together with the **Dißmann
//! 2013** sequential structure-selection algorithm, is not yet
//! implemented.
//!
//! [`RVine`] is an enum that wraps either a [`super::dvine::DVine`] or a
//! [`super::cvine::CVine`] — the two structure-special cases of the
//! regular vine. All `MultivariateExt` methods delegate to the wrapped
//! type, giving a unified API; constructor methods exist for the two
//! supported special cases.
//!
//! For general R-vines, pre-route through `DVine` (path-shaped tree) or
//! `CVine` (star tree per tree-level) — both are special cases of the
//! R-vine.
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

use std::error::Error;

use ndarray::Array1;
use ndarray::Array2;

use super::CopulaType;
use super::cvine::CVine;
use super::dvine::DVine;
use crate::traits::MultivariateExt;

/// Regular vine pair-copula construction wrapper over the two
/// structure-special cases (D-vine and C-vine). The matrix-encoded
/// generic R-vine variant (Dißmann 2013 sequential selection + density
/// propagation per Joe-Kurowicka) is not yet implemented.
#[derive(Debug, Clone)]
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

  /// All-Independence R-vine of the given dimension (D-vine backed; the
  /// independence copula is structure-invariant so the choice between
  /// D / C wrapping is purely conventional).
  pub fn independence(dim: usize) -> Result<Self, Box<dyn Error>> {
    DVine::independence(dim).map(RVine::D)
  }

  /// Dimensionality (number of marginals).
  pub fn dim(&self) -> usize {
    match self {
      RVine::D(d) => d.dim(),
      RVine::C(c) => c.dim(),
    }
  }

  /// Tree shape descriptor — useful for downstream consumers needing to
  /// branch on the underlying structure (e.g. routing to a structure-aware
  /// likelihood pipeline).
  pub fn kind(&self) -> &'static str {
    match self {
      RVine::D(_) => "DVine",
      RVine::C(_) => "CVine",
    }
  }
}

impl MultivariateExt for RVine {
  fn r#type(&self) -> CopulaType {
    CopulaType::RVine
  }

  fn sample(&self, n: usize) -> Result<Array2<f64>, Box<dyn Error>> {
    match self {
      RVine::D(d) => d.sample(n),
      RVine::C(c) => c.sample(n),
    }
  }

  fn fit(&mut self, x: Array2<f64>) -> Result<(), Box<dyn Error>> {
    match self {
      RVine::D(d) => d.fit(x),
      RVine::C(c) => c.fit(x),
    }
  }

  fn check_fit(&self, x: &Array2<f64>) -> Result<(), Box<dyn Error>> {
    match self {
      RVine::D(d) => d.check_fit(x),
      RVine::C(c) => c.check_fit(x),
    }
  }

  fn pdf(&self, x: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    match self {
      RVine::D(d) => d.pdf(x),
      RVine::C(c) => c.pdf(x),
    }
  }

  fn log_pdf(&self, x: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    match self {
      RVine::D(d) => d.log_pdf(x),
      RVine::C(c) => c.log_pdf(x),
    }
  }

  fn cdf(&self, x: Array2<f64>) -> Result<Array1<f64>, Box<dyn Error>> {
    match self {
      RVine::D(d) => d.cdf(x),
      RVine::C(c) => c.cdf(x),
    }
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;

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
    let lp_dv = dv.log_pdf(q.clone()).unwrap();
    let lp_rv = rv.log_pdf(q).unwrap();
    for i in 0..lp_dv.len() {
      assert!(
        (lp_dv[i] - lp_rv[i]).abs() < 1e-12,
        "DVine wrapper diverges from underlying DVine at row {i}: {} vs {}",
        lp_dv[i],
        lp_rv[i]
      );
    }
    assert_eq!(rv.kind(), "DVine");
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
    let lp_cv = cv.log_pdf(q.clone()).unwrap();
    let lp_rv = rv.log_pdf(q).unwrap();
    for i in 0..lp_cv.len() {
      assert!(
        (lp_cv[i] - lp_rv[i]).abs() < 1e-12,
        "CVine wrapper diverges from underlying CVine at row {i}: {} vs {}",
        lp_cv[i],
        lp_rv[i]
      );
    }
    assert_eq!(rv.kind(), "CVine");
  }

  /// Sampling delegates to the wrapped vine and preserves uniform
  /// marginals for the independence case.
  #[test]
  fn rvine_independence_sampling_uniform_marginals() {
    let rv = RVine::independence(3).unwrap();
    let s = rv.sample(10_000).unwrap();
    for j in 0..3 {
      let col = s.column(j);
      let m: f64 = col.iter().sum::<f64>() / col.len() as f64;
      assert!(
        (m - 0.5).abs() < 0.02,
        "marginal {j} mean = {m}, expected ~0.5"
      );
    }
    assert_eq!(rv.kind(), "DVine"); // independence() defaults to DVine backing
  }
}
