//! # Cgns
//!
//! $$
//! Z_t=L\varepsilon_t,\quad \varepsilon_t\sim\mathcal N(0,I),\ LL^\top=\Sigma
//! $$
//!
use ndarray::Array1;

use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

#[derive(Copy, Clone)]
pub struct CGNS<T: FloatExt> {
  /// Instantaneous correlation parameter.
  pub rho: T,
  /// Number of discrete simulation points (or samples).
  pub n: usize,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  gn: Gn<T>,
}

impl<T: FloatExt> CGNS<T> {
  pub fn new(rho: T, n: usize, t: Option<T>) -> Self {
    assert!(
      (-T::one()..=T::one()).contains(&rho),
      "Correlation coefficient must be in [-1, 1]"
    );

    Self {
      rho,
      n,
      t,
      gn: Gn::new(n, t),
    }
  }
}

impl<T: FloatExt> ProcessExt<T> for CGNS<T> {
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let gn1 = self.gn.sample();
    let z = self.gn.sample();
    let c = (T::one() - self.rho.powi(2)).sqrt();
    let mut gn2 = Array1::zeros(self.n);

    for i in 0..self.n {
      gn2[i] = self.rho * gn1[i] + c * z[i];
    }

    [gn1, gn2]
  }
}

impl<T: FloatExt> CGNS<T> {
  pub fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize(self.n).unwrap()
  }
}

py_process_2x1d!(PyCGNS, CGNS,
  sig: (rho, n, t=None, dtype=None),
  params: (rho: f64, n: usize, t: Option<f64>)
);