//! # Empirical
//!
//! $$
//! C_n(u,v)=\frac{1}{n}\sum_{i=1}^n \mathbf 1\{U_i\le u,\,V_i\le v\}
//! $$
//! on the pseudo-observations `U_i = rank_i/(n+1)`, a tie group taking the mean of its ranks: asymptotically the empirical
//! copula of Deheuvels (1979).
//!
//! Reference: Genest, C., Rémillard, B. & Beaudoin, D. (2009), "Goodness-of-fit tests for copulas: A review and a power study", Insurance: Mathematics and Economics 44(2), 199–213, doi:10.1016/j.insmatheco.2007.10.005 (§2.2: `U_ij = R_ij/(n+1)`; §3.1, Eq. (1): `C_n`).
//! Reference: Deheuvels, P. (1979), "La fonction de dépendance empirique et ses propriétés. Un test non paramétrique d'indépendance", Acad. Roy. Belg. Bull. Cl. Sci. (5) 65, 274–292.

use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Axis;
use ndarray::stack;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::uniform::SimdUniform;

use crate::gof::pseudo_observations;

/// Empirical copula of two series, on their pseudo-observations.
#[derive(Clone, Debug)]
pub struct EmpiricalCopula2D {
  pseudo: Array2<f64>,
}

impl EmpiricalCopula2D {
  /// Panics on a length mismatch, an empty series or a NaN entry.
  pub fn new(x: ArrayView1<f64>, y: ArrayView1<f64>) -> Self {
    assert_eq!(
      x.len(),
      y.len(),
      "x and y must satisfy `x.len() == y.len()`, got x.len() = {}, y.len() = {}",
      x.len(),
      y.len()
    );
    assert!(
      !x.is_empty(),
      "x must satisfy `!x.is_empty()`, got x.len() = 0"
    );
    assert!(
      !x.iter().chain(y.iter()).any(|v| v.is_nan()),
      "x and y must satisfy `no entry is NaN`, got a NaN"
    );
    Self {
      pseudo: pseudo_observations(stack![Axis(1), x, y].view()),
    }
  }

  /// The pseudo-observations `(n, 2)`, `rank/(n+1)` per column.
  pub fn rank_data(&self) -> ArrayView2<'_, f64> {
    self.pseudo.view()
  }

  /// `C_n(u, v) = n^{-1} Σ 1{U_i ≤ u, V_i ≤ v}` at every row of `x`, NaN for a NaN coordinate; panics unless `x` has
  /// two columns.
  pub fn cdf(&self, x: ArrayView2<f64>) -> Array1<f64> {
    assert_eq!(
      x.ncols(),
      2,
      "x must satisfy `x.ncols() == 2`, got x.ncols() = {}",
      x.ncols()
    );
    let n = self.pseudo.nrows() as f64;
    x.rows()
      .into_iter()
      .map(|q| {
        if q[0].is_nan() || q[1].is_nan() {
          return f64::NAN;
        }
        let below = self
          .pseudo
          .rows()
          .into_iter()
          .filter(|p| p[0] <= q[0] && p[1] <= q[1])
          .count();
        below as f64 / n
      })
      .collect()
  }

  /// Bootstrap of `n` observed pairs with replacement, drawn from `seed`.
  pub fn sample<S: SeedExt>(&self, n: usize, seed: &S) -> Array2<f64> {
    self.sample_with_uniform(SimdUniform::<f64>::new(0.0, 1.0).seeded(seed), n)
  }

  fn sample_with_uniform(&self, mut ud: Seeded<SimdUniform<f64>>, n: usize) -> Array2<f64> {
    let n_rows = self.pseudo.nrows();
    let mut out = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
      let idx = ((ud.sample() * n_rows as f64) as usize).min(n_rows - 1);
      out[[i, 0]] = self.pseudo[[idx, 0]];
      out[[i, 1]] = self.pseudo[[idx, 1]];
    }
    out
  }
}

#[cfg(test)]
mod tests {
  use ndarray::array;
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::*;

  /// Pseudo-observations `k/(n+1)` (`gof::pseudo_observations`), the empirical copula on them, and a seeded bootstrap.
  #[test]
  fn empirical_copula_ranks_cdf_and_bootstrap() {
    let x = Array1::from_vec(vec![3.0, 1.0, 2.0, 4.0]);
    let y = Array1::from_vec(vec![10.0, 30.0, 20.0, 40.0]);
    let ec = EmpiricalCopula2D::new(x.view(), y.view());
    assert_eq!(ec.rank_data().row(0).to_vec(), vec![0.6, 0.2]);
    assert_eq!(
      ec.rank_data(),
      pseudo_observations(stack![Axis(1), x, y].view()).view()
    );
    let c = ec.cdf(array![[0.5, 0.5], [1.0, 1.0], [0.1, 0.9]].view());
    assert_eq!(c.to_vec(), vec![0.25, 1.0, 0.0]);
    let a = ec.sample(17, &Deterministic::new(42));
    assert_eq!(a.dim(), (17, 2));
    assert_eq!(a, ec.sample(17, &Deterministic::new(42)));
    assert_ne!(a, ec.sample(17, &Deterministic::new(7)));
    assert!(a.iter().all(|v| (0.2..=0.8).contains(v)));
    assert_eq!(ec.sample(5, &Unseeded).nrows(), 5);
  }

  /// The cdf is a step function of the pseudo-observations, defined off the unit square too; a NaN coordinate has no value.
  #[test]
  fn the_cdf_is_defined_off_the_grid_and_nan_for_nan() {
    let ec = EmpiricalCopula2D::new(
      array![3.0, 1.0, 2.0, 4.0].view(),
      array![10.0, 30.0, 20.0, 40.0].view(),
    );
    let c = ec.cdf(
      array![
        [0.4, 0.4],
        [-1.0, 5.0],
        [9.0, 9.0],
        [f64::NAN, 0.5],
        [0.5, f64::NAN]
      ]
      .view(),
    );
    assert_eq!(c.to_vec()[..3], [0.25, 0.0, 1.0]);
    assert!(c[3].is_nan() && c[4].is_nan());
  }

  /// The empirical copula is a function of the data set: the same tied pairs in two row orders give the same cdf.
  #[test]
  fn the_cdf_does_not_depend_on_the_row_order_of_tied_data() {
    let x = array![1.0, 1.0, 2.0, 3.0];
    let a = EmpiricalCopula2D::new(x.view(), array![1.0, 2.0, 2.0, 3.0].view());
    let b = EmpiricalCopula2D::new(x.view(), array![2.0, 1.0, 2.0, 3.0].view());
    let q = array![[0.25, 0.25], [0.3, 0.2], [0.3, 0.5], [0.6, 0.5], [1.0, 1.0]];
    assert_eq!(a.cdf(q.view()), b.cdf(q.view()));
    assert_eq!(a.cdf(q.view()).to_vec(), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
  }

  #[test]
  #[should_panic(
    expected = "x and y must satisfy `x.len() == y.len()`, got x.len() = 2, y.len() = 1"
  )]
  fn mismatched_series_are_a_precondition() {
    let _ = EmpiricalCopula2D::new(array![1.0, 2.0].view(), array![1.0].view());
  }

  #[test]
  #[should_panic(expected = "x must satisfy `!x.is_empty()`, got x.len() = 0")]
  fn an_empty_series_is_a_precondition() {
    let _ = EmpiricalCopula2D::new(
      Array1::<f64>::zeros(0).view(),
      Array1::<f64>::zeros(0).view(),
    );
  }

  #[test]
  #[should_panic(expected = "x and y must satisfy `no entry is NaN`, got a NaN")]
  fn a_nan_entry_is_a_precondition() {
    let _ = EmpiricalCopula2D::new(
      array![1.0, f64::NAN, 3.0].view(),
      array![1.0, 2.0, 3.0].view(),
    );
  }

  #[test]
  #[should_panic(expected = "x must satisfy `x.ncols() == 2`, got x.ncols() = 1")]
  fn the_cdf_takes_pairs() {
    let ec = EmpiricalCopula2D::new(array![1.0, 2.0].view(), array![2.0, 1.0].view());
    let _ = ec.cdf(array![[0.5], [0.7]].view());
  }
}
