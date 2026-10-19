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

use crate::bivariate::conditional::check_unit;
use crate::error::CopulaError;
use crate::gof::pseudo_observations;

/// Empirical copula of two series, on their pseudo-observations.
#[derive(Clone, Debug)]
pub struct EmpiricalCopula2D {
  pseudo: Array2<f64>,
}

impl EmpiricalCopula2D {
  /// The empirical copula of `x` and `y`: `DimensionMismatch` for unequal lengths, `InsufficientData` for empty series,
  /// `MarginalOutOfRange` for a NaN entry.
  pub fn new(x: ArrayView1<f64>, y: ArrayView1<f64>) -> Result<Self, CopulaError> {
    if x.len() != y.len() {
      return Err(CopulaError::DimensionMismatch {
        expected: x.len(),
        got: y.len(),
      });
    }
    if x.is_empty() {
      return Err(CopulaError::InsufficientData { needed: 1, got: 0 });
    }
    if x.iter().chain(y).any(|v| v.is_nan()) {
      return Err(CopulaError::MarginalOutOfRange);
    }
    Ok(Self {
      pseudo: pseudo_observations(stack![Axis(1), x, y].view()),
    })
  }

  /// The pseudo-observations `(n, 2)`, `rank/(n+1)` per column.
  pub fn rank_data(&self) -> ArrayView2<'_, f64> {
    self.pseudo.view()
  }

  /// `C_n(u, v) = n^{-1} Σ 1{U_i ≤ u, V_i ≤ v}` at every row of `x`: `DimensionMismatch` unless `x` has two columns, then
  /// `MarginalOutOfRange` for a value outside `[0, 1]` or NaN, as every copula's batch methods.
  pub fn cdf(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, CopulaError> {
    if x.ncols() != 2 {
      return Err(CopulaError::DimensionMismatch {
        expected: 2,
        got: x.ncols(),
      });
    }
    check_unit(x)?;
    let n = self.pseudo.nrows() as f64;
    Ok(
      x.rows()
        .into_iter()
        .map(|q| {
          let below = self
            .pseudo
            .rows()
            .into_iter()
            .filter(|p| p[0] <= q[0] && p[1] <= q[1])
            .count();
          below as f64 / n
        })
        .collect(),
    )
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
    let ec = EmpiricalCopula2D::new(x.view(), y.view()).unwrap();
    assert_eq!(ec.rank_data().row(0).to_vec(), vec![0.6, 0.2]);
    assert_eq!(
      ec.rank_data(),
      pseudo_observations(stack![Axis(1), x, y].view()).view()
    );
    let c = ec
      .cdf(array![[0.5, 0.5], [1.0, 1.0], [0.1, 0.9]].view())
      .unwrap();
    assert_eq!(c.to_vec(), vec![0.25, 1.0, 0.0]);
    let a = ec.sample(17, &Deterministic::new(42));
    assert_eq!(a.dim(), (17, 2));
    assert_eq!(a, ec.sample(17, &Deterministic::new(42)));
    assert_ne!(a, ec.sample(17, &Deterministic::new(7)));
    assert!(a.iter().all(|v| (0.2..=0.8).contains(v)));
    assert_eq!(ec.sample(5, &Unseeded).nrows(), 5);
  }

  /// The cdf is a step function of the pseudo-observations on the closed square; a value off it or NaN is refused.
  #[test]
  fn the_cdf_is_defined_on_the_square_and_refuses_off_it() {
    let ec = EmpiricalCopula2D::new(
      array![3.0, 1.0, 2.0, 4.0].view(),
      array![10.0, 30.0, 20.0, 40.0].view(),
    )
    .unwrap();
    let c = ec
      .cdf(array![[0.4, 0.4], [0.0, 1.0], [1.0, 1.0]].view())
      .unwrap();
    assert_eq!(c.to_vec(), [0.25, 0.0, 1.0]);
    for row in [[-1.0, 0.5], [0.5, 9.0], [f64::NAN, 0.5], [0.5, f64::NAN]] {
      assert_eq!(
        ec.cdf(array![row].view()),
        Err(CopulaError::MarginalOutOfRange),
        "{row:?}"
      );
    }
  }

  /// The empirical copula is a function of the data set: the same tied pairs in two row orders give the same cdf.
  #[test]
  fn the_cdf_does_not_depend_on_the_row_order_of_tied_data() {
    let x = array![1.0, 1.0, 2.0, 3.0];
    let a = EmpiricalCopula2D::new(x.view(), array![1.0, 2.0, 2.0, 3.0].view()).unwrap();
    let b = EmpiricalCopula2D::new(x.view(), array![2.0, 1.0, 2.0, 3.0].view()).unwrap();
    let q = array![[0.25, 0.25], [0.3, 0.2], [0.3, 0.5], [0.6, 0.5], [1.0, 1.0]];
    assert_eq!(a.cdf(q.view()), b.cdf(q.view()));
    assert_eq!(
      a.cdf(q.view()).unwrap().to_vec(),
      vec![0.0, 0.25, 0.5, 0.75, 1.0]
    );
  }

  /// Data errors are answered, not panicked on: unequal lengths, empty series, a NaN entry.
  #[test]
  fn data_errors_are_answered() {
    let mismatch = CopulaError::DimensionMismatch {
      expected: 2,
      got: 1,
    };
    assert_eq!(
      EmpiricalCopula2D::new(array![1.0, 2.0].view(), array![1.0].view()).err(),
      Some(mismatch)
    );
    let empty = Array1::<f64>::zeros(0);
    let short = CopulaError::InsufficientData { needed: 1, got: 0 };
    assert_eq!(
      EmpiricalCopula2D::new(empty.view(), empty.view()).err(),
      Some(short)
    );
    let nan = EmpiricalCopula2D::new(
      array![1.0, f64::NAN, 3.0].view(),
      array![1.0, 2.0, 3.0].view(),
    );
    assert_eq!(nan.err(), Some(CopulaError::MarginalOutOfRange));
  }

  #[test]
  fn the_cdf_takes_pairs() {
    let ec = EmpiricalCopula2D::new(array![1.0, 2.0].view(), array![2.0, 1.0].view()).unwrap();
    let want = Err(CopulaError::DimensionMismatch {
      expected: 2,
      got: 1,
    });
    assert_eq!(ec.cdf(array![[0.5], [0.7]].view()), want);
    assert_eq!(
      ec.cdf(array![[0.5, 0.5, 0.5]].view()),
      Err(CopulaError::DimensionMismatch {
        expected: 2,
        got: 3
      })
    );
  }
}
