//! A vine's cdf by Genz's separation of variables through its Rosenblatt map: `C(u) = ∫_{[0,1]^{d−1}} ∏_{i=1}^{d} e_i dw`,
//! `e_1 = u_1`, `e_i = F(u_i | x_1, …, x_{i−1})` and `x_i = F⁻¹(w_i e_i | x_1, …, x_{i−1})`, averaged over the Sobol points.
//! Reference: Rosenblatt, M. (1952), "Remarks on a Multivariate Transformation", Ann. Math. Statist. 23(3) 470–472, DOI 10.1214/aoms/1177729394
//! Reference: Genz, A. (1992), "Numerical Computation of Multivariate Normal Probabilities", JCGS 1(2) 141–149, DOI 10.1080/10618600.1992.10477010 (§2)

use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;

use super::qmc::sobol_points;

/// The inverse h-chain a vine's sampler walks, and the forward chain over the same conditioning values.
pub(super) trait Rosenblatt {
  fn workspace(&self) -> Vec<Vec<f64>>;

  /// Places `x_i` in `v[i][1]` at the inverse h-chain of `target(i, v)` for `i = 1, 2, …`; `None` stops the walk.
  fn walk(&self, v: &mut [Vec<f64>], target: impl FnMut(usize, &[Vec<f64>]) -> Option<f64>);

  /// `F(u | x_1, …, x_{i−1})` over the conditioning values the walk has left in `v`.
  fn conditional_cdf(&self, i: usize, u: f64, v: &[Vec<f64>]) -> f64;
}

/// Each row of `x`, in vine order, is 0 when a coordinate is 0 and otherwise the mean of `∏ e_i` over the Sobol points.
pub(super) fn cdf(vine: &impl Rosenblatt, x: ArrayView2<f64>) -> Array1<f64> {
  let mut v = vine.workspace();
  let mut points = None;
  let mut values = Vec::new();
  x.rows()
    .into_iter()
    .map(|u| {
      if u.iter().any(|&ui| ui == 0.0) {
        return 0.0;
      }
      let points = points.get_or_insert_with(|| sobol_points(u.len() - 1));
      values.clear();
      values.extend(
        points
          .rows()
          .into_iter()
          .map(|w| integrand(vine, u, w.as_slice().expect("row-major"), &mut v)),
      );
      pairwise_sum(&values) / values.len() as f64
    })
    .collect()
}

/// The walk places `x_i` at `w_i e_i` and stops once `e_d` is in the product; `F(1 | ·) = 1` exactly.
fn integrand(vine: &impl Rosenblatt, u: ArrayView1<f64>, w: &[f64], v: &mut [Vec<f64>]) -> f64 {
  let d = u.len();
  let mut product = 1.0;
  vine.walk(v, |i, v| {
    let e = if u[i - 1] == 1.0 {
      1.0
    } else {
      vine.conditional_cdf(i, u[i - 1], v)
    };
    product *= e;
    (i < d).then(|| w[i - 1] * e)
  });
  product
}

/// Halves of equal values add without rounding, so over the power-of-two point count a constant integrand is exact.
fn pairwise_sum(x: &[f64]) -> f64 {
  match x {
    [] => 0.0,
    [a] => *a,
    _ => {
      let (left, right) = x.split_at(x.len() / 2);
      pairwise_sum(left) + pairwise_sum(right)
    }
  }
}
