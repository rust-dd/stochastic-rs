//! The quadratic model of an objective at a point: its Hessian by central second differences and
//! the standard errors it implies.

use super::search::NO_VALUE;

/// `√(H⁻¹)ᵢᵢ` of `objective` at `at`, the Hessian by second differences at `steps`; `None` where
/// a point has no value or the Hessian is not positive definite.
pub(crate) fn standard_errors(
  objective: &dyn Fn(&[f64]) -> f64,
  at: &[f64],
  steps: &[f64],
) -> Option<Vec<f64>> {
  let n = at.len();
  let moved = |shifts: &[(usize, f64)]| {
    let mut point = at.to_vec();
    for &(i, side) in shifts {
      point[i] += side * steps[i];
    }
    Some(objective(&point)).filter(|cost| *cost < NO_VALUE)
  };
  let here = moved(&[])?;
  let mut hessian = vec![vec![0.0; n]; n];
  for i in 0..n {
    let (up, down) = (moved(&[(i, 1.0)])?, moved(&[(i, -1.0)])?);
    hessian[i][i] = (up + down - 2.0 * here) / (steps[i] * steps[i]);
    for j in 0..i {
      let corner = |a: f64, b: f64| moved(&[(i, a), (j, b)]);
      let cross = corner(1.0, 1.0)? - corner(1.0, -1.0)? - corner(-1.0, 1.0)? + corner(-1.0, -1.0)?;
      hessian[i][j] = cross / (4.0 * steps[i] * steps[j]);
    }
  }
  // Cholesky `H = LLᵀ`; `(H⁻¹)ᵢᵢ` is the squared norm of column `i` of `L⁻¹`.
  let mut lower = vec![vec![0.0; n]; n];
  for i in 0..n {
    for j in 0..=i {
      let sum = (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
      if i == j {
        let pivot = hessian[i][i] - sum;
        lower[i][i] = Some(pivot).filter(|p| *p > 0.0)?.sqrt();
      } else {
        lower[i][j] = (hessian[i][j] - sum) / lower[j][j];
      }
    }
  }
  let error = |i: usize| {
    let mut column = vec![0.0; n];
    for k in i..n {
      let sum = (i..k).map(|m| lower[k][m] * column[m]).sum::<f64>();
      column[k] = (f64::from(u8::from(k == i)) - sum) / lower[k][k];
    }
    column.iter().map(|v| v * v).sum::<f64>().sqrt()
  };
  Some((0..n).map(error).collect())
}

#[cfg(test)]
mod tests {
  use super::standard_errors;

  /// `½ xᵀHx` with `H = [[4, 1], [1, 2]]`: `H⁻¹` has the diagonal `2/7`, `4/7`.
  #[test]
  fn a_quadratic_gives_its_own_standard_errors() {
    let bowl = |p: &[f64]| 2.0 * p[0] * p[0] + p[0] * p[1] + p[1] * p[1];
    let errors = standard_errors(&bowl, &[0.0, 0.0], &[0.1, 0.1]).unwrap();
    assert!((errors[0] - (2.0_f64 / 7.0).sqrt()).abs() < 1e-12);
    assert!((errors[1] - (4.0_f64 / 7.0).sqrt()).abs() < 1e-12);
  }

  #[test]
  fn a_saddle_has_no_model() {
    let saddle = |p: &[f64]| p[0] * p[0] - p[1] * p[1];
    assert!(standard_errors(&saddle, &[0.0, 0.0], &[0.1, 0.1]).is_none());
  }
}
