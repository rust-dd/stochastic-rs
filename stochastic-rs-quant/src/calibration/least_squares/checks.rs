use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::RealExt;

use super::LeastSquaresProblem;
use crate::calibration::stencil::step;

/// `∂θ/∂x` of a problem's natural parameters by central differences of step `h` around `x`.
pub(crate) fn natural_jacobian_by_differences<P: LeastSquaresProblem>(
  problem: &mut P,
  x: ArrayView1<f64>,
  h: f64,
) -> Array2<f64> {
  let mut at = |y: &Array1<f64>| {
    problem.set_params(y.view());
    Array1::from_vec(problem.natural_params())
  };
  let rows = at(&x.to_owned()).len();
  let mut jacobian = Array2::zeros((rows, x.len()));
  for column in 0..x.len() {
    let (mut up, mut down) = (x.to_owned(), x.to_owned());
    up[column] += h;
    down[column] -= h;
    let derivative = (at(&up) - at(&down)) / (2.0 * h);
    jacobian.column_mut(column).assign(&derivative);
  }
  problem.set_params(x);
  jacobian
}

/// The largest entrywise `|a − b| / max(|a|, |b|)`, zero where both are zero; NaN propagates.
pub(crate) fn worst_relative_error(a: ArrayView2<f64>, b: ArrayView2<f64>) -> f64 {
  let ratio = |(x, y): (&f64, &f64)| {
    let scale = x.abs().max(y.abs());
    if scale == 0.0 {
      0.0
    } else {
      (x - y).abs() / scale
    }
  };
  f64::max_or_nan_fold(a.iter().zip(&b).map(ratio))
}

/// Per column, `‖J − D‖` and `‖D‖` at `x`, `D` the central difference of the residuals over
/// `step(x_j)`.
pub(crate) fn jacobian_gaps<P: LeastSquaresProblem>(
  problem: &mut P,
  x: ArrayView1<f64>,
  step: impl Fn(usize, f64) -> f64,
) -> Vec<(f64, f64)> {
  let mut at = |y: &Array1<f64>| {
    problem.set_params(y.view());
    problem.residuals().expect("residuals beside the point")
  };
  let columns = (0..x.len())
    .map(|column| {
      let h = step(column, x[column]);
      let (mut up, mut down) = (x.to_owned(), x.to_owned());
      up[column] += h;
      down[column] -= h;
      (at(&up) - at(&down)) / (2.0 * h)
    })
    .collect::<Vec<_>>();
  problem.set_params(x);
  let jacobian = problem.jacobian().expect("a Jacobian at the point");
  let norm = |v: &Array1<f64>| v.dot(v).sqrt();
  let columns = columns.iter().enumerate();
  columns
    .map(|(j, difference)| (norm(&(&jacobian.column(j) - difference)), norm(difference)))
    .collect()
}

/// MINPACK's `chkder` as a test: the worst column's `‖J − D‖/‖D‖` of [`jacobian_gaps`]; two zero
/// columns read 0.
pub(crate) fn jacobian_error<P: LeastSquaresProblem>(
  problem: &mut P,
  x: ArrayView1<f64>,
  step: impl Fn(usize, f64) -> f64,
) -> f64 {
  let gaps = jacobian_gaps(problem, x, step).into_iter();
  f64::max_or_nan_fold(gaps.map(|(gap, scale)| if scale == 0.0 { gap } else { gap / scale }))
}

/// The residuals' rounding at `x` along coordinate `j`: half the largest second difference over
/// steps of one to six parts in `1e9`, where no curvature shows.
pub(crate) fn residual_rounding<P: LeastSquaresProblem>(
  problem: &mut P,
  x: ArrayView1<f64>,
  j: usize,
) -> f64 {
  let mut at = |offset: f64| {
    let mut y = x.to_owned();
    y[j] += offset;
    problem.set_params(y.view());
    problem.residuals().expect("residuals beside the point")
  };
  let centre = at(0.0);
  let seconds = (1..=6).map(|k| {
    let delta = 1e-9 * f64::from(k) * x[j].abs().max(1e-2);
    let second = at(delta) - 2.0 * &centre + at(-delta);
    0.5 * f64::max_or_nan_fold(second.iter().map(|v| v.abs()))
  });
  f64::max_or_nan_fold(seconds.collect::<Vec<_>>())
}

/// Each column of [`jacobian_gaps`] within `tolerance` of `‖D‖`, or past it by at most twice the
/// `(4/h + 1/h_ref)·√m·σ` that the residuals' rounding `σ` explains, `h` the fit's own step.
pub(crate) fn assert_slope<P: LeastSquaresProblem>(
  problem: &mut P,
  label: &str,
  x: ArrayView1<f64>,
  reference: impl Fn(usize, f64) -> f64,
  tolerance: f64,
) {
  let gaps = jacobian_gaps(problem, x, &reference);
  let folded = problem.params();
  let rows = problem.residuals().expect("residuals at the point").len() as f64;
  for (j, (gap, scale)) in gaps.into_iter().enumerate() {
    let width = (4.0 / step(folded[j]) + 1.0 / reference(j, x[j])) * rows.sqrt();
    let within = gap <= tolerance * scale
      || gap <= tolerance * scale + 2.0 * width * residual_rounding(problem, x, j);
    assert!(
      within,
      "{label}, column {j}: {:e} of {:e}",
      gap / scale,
      tolerance
    );
  }
}

/// `base`, then per coordinate each bound and, where the box reflects, `base`'s mirror image
/// below the lower one; 1 % inside a bound, since a stencil that reaches it is one-sided.
pub(crate) fn box_points(
  base: ArrayView1<f64>,
  bounds: &[(f64, f64)],
  reflecting: &[bool],
) -> Vec<(String, Array1<f64>)> {
  let mut points = vec![("base".to_string(), base.to_owned())];
  for (j, &(lower, upper)) in bounds.iter().enumerate() {
    let inset = 0.01 * (upper - lower);
    let mirror = reflecting[j].then_some(("mirror", 2.0 * lower - base[j]));
    let sides = [("lower", lower + inset), ("upper", upper - inset)];
    for (side, value) in sides.into_iter().chain(mirror) {
      let mut point = base.to_owned();
      point[j] = value;
      points.push((format!("{side} {j}"), point));
    }
  }
  points
}

/// Per coordinate, a point 0.3 of the shared step inside each bound of `base`'s box, where a
/// central stencil would cross the bound.
pub(crate) fn beside_points(
  base: ArrayView1<f64>,
  bounds: &[(f64, f64)],
) -> Vec<(String, Array1<f64>)> {
  let mut points = Vec::new();
  for (j, &(lower, upper)) in bounds.iter().enumerate() {
    let sides = [
      ("lower", lower + 0.3 * step(lower)),
      ("upper", upper - 0.3 * step(upper)),
    ];
    for (side, value) in sides.into_iter().filter(|(_, value)| value.is_finite()) {
      let mut point = base.to_owned();
      point[j] = value;
      points.push((format!("beside {side} {j}"), point));
    }
  }
  points
}

/// [`jacobian_error`]'s step at a beside point: a tenth of the shared step, which stays inside the
/// bound and never shrinks under `1e-6`.
pub(crate) fn beside_reference(_: usize, x: f64) -> f64 {
  0.1 * step(x)
}

/// The worst column's `‖J(h) − J(2h)‖ / ‖J(2h)‖`, `jacobian(s)` at `s` times the step.
pub(crate) fn step_doubling_error(jacobian: impl Fn(f64) -> Array2<f64>) -> f64 {
  let (single, double) = (jacobian(1.0), jacobian(2.0));
  let norm = |v: ArrayView1<f64>| v.dot(&v).sqrt();
  let columns = single.columns().into_iter().zip(double.columns());
  f64::max_or_nan_fold(columns.map(|(a, b)| {
    let scale = norm(b);
    let gap = norm((&a - &b).view());
    if scale == 0.0 { gap } else { gap / scale }
  }))
}
