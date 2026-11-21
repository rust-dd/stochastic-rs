use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use stochastic_rs_distributions::RealExt;

use super::LeastSquaresProblem;

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

/// MINPACK's `chkder` as a test: the worst column's `‖J − D‖/‖D‖` at `x`, `D` the central
/// difference of the residuals over `step(x_j)`; two zero columns read 0.
pub(crate) fn jacobian_error<P: LeastSquaresProblem>(
  problem: &mut P,
  x: ArrayView1<f64>,
  step: impl Fn(usize, f64) -> f64,
) -> f64 {
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
  f64::max_or_nan_fold(columns.iter().enumerate().map(|(j, difference)| {
    let gap = norm(&(&jacobian.column(j) - difference));
    let scale = norm(difference);
    if scale == 0.0 { gap } else { gap / scale }
  }))
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

/// Per coordinate, a point 0.3 of the step `max(1e-5, 1e-3·|x|)` inside each bound of `base`'s
/// box, where a central stencil would cross the bound.
pub(crate) fn beside_points(
  base: ArrayView1<f64>,
  bounds: &[(f64, f64)],
) -> Vec<(String, Array1<f64>)> {
  let step = |x: f64| 1e-5_f64.max(1e-3 * x.abs());
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
