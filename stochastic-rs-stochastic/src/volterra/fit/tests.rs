use super::*;
use crate::rough::kernel::RlKernel;

/// Grid resolution of every measurement: large enough that the log-grid trapezoidal rule is not
/// the limiting error, and shared by the fit and the metric.
const GRID: usize = 4000;

/// The scored interval: it excludes the origin, where the Riemann–Liouville kernel is singular,
/// and spans four decades so the fit cannot specialise to one scale.
const T_MIN: f64 = 1e-4;
const T_MAX: f64 = 1.0;

/// [`fit_l1`] lowers the $L^1$ error at the same degree and nodes, strictly.
/// Reference: Bayer & Breneis (2023), arXiv:2309.07023, the weak error follows the $L^1$ error
#[test]
fn l1_fit_beats_the_quadrature_fit_in_l1() {
  for &h in &[0.05, 0.1, 0.3, 0.45] {
    let quadrature = RlKernel::<f64>::new(h, 40);
    let refit = fit_l1(&quadrature, T_MIN, T_MAX, GRID);

    let before = l1_error(&quadrature, &quadrature, T_MIN, T_MAX, GRID);
    let after = l1_error(&quadrature, &refit, T_MIN, T_MAX, GRID);

    assert!(
      after < before,
      "h={h}: refit did not improve L1 (before={before:e}, after={after:e})"
    );
  }
}

/// The refit is weight-only: a change that starts moving nodes has to say so.
#[test]
fn l1_fit_keeps_the_nodes() {
  let quadrature = RlKernel::<f64>::new(0.3, 32);
  let refit = fit_l1(&quadrature, T_MIN, T_MAX, GRID);

  assert_eq!(refit.degree(), quadrature.degree());
  for (a, b) in refit.nodes().iter().zip(quadrature.nodes().iter()) {
    assert_eq!(a, b, "nodes must be carried over unchanged");
  }
}

/// The worst relative error over the interval does not regress against the quadrature fit: at
/// degree 40 on `[1e-4, 1]` the measured ratio is 0.70 to 0.86 for `H` from 0.45 down to 0.05.
#[test]
fn l1_refit_does_not_regress_the_worst_pointwise_error() {
  for &h in &[0.05, 0.1, 0.3, 0.45] {
    let quadrature = RlKernel::<f64>::new(h, 40);
    let refit = fit_l1(&quadrature, T_MIN, T_MAX, GRID);

    let (mut worst_quadrature, mut worst_refit) = (0.0_f64, 0.0_f64);
    for i in 0..400 {
      let t = T_MIN * (T_MAX / T_MIN).powf(i as f64 / 399.0);
      let truth = VolterraKernel::evaluate(&quadrature, t);
      let sum = |k: &dyn Fn(usize) -> (f64, f64), n: usize| -> f64 {
        (0..n)
          .map(|l| {
            let (x, w) = k(l);
            w * (-x * t).exp()
          })
          .sum()
      };
      let aq = sum(
        &|l| {
          (
            quadrature.nodes()[l],
            VolterraKernel::weights(&quadrature)[l],
          )
        },
        quadrature.degree(),
      );
      let ar = sum(&|l| (refit.nodes()[l], refit.weights()[l]), refit.degree());
      worst_quadrature = worst_quadrature.max((aq - truth).abs() / truth.abs());
      worst_refit = worst_refit.max((ar - truth).abs() / truth.abs());
    }

    assert!(
      worst_refit <= worst_quadrature,
      "h={h}: refit's worst pointwise error regressed \
       ({worst_refit:e} vs quadrature {worst_quadrature:e})"
    );
  }
}

#[test]
#[should_panic(expected = "t_min must be strictly positive")]
fn rejects_a_nonpositive_t_min() {
  let k = RlKernel::<f64>::new(0.3, 16);
  let _ = fit_l1(&k, 0.0, 1.0, 100);
}

#[test]
#[should_panic(expected = "t_max must exceed t_min")]
fn rejects_an_inverted_interval() {
  let k = RlKernel::<f64>::new(0.3, 16);
  let _ = fit_l1(&k, 1.0, 1e-4, 100);
}

#[test]
#[should_panic(expected = "or the fit is underdetermined")]
fn rejects_a_grid_smaller_than_the_degree() {
  let k = RlKernel::<f64>::new(0.3, 32);
  let _ = fit_l1(&k, T_MIN, T_MAX, 16);
}
