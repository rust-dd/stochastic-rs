use super::*;
use crate::rough::kernel::RlKernel;

/// Grid resolution for every measurement here. Large enough that the
/// trapezoidal rule on a log grid is not itself the limiting error, and the
/// same value is used for the fit and the metric so the comparison is
/// like-for-like.
const GRID: usize = 4000;

/// The interval the fit is scored on. It excludes the origin because the
/// Riemann–Liouville kernel is singular there, and it spans four decades so
/// the fit cannot win by specialising to one scale.
const T_MIN: f64 = 1e-4;
const T_MAX: f64 = 1.0;

/// The whole claim of [`fit_l1`] is that it lowers the $L^1$ error at the same
/// degree and the same nodes. If it does not, the fit is not doing its job —
/// so this asserts a strict improvement, not merely "close enough".
///
/// Reference for why $L^1$ is the objective worth optimising: Bayer & Breneis
/// (2023), *Weak Markovian approximations of rough Heston*, arXiv:2309.07023,
/// which bounds the weak (pricing) error by the $L^1$ kernel error.
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

/// The refit keeps the quadrature's nodes by construction — it is a weight-only
/// optimisation. Pinning that here means a future change that starts moving
/// nodes has to say so rather than silently altering what the function does.
#[test]
fn l1_fit_keeps_the_nodes() {
  let quadrature = RlKernel::<f64>::new(0.3, 32);
  let refit = fit_l1(&quadrature, T_MIN, T_MAX, GRID);

  assert_eq!(refit.degree(), quadrature.degree());
  for (a, b) in refit.nodes().iter().zip(quadrature.nodes().iter()) {
    assert_eq!(a, b, "nodes must be carried over unchanged");
  }
}

/// An $L^1$-optimal fit is allowed to be locally worse — that is the point of
/// integrating the error rather than bounding it pointwise — so the guard that
/// matters is that it does not buy its integral by oscillating wildly
/// somewhere. This asserts the **worst** relative error over the whole fitted
/// interval does not regress against the quadrature fit.
///
/// Measured at degree 40 on `[1e-4, 1]`, the refit is better on both metrics:
///
/// | $H$ | $L^1$ before → after | max pointwise rel. before → after |
/// |---|---|---|
/// | 0.05 | 2.98e-2 → 1.29e-2 (2.31×) | 8.30e-1 → 7.14e-1 (0.86×) |
/// | 0.10 | 2.10e-2 → 9.02e-3 (2.33×) | 7.91e-1 → 6.69e-1 (0.85×) |
/// | 0.30 | 3.83e-3 → 1.35e-3 (2.83×) | 5.31e-1 → 4.19e-1 (0.79×) |
/// | 0.45 | 4.16e-4 → 1.14e-4 (3.64×) | 1.68e-1 → 1.17e-1 (0.70×) |
///
/// The bar is therefore "no worse", not a tuned constant: every measured ratio
/// sits at 0.86 or below, so a regression past 1.0 would mean a real change in
/// behaviour. Note the absolute figures are large because the sample reaches
/// `t = 1e-4`, where a degree-40 exponential sum cannot track the singularity;
/// that limitation belongs to both fits equally, which is why this compares
/// them rather than bounding either.
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
