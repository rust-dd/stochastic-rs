use ndarray::Array1;

use super::*;
use crate::qmle::fit_diffusion_qmle;
use crate::qmle::test_support::simulate_cir;
use crate::qmle::test_support::simulate_ou;

const OU: DiffusionKind = DiffusionKind::OrnsteinUhlenbeck;

fn paths(n: usize) -> [(DiffusionKind, Array1<f64>); 2] {
  let dt = 1.0 / 252.0;
  [
    (OU, simulate_ou(2.0, 0.5, 0.3, 0.5, dt, n, 5)),
    (
      DiffusionKind::Cir,
      simulate_cir(2.0, 0.04, 0.2, 0.04, dt, n, 5),
    ),
  ]
}

/// The posterior of `c x` in the units of `x`: the level over `c` and `sigma` over `s`.
fn posterior(x: &Array1<f64>, kind: DiffusionKind, c: f64, s: f64) -> [f64; 10] {
  let y = x.mapv(|v| c * v);
  let step = [0.3, 0.3, 0.05];
  let f = fit_diffusion_bayesian(y.view(), 1.0 / 252.0, kind, step, 3000, 1000, 1).unwrap();
  [
    f.theta_mean,
    f.theta_ci.0,
    f.theta_ci.1,
    f.mu_mean / c,
    f.mu_ci.0 / c,
    f.mu_ci.1 / c,
    f.sigma_mean / s,
    f.sigma_ci.0 / s,
    f.sigma_ci.1 / s,
    f.acceptance_rate,
  ]
}

/// The chain runs on the path at unit scale, a power of two away, so `2^k x` has the posterior of
/// `x` to the last bit, at scales whose conditional variances and sums leave the double range.
#[test]
fn a_power_of_two_scales_the_posterior_exactly() {
  for (kind, x) in paths(200) {
    let unit = posterior(&x, kind, 1.0, 1.0);
    for k in [-1000, -200, 40, 900] {
      let (c, root) = (2f64.powi(k), 2f64.powi(k / 2));
      let s = if kind == OU { c } else { root };
      assert_eq!(posterior(&x, kind, c, s), unit, "{kind:?} at 2^{k}");
    }
  }
}

/// The prior on `ln σ` and the chain's start sit at the regression's `σ₀`, so a factor that is no
/// power of two leaves `σ` by the QMLE's and the speed by the unit path's posterior.
#[test]
fn a_decimal_scale_keeps_the_posterior_by_the_qmle() {
  for (kind, x) in paths(1000) {
    let qmle = fit_diffusion_qmle(x.view(), 1.0 / 252.0, kind).unwrap();
    let unit = posterior(&x, kind, 1.0, 1.0);
    for c in [1e-160, 1e-12, 1e-7, 1e-4, 1e4, 1e150] {
      let s = if kind == OU { c } else { f64::sqrt(c) };
      let scaled = posterior(&x, kind, c, s);
      assert!(
        (scaled[6] / qmle.sigma - 1.0).abs() < 0.2,
        "{kind:?} x{c:e}: σ {} for {}",
        scaled[6],
        qmle.sigma
      );
      assert!(
        (scaled[0] / unit[0] - 1.0).abs() < 0.5,
        "{kind:?} x{c:e}: θ {} for {}",
        scaled[0],
        unit[0]
      );
    }
  }
}

/// A path near `1e153` has its posterior in its own unit; one whose interval's end or
/// whose `σ` passes the largest double there, or whose start has no finite `σ₀`, is a `Numerical`.
#[test]
fn a_path_at_the_double_ranges_end_has_a_posterior_or_a_numerical_err() {
  let dt = 1.0 / 252.0;
  let run =
    |x: &Array1<f64>, dt: f64| fit_diffusion_bayesian(x.view(), dt, OU, [0.1; 3], 100, 10, 1);
  let x = Array1::from(vec![1.0, 3.0, 5.0, 4.0, 2.0, 1.0, 2.0, 4.0]);
  let fit = run(&x.mapv(|v| 1e153 * v), dt).unwrap();
  assert!(1e153 < fit.mu_mean && fit.mu_mean < 5e153, "{fit:?}");
  assert!(
    fit.sigma_mean.is_finite() && fit.sigma_mean > 1e153,
    "{fit:?}"
  );
  let outside = Err(StatsError::Numerical {
    stage: "data scale",
  });
  let wide = Array1::from(vec![1e308, 1.5e308, 1e308, 0.5e308, 1.2e308]);
  assert_eq!(run(&wide, dt).map(|f| f.mu_mean), outside);
  let top = x.mapv(|v| 2f64.powi(1020) * v);
  assert_eq!(run(&top, dt).map(|f| f.sigma_mean), outside);
  assert_eq!(
    run(&x, f64::from_bits(1)).map(|f| f.sigma_mean),
    Err(StatsError::Numerical {
      stage: "starting point"
    })
  );
}
