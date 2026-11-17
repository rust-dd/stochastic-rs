use ndarray::Array1;

use super::test_support::simulate_cir;
use super::test_support::simulate_ou;
use super::*;
use crate::gmm_cir::GmmCirFit;
use crate::gmm_cir::fit_cir_gmm;

type Fit = Result<(f64, f64, f64, usize, bool), StatsError>;

/// `(θ, μ/c, σ/s)` and how the search ended: what a path scaled by `c` shares with the path itself.
fn read(fit: Result<QmleFit, StatsError>, c: f64, s: f64) -> Fit {
  fit.map(|f| (f.theta, f.mu / c, f.sigma / s, f.iterations, f.converged))
}

/// [`read`] of a GMM fit, with `J` and its p-value, which no scale moves.
fn read_gmm(fit: Result<GmmCirFit, StatsError>, c: f64, s: f64) -> (Fit, Option<(f64, f64)>) {
  let j = fit.as_ref().ok().map(|f| (f.j_stat, f.j_pvalue));
  let fit = fit.map(|f| (f.theta, f.mu / c, f.sigma / s, f.iterations, f.converged));
  (fit, j)
}

/// The four fits of a non-negative `x` scaled by `2^k`, an even `k`, read in the units of `x`.
fn fits_at(x: &Array1<f64>, dt: f64, k: i32) -> ([Fit; 4], Option<(f64, f64)>) {
  let (c, root) = (2f64.powi(k), 2f64.powi(k / 2));
  let y = x.mapv(|v| c * v);
  let (gmm, j) = read_gmm(fit_cir_gmm(y.view(), dt), c, root);
  let ou = DiffusionKind::OrnsteinUhlenbeck;
  let fits = [
    read(fit_diffusion_qmle(y.view(), dt, ou), c, c),
    read(fit_ou_mle(y.view(), dt), c, c),
    read(
      fit_diffusion_qmle(y.view(), dt, DiffusionKind::Cir),
      c,
      root,
    ),
    gmm,
  ];
  (fits, j)
}

/// Each fit reads the path at unit scale, a power of two away, so `2^k x` fits as `x` to the last
/// bit, at scales whose conditional variances and whose sums leave the double range.
#[test]
fn a_power_of_two_scales_each_fit_exactly() {
  let dt = 1.0 / 252.0;
  for x in [
    simulate_ou(2.0, 0.5, 0.1, 0.5, dt, 1000, 11),
    simulate_cir(2.0, 0.04, 0.2, 0.04, dt, 2000, 7),
  ] {
    let unit = fits_at(&x, dt, 0);
    assert!(unit.0.iter().all(Result::is_ok), "{:?}", unit.0);
    for k in [-1000, -532, -200, 40, 900] {
      assert_eq!(fits_at(&x, dt, k), unit, "2^{k}");
    }
  }
}

/// A density of `cx` is `1/c` of that of `x`, so the log-likelihood falls by `ln c` a transition.
#[test]
fn the_log_likelihood_moves_by_the_log_of_the_scale() {
  let dt = 1.0 / 252.0;
  let x = simulate_ou(2.0, 0.5, 0.1, 0.5, dt, 1000, 11);
  let fits = |x: &Array1<f64>| {
    [
      fit_diffusion_qmle(x.view(), dt, DiffusionKind::OrnsteinUhlenbeck),
      fit_ou_mle(x.view(), dt),
      fit_diffusion_qmle(x.view(), dt, DiffusionKind::Cir),
    ]
    .map(|fit| fit.unwrap().log_likelihood)
  };
  let unit = fits(&x);
  for k in [-1000, -200, 40, 900] {
    let shift = 999.0 * f64::from(k) * std::f64::consts::LN_2;
    for (got, unit) in fits(&x.mapv(|v| 2f64.powi(k) * v)).into_iter().zip(unit) {
      let want = unit - shift;
      assert!((got / want - 1.0).abs() < 1e-12, "2^{k}: {got} for {want}");
    }
  }
}

/// A path whose sum or whose squares leave the double range gets the answer of the path at unit
/// scale, an `Err` as a fit: no stage of a fit sees the data's unit.
#[test]
fn a_path_at_the_double_ranges_end_answers_as_at_unit_scale() {
  let paths = [
    vec![1.0, 3.0, 5.0, 4.0, 2.0, 1.0, 2.0, 4.0],
    vec![1.0, 1.5, 1.0, 0.5, 1.25],
    vec![1.5, 0.0, 1.5, 0.0, 1.5],
    vec![1.0, 2.0, 1.5, 3.0, 2.5],
  ];
  for x in paths.map(Array1::from) {
    let unit = fits_at(&x, 1.0, 0);
    for k in [-1000, 1020] {
      assert_eq!(fits_at(&x, 1.0, k), unit, "{x} at 2^{k}");
    }
  }
}

/// At `2^1020` a `σ̂` of 32 in the path's unit is past the largest double, and at a step of `1e60`
/// one near `1e-30` of a path at `2^-1000` is below the smallest.
#[test]
fn a_fit_outside_the_double_range_is_a_numerical_error() {
  let x = Array1::from(vec![1.0, 3.0, 5.0, 4.0, 2.0, 1.0, 2.0, 4.0]);
  let refused = Err(StatsError::Numerical {
    stage: "data scale",
  });
  let high = x.mapv(|v| 2f64.powi(1020) * v);
  let ou = fit_diffusion_qmle(high.view(), 1.0 / 252.0, DiffusionKind::OrnsteinUhlenbeck);
  assert_eq!(ou.map(|f| f.sigma), refused);
  let closed = fit_ou_mle(high.view(), 1.0 / 252.0);
  assert_eq!(closed.map(|f| f.sigma), refused);
  let low = x.mapv(|v| 2f64.powi(-1000) * v);
  assert_eq!(fit_ou_mle(low.view(), 1e60).map(|f| f.sigma), refused);
}

/// At the smallest step the conditional variance at `σ = 1` underflows, so `σ₀` is `∞`.
#[test]
fn a_step_whose_unit_variance_underflows_has_no_start() {
  let x = simulate_cir(2.0, 0.04, 0.2, 0.04, 1.0 / 252.0, 200, 7);
  let dt = f64::from_bits(1);
  let refused = Err(StatsError::Numerical {
    stage: "starting point",
  });
  for kind in [DiffusionKind::OrnsteinUhlenbeck, DiffusionKind::Cir] {
    assert_eq!(
      fit_diffusion_qmle(x.view(), dt, kind).map(|f| f.sigma),
      refused
    );
  }
  assert_eq!(fit_cir_gmm(x.view(), dt).map(|f| f.sigma), refused);
}

/// Stage 1 weighs the moments of the path over its mean, so a factor that is no power of two, and
/// so rounds the path, moves the GMM fit by that rounding; the path's own would follow its unit.
#[test]
fn a_decimal_scale_moves_the_gmm_fit_by_its_rounding_alone() {
  let dt = 1.0 / 252.0;
  for seed in [3u64, 7, 29] {
    let x = simulate_cir(2.0, 0.06, 0.2, 0.06, dt, 500, seed);
    let unit = fit_cir_gmm(x.view(), dt).unwrap();
    for c in [1e-7, 1e-4, 1e4, 1e7] {
      let fit = fit_cir_gmm(x.mapv(|v| c * v).view(), dt).unwrap();
      let pairs = [
        (fit.theta, unit.theta),
        (fit.mu / c, unit.mu),
        (fit.sigma / c.sqrt(), unit.sigma),
        (1.0 + fit.j_stat, 1.0 + unit.j_stat),
      ];
      for (a, b) in pairs {
        assert!(
          (a / b - 1.0).abs() < 1e-8,
          "seed {seed} x{c:e}: {a} for {b}"
        );
      }
    }
  }
}

/// On `X_{t+1} = a X_t + b` in doubles the GMM's `σ̂` is where its simplex stops on its way to 0,
/// which moves with the scale, so it refuses the path as the CIR QMLE does.
#[test]
fn the_gmm_refuses_an_exact_fit_as_the_cir_qmle_does() {
  let dt = 1.0 / 252.0;
  let refused = Err(StatsError::DegenerateData("zero residual variance"));
  for (a, b, start, n) in [(0.9, 0.1, 2.0, 200), (0.37, 1.1, 3.0, 12)] {
    let mut x = vec![start; n];
    for t in 1..n {
      x[t] = a * x[t - 1] + b;
    }
    for c in [1.0, 1e-4, 1e4, 3.0, 0.7] {
      let y = Array1::from_iter(x.iter().map(|v| c * v));
      let qmle = fit_diffusion_qmle(y.view(), dt, DiffusionKind::Cir);
      assert_eq!(qmle.map(|f| f.sigma), refused, "{a} X + {b} x{c}");
      let gmm = fit_cir_gmm(y.view(), dt);
      assert_eq!(gmm.map(|f| f.sigma), refused, "{a} X + {b} x{c}");
    }
  }
}
