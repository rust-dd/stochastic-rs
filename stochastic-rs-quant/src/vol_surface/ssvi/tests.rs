use super::calibrate::calibrate_ssvi;
use super::params::SsviParams;
use super::params::SsviSlice;
use super::surface::SsviSurface;

#[test]
fn ssvi_evaluation() {
  let p = SsviParams::<f64>::new(-0.3, 0.5, 0.5);
  assert!(p.satisfies_no_butterfly_condition());

  let theta = 0.04;
  let w0 = p.total_variance(0.0, theta);
  assert!(w0 > 0.0);
  assert!(
    (w0 - theta).abs() < 1e-10,
    "ATM total variance should equal theta: w0={w0}, theta={theta}"
  );
}

#[test]
fn ssvi_derivatives() {
  let p = SsviParams::<f64>::new(-0.3, 0.5, 0.5);
  let theta = 0.04;
  let k = 0.1;
  let h = 1e-5;

  let num_first = (p.total_variance(k + h, theta) - p.total_variance(k - h, theta)) / (2.0 * h);
  let num_second = (p.total_variance(k + h, theta) - 2.0 * p.total_variance(k, theta)
    + p.total_variance(k - h, theta))
    / (h * h);

  assert!(
    (p.w_prime_k(k, theta) - num_first).abs() < 1e-5,
    "w'(k) mismatch"
  );
  let rel_err = (p.w_double_prime_k(k, theta) - num_second).abs()
    / p.w_double_prime_k(k, theta).abs().max(1e-14);
  assert!(
    rel_err < 1e-3,
    "w''(k) mismatch: analytic={} numeric={} rel_err={}",
    p.w_double_prime_k(k, theta),
    num_second,
    rel_err
  );
}

#[test]
fn ssvi_calibration_exact() {
  let true_params = SsviParams::<f64>::new(-0.3, 0.5, 0.5);
  let thetas = [0.02, 0.04, 0.08];

  let slices: Vec<SsviSlice<f64>> = thetas
    .iter()
    .map(|&theta| {
      let ks: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.1).collect();
      let ws: Vec<f64> = ks
        .iter()
        .map(|&k| true_params.total_variance(k, theta))
        .collect();
      SsviSlice {
        log_moneyness: ks,
        total_variance: ws,
        theta,
      }
    })
    .collect();

  let fitted = calibrate_ssvi(&slices, None);

  assert!((fitted.rho - true_params.rho).abs() < 1e-3, "rho mismatch");
  assert!((fitted.eta - true_params.eta).abs() < 1e-3, "eta mismatch");
  assert!(
    (fitted.gamma - true_params.gamma).abs() < 1e-3,
    "gamma mismatch"
  );
}

#[test]
fn ssvi_surface_interpolation() {
  let params = SsviParams::<f64>::new(-0.3, 0.5, 0.5);
  let surface = SsviSurface::new(params, vec![0.02, 0.04, 0.08], vec![0.25, 0.50, 1.0]);

  let ks = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
  assert!(surface.is_calendar_spread_free(&ks));
  assert!(surface.is_atm_calendar_spread_free());

  let iv = surface.implied_vol(0.0, 0.5);
  assert!(iv.is_finite() && iv > 0.0);

  let iv_interp = surface.implied_vol(0.0, 0.375);
  assert!(iv_interp.is_finite() && iv_interp > 0.0);
}

/// Regression: a non-monotonic θ_t term structure must fail BOTH the ATM
/// and the smile-wide checks. A monotonic θ_t with a strong-skew SSVI can
/// still violate calendar arb off-ATM — pre-rc.1 the surface flag would
/// say "arb-free" in that scenario, hiding the issue.
#[test]
fn calendar_spread_free_grid_catches_off_atm_violations() {
  let params = SsviParams::<f64>::new(-0.3, 0.5, 0.5);
  let surface_decreasing = SsviSurface::new(params, vec![0.04, 0.03, 0.02], vec![0.25, 0.50, 1.0]);
  let ks = vec![-1.0, 0.0, 1.0];
  assert!(!surface_decreasing.is_calendar_spread_free(&ks));
  assert!(!surface_decreasing.is_atm_calendar_spread_free());
}
