use super::*;

#[test]
fn a_nan_quote_is_not_clamped_out_of_the_initial_guess() {
  assert!(
    svi_initial_guess(&[f64::NAN, 0.0, 0.1], &[0.05, 0.04, 0.05])
      .sigma
      .is_nan()
  );
  assert!(
    svi_initial_guess(&[-0.1, 0.0, 0.1], &[0.05, f64::NAN, 0.05])
      .b
      .is_nan()
  );
  let finite = svi_initial_guess(&[-0.1, 0.0, 0.1], &[0.05, 0.04, 0.05]);
  assert!((finite.sigma - 0.06).abs() < 1e-15 && finite.b == 0.01);
}

#[test]
fn a_nan_quote_reaches_the_calibrated_parameters() {
  let ks = [-0.1, 0.0, 0.1];
  let nan_w = calibrate_svi(&ks, &[0.05, f64::NAN, 0.05], None);
  assert!(nan_w.params.b.is_nan() && nan_w.params.rho.is_nan());
  assert!(!nan_w.converged && nan_w.message.as_deref() == Some("solver failed"));
  let nan_k = calibrate_svi(&[-0.1, f64::NAN, 0.1], &[0.05, 0.04, 0.05], None);
  assert!(nan_k.params.sigma.is_nan() && !nan_k.converged);
  let mut projected = SviRawParams::<f64>::new(0.04, -0.1, 1.5, 0.0, 0.0);
  projected.project();
  assert_eq!(
    (projected.b, projected.rho, projected.sigma),
    (0.0, 0.9999, 1e-8)
  );
}

#[test]
fn svi_evaluation() {
  let p = SviRawParams::<f64>::new(0.04, 0.4, -0.4, 0.1, 0.2);
  assert!(p.is_admissible());

  let w0 = p.total_variance(0.0);
  assert!(w0 > 0.0);

  let w_right = p.total_variance(0.5);
  let w_left = p.total_variance(-0.5);
  assert!(w_left > w0, "put-side should be higher with negative rho");
  assert!(w_right > 0.0);
}

#[test]
fn svi_derivatives() {
  let p = SviRawParams::<f64>::new(0.04, 0.4, -0.4, 0.1, 0.2);
  let k = 0.1;
  let h = 1e-7;

  let num_first = (p.total_variance(k + h) - p.total_variance(k - h)) / (2.0 * h);
  let num_second =
    (p.total_variance(k + h) - 2.0 * p.total_variance(k) + p.total_variance(k - h)) / (h * h);

  assert!(
    (p.w_prime(k) - num_first).abs() < 1e-5,
    "w'(k) mismatch: analytic={} numeric={}",
    p.w_prime(k),
    num_first
  );
  let rel_err = (p.w_double_prime(k) - num_second).abs() / p.w_double_prime(k).abs().max(1e-14);
  assert!(
    rel_err < 1e-3,
    "w''(k) mismatch: analytic={} numeric={} rel_err={}",
    p.w_double_prime(k),
    num_second,
    rel_err
  );
}

#[test]
fn svi_calibration_exact() {
  let true_params = SviRawParams::<f64>::new(0.04, 0.4, -0.4, 0.0, 0.2);
  let ks: Vec<f64> = (-10..=10).map(|i| i as f64 * 0.1).collect();
  let ws: Vec<f64> = ks.iter().map(|&k| true_params.total_variance(k)).collect();

  let fit = calibrate_svi(&ks, &ws, None);
  assert!(
    fit.converged && fit.message.as_deref() == Some("converged"),
    "{fit:?}"
  );
  let fitted = fit.params;

  assert!((fitted.a - true_params.a).abs() < 1e-4, "a mismatch");
  assert!((fitted.b - true_params.b).abs() < 1e-4, "b mismatch");
  assert!((fitted.rho - true_params.rho).abs() < 1e-4, "rho mismatch");
  assert!((fitted.m - true_params.m).abs() < 1e-4, "m mismatch");
  assert!(
    (fitted.sigma - true_params.sigma).abs() < 1e-4,
    "sigma mismatch"
  );
}

#[test]
fn jump_wings_round_trip() {
  let raw = SviRawParams::<f64>::new(0.04, 0.4, -0.4, 0.0, 0.2);
  let jw = raw.jump_wings(1.0);

  assert!(jw.v_t > 0.0);
  assert!(jw.p_t > 0.0);
  assert!(jw.c_t > 0.0);
  assert!(jw.v_tilde > 0.0);
}

#[test]
fn calibration_preserves_the_smile_with_negative_sigma() {
  let truth = SviRawParams::<f64>::new(0.04, 0.4, -0.4, 0.0, 0.2);
  let ks = (-10..=10).map(|i| i as f64 * 0.1).collect::<Vec<_>>();
  let ws = ks
    .iter()
    .map(|&k| truth.total_variance(k))
    .collect::<Vec<_>>();
  let initial = SviRawParams {
    sigma: -0.3,
    ..truth
  };
  let fitted = calibrate_svi(&ks, &ws, Some(initial)).params;
  assert!((fitted.sigma - truth.sigma).abs() < 1e-6);
  for (&k, &w) in ks.iter().zip(&ws) {
    assert!((fitted.total_variance(k) - w).abs() < 1e-10);
  }
}

#[test]
fn calibration_recovers_poorly_conditioned_smiles() {
  for (name, width, truth) in [
    (
      "narrow",
      0.025,
      SviRawParams::new(0.04, 0.2, -0.4, 0.05, 0.15),
    ),
    ("flat", 0.5, SviRawParams::new(0.04, 1e-5, -0.4, 0.05, 0.5)),
    (
      "correlated",
      0.5,
      SviRawParams::new(0.04, 0.2, -0.999, 0.05, 0.05),
    ),
  ] {
    let ks = (-10..=10)
      .map(|i| width * i as f64 / 10.0)
      .collect::<Vec<_>>();
    let ws = ks
      .iter()
      .map(|&k| truth.total_variance(k))
      .collect::<Vec<_>>();
    let fitted = calibrate_svi(&ks, &ws, None).params;
    // Weakly identified parameters can differ while describing the same smile.
    for (&k, &w) in ks.iter().zip(&ws) {
      assert!(
        (fitted.total_variance(k) - w).abs() < 1e-10,
        "{name}: {fitted:?}"
      );
    }
  }
}

/// The unconstrained raw-SVI Jacobian at a start, at the truth and at a far point.
#[test]
fn the_jacobian_is_the_slope_of_the_residuals() {
  use crate::calibration::least_squares::checks::jacobian_error;
  let truth = SviRawParams::<f64>::new(0.04, 0.4, -0.4, 0.1, 0.2);
  let ks = vec![-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4];
  let ws = ks.iter().map(|&k| truth.total_variance(k)).collect();
  let params = truth.into_dvector();
  let mut problem = SviLmProblem { ks, ws, params };
  let points = [
    ndarray::array![0.01, 0.1, 0.0, 0.0, 0.1],
    ndarray::array![0.04, 0.4, -0.4, 0.1, 0.2],
    ndarray::array![0.5, 3.0, 0.9, -1.0, 2.0],
  ];
  for point in points {
    let error = jacobian_error(&mut problem, point.view(), |_, _| 1e-6);
    assert!(error < 1e-8, "{point}: {error:e}");
  }
}
