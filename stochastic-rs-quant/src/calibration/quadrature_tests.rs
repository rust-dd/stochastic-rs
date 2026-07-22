use super::integrate_gl_to_convergence;

#[test]
fn panel_quadrature_scales_domain_and_resolution() {
  let [decay, oscillatory] = integrate_gl_to_convergence(
    |u| Some([(-0.01 * u).exp(), (-0.01 * u).exp() * (0.7 * u).cos()]),
    1e-8,
  )
  .expect("decaying integrands should converge");

  let expected_decay = 100.0;
  let expected_oscillatory = 0.01 / (0.01_f64.powi(2) + 0.7_f64.powi(2));
  assert!(
    (decay - expected_decay).abs() < 2e-7,
    "decay={decay}, expected={expected_decay}"
  );
  assert!(
    (oscillatory - expected_oscillatory).abs() < 1e-8,
    "oscillatory={oscillatory}, expected={expected_oscillatory}"
  );
}

#[test]
fn panel_quadrature_reports_non_convergence() {
  let result = integrate_gl_to_convergence(|_| Some([1.0]), 1e-8);
  assert!(result.is_none());
}
