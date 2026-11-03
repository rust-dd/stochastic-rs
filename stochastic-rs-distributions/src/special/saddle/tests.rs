use super::*;

/// mpmath at 60 and 120 digits on both sides of the series' edge `|s| = 0.2` (`x = −1/3`, `½`) and at small `x`.
#[test]
fn ln_1p_mx_holds_both_forms() {
  for (x, want) in [
    (1e-10, -4.999999999666667e-21),
    (-0.3, -0.056674943938732375),
    (-0.34, -0.07551544396166583),
    (0.49, -0.09122388004263222),
    (0.51, -0.09789034917316704),
    (3.0, -1.6137056388801094),
  ] {
    let got = ln_1p_mx(x);
    assert!(
      (got - want).abs() <= 4.0 * f64::EPSILON * want.abs(),
      "{x}: {got:e}, want {want:e}"
    );
  }
}

/// The saddle forms equal the scaled `K`s' logarithms where those hold, at order 60.
#[test]
fn the_saddle_forms_meet_the_scaled_ks() {
  let (lambda, chi, psi) = (60.5_f64, 0.7_f64, 1.3_f64);
  let omega = (chi * psi).sqrt();
  let ln_n = |l: f64| 0.5 * l * (chi / psi).ln() + super::super::ln_bessel_ke(l, omega) - omega;
  let step = ln_order_step(lambda, 2.0, (chi, psi));
  assert!(
    (step - (ln_n(lambda + 2.0) - ln_n(lambda))).abs() <= 1e-13,
    "{step}"
  );
  let shifted = psi - 2.0 * 0.4;
  let ln_ns = 0.5 * lambda * (chi / shifted).ln()
    + super::super::ln_bessel_ke(lambda, (chi * shifted).sqrt())
    - (chi * shifted).sqrt();
  let shift = ln_shift(lambda, 0.4, (chi, psi, shifted));
  assert!((shift - (ln_ns - ln_n(lambda))).abs() <= 1e-13, "{shift}");
}
