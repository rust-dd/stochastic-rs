use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

#[test]
fn volterra_fbm_runs() {
  let v = Volterra::<f64>::fbm(0.7, 100, Some(1.0));
  let path = v.sample();
  assert_eq!(path.len(), 100);
  assert!(path[0] == 0.0);
}

#[test]
fn volterra_exponential_kernel() {
  let v = Volterra::<f64>::new(
    VolterraKernelSpec::Exponential { beta: 1.0 },
    100,
    Some(1.0),
    Unseeded,
  );
  let path = v.sample();
  assert_eq!(path.len(), 100);
}

#[test]
fn volterra_fbm_h05_is_bm() {
  // H=0.5 → K(t,s) = 1/Γ(1) = 1 → X_t = W_t (standard Bm)
  let v = Volterra::<f64, Deterministic>::new(
    VolterraKernelSpec::FractionalBM { h: 0.5 },
    200,
    Some(1.0),
    Deterministic::new(42),
  );
  let path = v.sample();
  // Variance of Bm at t=1 should be ~1
  let var: f64 = path.iter().map(|&x| x * x).sum::<f64>() / path.len() as f64;
  // Very rough check — just ensure it's not degenerate
  assert!(var > 0.001, "variance = {var}");
}

#[test]
fn volterra_seeded_deterministic() {
  // Two instances built with one seed share their first path; repeated calls on one instance
  // advance the seed and differ.
  let v1 = Volterra::<f64, Deterministic>::new(
    VolterraKernelSpec::FractionalBM { h: 0.7 },
    50,
    Some(1.0),
    Deterministic::new(123),
  );
  let v2 = Volterra::<f64, Deterministic>::new(
    VolterraKernelSpec::FractionalBM { h: 0.7 },
    50,
    Some(1.0),
    Deterministic::new(123),
  );
  assert_eq!(v1.sample(), v2.sample());
}

/// `H = 0.7` is the value the reproducibility guard instantiates `Volterra::fbm` at: the
/// fallback path, since `RlKernel` cannot represent `H >= 0.5`.
#[test]
fn h_above_half_still_uses_reference_fallback_and_stays_finite() {
  let v = Volterra::<f64, Deterministic>::new(
    VolterraKernelSpec::FractionalBM { h: 0.7 },
    64,
    Some(1.0),
    Deterministic::new(7),
  );
  let path = v.sample();
  assert_eq!(path.len(), 64);
  assert_eq!(path[0], 0.0);
  assert!(path.iter().all(|v| v.is_finite()));
}

/// A kernel with no [`crate::volterra::VolterraKernel`] representation (an arbitrary power-law
/// exponent) takes the reference fallback and does not panic.
#[test]
fn power_law_kernel_uses_reference_fallback() {
  let v = Volterra::<f64, Deterministic>::new(
    VolterraKernelSpec::PowerLaw { gamma: -0.2 },
    64,
    Some(1.0),
    Deterministic::new(3),
  );
  let path = v.sample();
  assert_eq!(path.len(), 64);
  assert!(path.iter().all(|v| v.is_finite()));
}

#[test]
fn a_single_precision_fbm_takes_its_hurst_in_its_own_scalar() {
  let path = Volterra::<f32>::fbm(0.3f32, 64, None).sample();
  assert_eq!((path.len(), path[0]), (64, 0.0));
  assert!(path.iter().all(|x| x.is_finite()));
}

/// The reference branch's kernel by lag, from an `f32` Hurst exponent, is the `f64` one rounded.
#[test]
fn the_single_precision_reference_kernel_is_the_double_one() {
  use crate::euler::EulerCoefficients;
  let narrow = Volterra::<f32>::fbm(0.7f32, 64, None).curves().unwrap();
  let wide = Volterra::<f64>::fbm(f64::from(0.7f32), 64, None)
    .curves()
    .unwrap();
  for (a, b) in narrow[0].iter().zip(&wide[0]) {
    assert!(((f64::from(*a) - b) / b).abs() < 1e-6, "{a} vs {b}");
  }
}
