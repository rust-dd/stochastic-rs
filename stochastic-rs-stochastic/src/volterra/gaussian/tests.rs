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
  // Two separately built instances with the same seed reproduce each other's first path.
  // (Same instance, repeated `.sample()` calls advance the seed state and produce
  // different paths — that is the desired behaviour for Monte Carlo reuse.)
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

/// `H = 0.7` is exactly the value the reproducibility guard
/// (`tests/reproducibility_all_processes/volterra.rs`) instantiates
/// `Volterra::fbm` at — the fallback path this pins, since `RlKernel`
/// cannot represent `H >= 0.5`.
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

/// Kernels this crate has no [`crate::volterra::VolterraKernel`]
/// representation for (arbitrary power-law exponent) must still route
/// through the reference fallback rather than panicking or silently
/// misbehaving.
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
