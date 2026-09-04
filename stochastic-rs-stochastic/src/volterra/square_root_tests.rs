use stochastic_rs_core::simd_rng::Deterministic;

use super::*;
use crate::rough::kernel::RlKernel;

/// Parameters that violate the Feller condition hard: $2\kappa\theta = 0.02$
/// against $\nu^2 = 0.25$, so the process is pushed onto the boundary
/// repeatedly rather than incidentally. Any nonnegativity claim that only
/// holds at comfortable parameters is worthless, so everything here runs at
/// these.
const KAPPA: f64 = 0.5;
const THETA: f64 = 0.02;
const NU: f64 = 0.5;
const HURST: f64 = 0.1;
const N: usize = 256;
const PATHS: usize = 400;

fn kernel() -> RlKernel<f64> {
  RlKernel::<f64>::new(HURST, 32)
}

fn process(seed: u64) -> VolterraSquareRoot<f64, RlKernel<f64>, Deterministic> {
  VolterraSquareRoot::new(
    kernel(),
    KAPPA,
    THETA,
    NU,
    N,
    Some(THETA),
    Some(1.0),
    Deterministic::new(seed),
  )
}

/// The type's entire reason to exist: the output is nonnegative at parameters
/// where the naive scheme is not. Checked across many paths, not one, because
/// a single path can easily avoid the boundary by luck.
#[test]
fn output_is_nonnegative_under_a_violated_feller_condition() {
  let paths = process(42).sample_par(PATHS);
  let mut minimum = f64::INFINITY;
  for path in &paths {
    for &v in path.iter() {
      assert!(v.is_finite(), "path produced a non-finite value: {v}");
      minimum = minimum.min(v);
    }
  }
  assert!(
    minimum >= 0.0,
    "a variance path went negative (min = {minimum:e}) — the truncation is not doing its job"
  );
}

/// The guard above is only meaningful if the thing it guards against actually
/// happens here. This runs the same lift, kernel and parameters, with the
/// coefficients evaluated on the raw state instead of
/// $V^+$ — and asserts it **does** go negative. The seed is *not* bit-identical
/// to the paired test's: `sampler()` builds its `Gn` from `seed.derive()` and
/// this constructs one directly, so the draws differ. That is immaterial here —
/// 400 independent paths under a 12.5x Feller violation go negative under
/// either derivation — but the claim is "same model, unprotected", not "same
/// numbers". If this ever stops failing, the nonnegativity test above has
/// become vacuous and both need revisiting.
#[test]
fn the_untruncated_scheme_does_go_negative_at_the_same_parameters() {
  use crate::noise::gn::Gn;
  use crate::volterra::lift::VolterraLift;

  let dt = 1.0 / (N - 1) as f64;
  let lift = VolterraLift::new(kernel(), dt);
  let gn = Gn::<f64, Deterministic> {
    backend: Cpu,
    n: N - 1,
    t: Some(1.0),
    seed: Deterministic::new(42),
  };

  let mut went_negative = false;
  for _ in 0..PATHS {
    let dw = gn.sample();
    // No `max(0)` anywhere: the raw state feeds both coefficients. `sqrt` of a
    // negative state yields NaN, which is itself a way of failing.
    let path = lift.simulate(
      THETA,
      |_, v| KAPPA * (THETA - v),
      |_, v| NU * v.sqrt(),
      dw.as_slice().unwrap(),
    );
    if path.iter().any(|&v| v < 0.0 || v.is_nan()) {
      went_negative = true;
      break;
    }
  }
  assert!(
    went_negative,
    "the untruncated scheme stayed nonnegative across {PATHS} paths, so the \
     nonnegativity guarantee is untested — pick parameters that stress the boundary"
  );
}

/// Two identically seeded instances must agree bit-for-bit, jumps of the
/// truncation included.
#[test]
fn is_seed_reproducible() {
  let a = process(7).sample();
  let b = process(7).sample();
  assert_eq!(a, b);
}

/// A different seed must give a different path, or the seed is not reaching
/// the output.
#[test]
fn different_seeds_give_different_paths() {
  let a = process(7).sample();
  let b = process(8).sample();
  assert_ne!(a, b);
}

/// `v0` defaults to `theta`, the stationary level — not to zero, which would
/// start the process on its boundary.
#[test]
fn v0_defaults_to_theta() {
  let p = VolterraSquareRoot::new(
    kernel(),
    KAPPA,
    THETA,
    NU,
    N,
    None,
    Some(1.0),
    Deterministic::new(1),
  );
  assert_eq!(p.sample()[0], THETA);
}

#[test]
#[should_panic(expected = "kappa must be strictly positive")]
fn rejects_a_nonpositive_kappa() {
  let _ = process(1).with_kappa(0.0);
}

#[test]
#[should_panic(expected = "nu must be strictly positive")]
fn rejects_a_nonpositive_nu() {
  let _ = process(1).with_nu(0.0);
}

#[test]
#[should_panic(expected = "theta must be non-negative")]
fn rejects_a_negative_theta() {
  let _ = process(1).with_theta(-1.0);
}

#[test]
#[should_panic(expected = "v0 must be non-negative")]
fn rejects_a_negative_v0() {
  let _ = process(1).with_v0(-1.0);
}

#[test]
#[should_panic(expected = "n must be at least 2")]
fn rejects_n_below_two() {
  let _ = VolterraSquareRoot::new(
    kernel(),
    KAPPA,
    THETA,
    NU,
    1,
    None,
    None,
    Deterministic::new(1),
  );
}
