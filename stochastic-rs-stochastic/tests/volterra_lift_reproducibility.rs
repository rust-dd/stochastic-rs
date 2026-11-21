//! The four `Rl*` processes' `Deterministic::new(42)` paths against bits captured before the lift
//! went through `VolterraKernel`, whose reassociated arithmetic moves them by up to 512 ULP.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::rough::rl_bs::RlBlackScholes;
use stochastic_rs_stochastic::rough::rl_fbm::RlFbm;
use stochastic_rs_stochastic::rough::rl_fou::RlFou;
use stochastic_rs_stochastic::rough::rl_heston::RlHeston;
use stochastic_rs_stochastic::traits::ProcessExt;

#[path = "volterra_lift_reproducibility/bs.rs"]
mod bs;
#[path = "volterra_lift_reproducibility/fbm.rs"]
mod fbm;
#[path = "volterra_lift_reproducibility/fou.rs"]
mod fou;
#[path = "volterra_lift_reproducibility/heston.rs"]
mod heston;

use bs::BS_SAMPLE;
use fbm::FBM_SAMPLE;
use fou::FOU_SAMPLE;
use heston::HESTON_SAMPLE_S;
use heston::HESTON_SAMPLE_V;

/// Matches `tests/reproducibility_all_processes/rough.rs`'s own `N`.
const N: usize = 24;

/// Relative tolerance, justified in the module doc above against the
/// measured maximum reassociation shift (≈512 ULP / ≈1.05e-13 relative).
const REL_TOL: f64 = 1e-11;

fn assert_close(label: &str, actual: &Array1<f64>, expected: &[u64]) {
  assert_eq!(
    actual.len(),
    expected.len(),
    "{label}: length mismatch (got {}, expected {})",
    actual.len(),
    expected.len()
  );
  for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
    let reference = f64::from_bits(e);
    let tol = REL_TOL * (1.0 + reference.abs());
    let diff = (a - reference).abs();
    assert!(
      diff <= tol,
      "{label}[{i}]: got {a:e} (0x{:016x}), reference {reference:e} (0x{:016x}) — \
       |diff|={diff:e} exceeds tol={tol:e} ({REL_TOL:e} relative)",
      a.to_bits(),
      e
    );
  }
}

#[test]
fn rl_black_scholes_reproducible() {
  let bs = RlBlackScholes::new(
    0.1,
    100.0,
    0.02,
    0.2,
    N,
    Some(1.0),
    None,
    Deterministic::new(42),
  );
  assert_close("RlBlackScholes::sample", &bs.sample(), &BS_SAMPLE);
}

#[test]
fn rl_fbm_reproducible() {
  let fbm = RlFbm::new(0.1, N, Some(1.0), None, Deterministic::new(42));
  assert_close("RlFbm::sample", &fbm.sample(), &FBM_SAMPLE);
}

#[test]
fn rl_fou_reproducible() {
  let fou = RlFou::new(
    0.1,
    1.0,
    0.0,
    0.3,
    N,
    Some(0.0),
    Some(1.0),
    None,
    Deterministic::new(42),
  );
  assert_close("RlFou::sample", &fou.sample(), &FOU_SAMPLE);
}

#[test]
fn rl_heston_reproducible() {
  let heston = RlHeston::new(
    0.1,
    100.0,
    0.04,
    1.5,
    0.04,
    0.3,
    -0.6,
    0.0,
    N,
    Some(1.0),
    None,
    Deterministic::new(42),
  );
  let [s, v] = heston.sample();
  assert_close("RlHeston::sample.s", &s, &HESTON_SAMPLE_S);
  assert_close("RlHeston::sample.v", &v, &HESTON_SAMPLE_V);
}
