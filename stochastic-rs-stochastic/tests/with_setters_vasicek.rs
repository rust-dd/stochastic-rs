//! TDD tests for A1-c Task 4: `with_*` builder setters on `Vasicek`
//! (`interest/vasicek.rs`). Cache: private `ou: Ou<T, S>` — `Vasicek`
//! delegates its entire sampler to an embedded, fully-parameterized `Ou`.
//! Every setter that touches `theta`/`mu`/`sigma`/`n`/`x0`/`t` must rebuild
//! `ou`.
//!
//! **Different from every Task 2/this-task `Cgns`-cache case**: `new()`
//! builds `ou` via `Ou::new(.., seed.derive())` — a *one-time* derive off
//! the constructor's own `seed` argument, whose result is then stored
//! permanently in `ou.seed` (itself `pub`). Rebuilding `ou` inside a setter
//! by calling `self.seed.derive()` *again* would derive a second, different
//! child from the (already-once-derived-from) `self.seed`, which would
//! *not* match what a fresh `Vasicek::new(new_field, .., seed)` produces
//! (that fresh call derives its *first* child from an unadvanced `seed`).
//! The fix: reuse `self.ou.seed.clone()` (the already-fixed derived seed)
//! when rebuilding `ou` for any setter *other than* `with_seed`; only
//! `with_seed` re-derives, from the *new* outer seed, exactly mirroring
//! `new()`'s own construction order. The bit-exact tests below are exactly
//! what would fail if this distinction were gotten wrong.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::interest::vasicek::Vasicek;
use stochastic_rs_stochastic::traits::ProcessExt;

fn vasicek_base_seeded<S: SeedExt>(seed: S) -> Vasicek<f64, S> {
  Vasicek::new(3.0, 0.03, 0.02, 64, Some(0.03), Some(1.0), seed)
}
fn vasicek_base() -> Vasicek<f64> {
  vasicek_base_seeded(Unseeded)
}

#[derive(Debug, PartialEq)]
struct VasicekFields {
  theta: f64,
  mu: f64,
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
}
fn fields<S: SeedExt>(x: &Vasicek<f64, S>) -> VasicekFields {
  VasicekFields {
    theta: x.theta,
    mu: x.mu,
    sigma: x.sigma,
    n: x.n,
    x0: x.x0,
    t: x.t,
  }
}
fn finite(out: &Array1<f64>) -> bool {
  out.iter().all(|v| v.is_finite())
}

macro_rules! cache_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr, $new_args:expr) => {
    #[test]
    fn $name() {
      let mut expected = vasicek_base();
      expected.$field = $val;
      let got = vasicek_base().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite(&got.sample()));

      let (theta, mu, sigma, n, x0, t): (f64, f64, f64, usize, Option<f64>, Option<f64>) =
        $new_args;
      let want = Vasicek::new(theta, mu, sigma, n, x0, t, Deterministic::new(7)).sample();
      let got_seeded = vasicek_base_seeded(Deterministic::new(7))
        .$setter($val)
        .sample();
      assert_eq!(
        want, got_seeded,
        "with_* must rebuild the embedded Ou (reusing its already-derived \
         seed), not just the mirrored scalar field"
      );
    }
  };
}

cache_test!(
  vasicek_with_theta_rebuilds_ou,
  with_theta,
  theta,
  1.0,
  (1.0, 0.03, 0.02, 64, Some(0.03), Some(1.0))
);
cache_test!(
  vasicek_with_mu_rebuilds_ou,
  with_mu,
  mu,
  0.05,
  (3.0, 0.05, 0.02, 64, Some(0.03), Some(1.0))
);
cache_test!(
  vasicek_with_sigma_rebuilds_ou,
  with_sigma,
  sigma,
  0.05,
  (3.0, 0.03, 0.05, 64, Some(0.03), Some(1.0))
);
cache_test!(
  vasicek_with_x0_rebuilds_ou,
  with_x0,
  x0,
  Some(0.06),
  (3.0, 0.03, 0.02, 64, Some(0.06), Some(1.0))
);
cache_test!(
  vasicek_with_steps_rebuilds_ou,
  with_steps,
  n,
  128,
  (3.0, 0.03, 0.02, 128, Some(0.03), Some(1.0))
);
cache_test!(
  vasicek_with_horizon_rebuilds_ou,
  with_horizon,
  t,
  Some(2.0),
  (3.0, 0.03, 0.02, 64, Some(0.03), Some(2.0))
);

#[test]
fn vasicek_with_seed_matches_fresh_construction() {
  let want = vasicek_base_seeded(Deterministic::new(13)).sample();
  let got = vasicek_base_seeded(Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
