//! `with_*` setters on `Vasicek`, whose private `ou: Ou<T, S>` every parameter setter rebuilds
//! with the child seed it holds; only `with_seed` derives again, in `new()`'s order.

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
  vasicek_with_points_rebuilds_ou,
  with_points,
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

/// A `Default`-rooted `with_mu` round-trip. It compares through the `VasicekFields` mirror:
/// `Vasicek` has a private `ou` cache, so a struct-update literal would not compile here.
#[test]
fn vasicek_default_with_mu_round_trip() {
  let base = Vasicek::<f64>::default();
  let got = Vasicek::<f64>::default().with_mu(0.05);
  let expected = VasicekFields {
    mu: 0.05,
    ..fields(&base)
  };
  assert_eq!(got.mu, 0.05);
  assert_eq!(fields(&got), expected);
  assert!(finite(&got.sample()));
}
