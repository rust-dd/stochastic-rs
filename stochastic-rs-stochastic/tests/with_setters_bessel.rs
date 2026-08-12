//! TDD tests for A1-c Task 4: `with_*` builder setters on `Bessel` and
//! `SquaredBessel` (`diffusion/bessel.rs`, one source file for both). No
//! persisted cache on either type: `sampler()` builds its Gaussian source
//! fresh from `self.{n,t,seed}` on every call.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::diffusion::bessel::Bessel;
use stochastic_rs_stochastic::diffusion::bessel::SquaredBessel;
use stochastic_rs_stochastic::traits::ProcessExt;

#[derive(Debug, PartialEq)]
struct BesselFields {
  delta: f64,
  n: usize,
  x0: Option<f64>,
  t: Option<f64>,
  use_sym: Option<bool>,
}
fn bes_fields(x: &Bessel<f64>) -> BesselFields {
  BesselFields {
    delta: x.delta,
    n: x.n,
    x0: x.x0,
    t: x.t,
    use_sym: x.use_sym,
  }
}
fn besq_fields(x: &SquaredBessel<f64>) -> BesselFields {
  BesselFields {
    delta: x.delta,
    n: x.n,
    x0: x.x0,
    t: x.t,
    use_sym: x.use_sym,
  }
}
fn finite(out: &Array1<f64>) -> bool {
  out.iter().all(|v| v.is_finite())
}

macro_rules! bes_plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let expected = Bessel::<f64> {
        $field: $val,
        ..Bessel::<f64>::default()
      };
      let got = Bessel::<f64>::default().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(bes_fields(&got), bes_fields(&expected));
      assert!(finite(&got.sample()));
    }
  };
}
macro_rules! besq_plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let expected = SquaredBessel::<f64> {
        $field: $val,
        ..SquaredBessel::<f64>::default()
      };
      let got = SquaredBessel::<f64>::default().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(besq_fields(&got), besq_fields(&expected));
      assert!(finite(&got.sample()));
    }
  };
}

bes_plain_test!(bessel_with_delta_round_trip, with_delta, delta, 4.0);
bes_plain_test!(bessel_with_x0_round_trip, with_x0, x0, Some(2.0));
bes_plain_test!(
  bessel_with_use_sym_round_trip,
  with_use_sym,
  use_sym,
  Some(true)
);
bes_plain_test!(bessel_with_steps_round_trip, with_steps, n, 64usize);
bes_plain_test!(bessel_with_horizon_round_trip, with_horizon, t, Some(2.0));

#[test]
fn bessel_with_seed_matches_fresh_construction() {
  let want = Bessel::new(3.0, 64, Some(1.0), Some(1.0), None, Deterministic::new(13)).sample();
  let got = Bessel::new(3.0, 64, Some(1.0), Some(1.0), None, Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}

besq_plain_test!(besq_with_delta_round_trip, with_delta, delta, 4.0);
besq_plain_test!(besq_with_x0_round_trip, with_x0, x0, Some(2.0));
besq_plain_test!(
  besq_with_use_sym_round_trip,
  with_use_sym,
  use_sym,
  Some(true)
);
besq_plain_test!(besq_with_steps_round_trip, with_steps, n, 64usize);
besq_plain_test!(besq_with_horizon_round_trip, with_horizon, t, Some(2.0));

#[test]
fn besq_with_seed_matches_fresh_construction() {
  let want =
    SquaredBessel::new(3.0, 64, Some(1.0), Some(1.0), None, Deterministic::new(13)).sample();
  let got = SquaredBessel::new(3.0, 64, Some(1.0), Some(1.0), None, Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
