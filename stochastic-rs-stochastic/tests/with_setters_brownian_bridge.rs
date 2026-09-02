//! TDD tests for A1-c Task 4: `with_*` builder setters on `BrownianBridge`
//! (`process/brownian_bridge.rs`). No persisted cache: `sampler()` builds
//! its Gaussian source fresh from `self.{n,t,seed}` on every call.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_stochastic::process::brownian_bridge::BrownianBridge;
use stochastic_rs_stochastic::traits::ProcessExt;

#[derive(Debug, PartialEq)]
struct BbFields {
  sigma: f64,
  n: usize,
  x0: Option<f64>,
  xt: Option<f64>,
  t: Option<f64>,
}

fn fields(x: &BrownianBridge<f64>) -> BbFields {
  BbFields {
    sigma: x.sigma,
    n: x.n,
    x0: x.x0,
    xt: x.xt,
    t: x.t,
  }
}

fn finite(out: &Array1<f64>) -> bool {
  out.iter().all(|v| v.is_finite())
}

macro_rules! plain_test {
  ($name:ident, $setter:ident, $field:ident, $val:expr) => {
    #[test]
    fn $name() {
      let expected = BrownianBridge::<f64> {
        $field: $val,
        ..BrownianBridge::<f64>::default()
      };
      let got = BrownianBridge::<f64>::default().$setter($val);
      assert_eq!(got.$field, $val);
      assert_eq!(fields(&got), fields(&expected));
      assert!(finite(&got.sample()));
    }
  };
}

plain_test!(bb_with_sigma_round_trip, with_sigma, sigma, 0.5);
plain_test!(bb_with_x0_round_trip, with_x0, x0, Some(0.5));
plain_test!(bb_with_xt_round_trip, with_xt, xt, Some(-1.0));
plain_test!(bb_with_steps_round_trip, with_steps, n, 64usize);
plain_test!(bb_with_horizon_round_trip, with_horizon, t, Some(2.0));

#[test]
fn bb_with_seed_matches_fresh_construction() {
  let want = BrownianBridge::new(1.0, 64, None, None, Some(1.0), Deterministic::new(13)).sample();
  let got = BrownianBridge::new(1.0, 64, None, None, Some(1.0), Deterministic::new(1))
    .with_seed(Deterministic::new(13))
    .sample();
  assert_eq!(want, got);
}
