//! `CgmysvPricer`'s four entry points off their domain.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::barrier::BarrierType;
use stochastic_rs_quant::pricing::cgmysv::CgmysvParams;
use stochastic_rs_quant::pricing::cgmysv::CgmysvPricer;
use stochastic_rs_stochastic::mc::McEstimate;

const K: f64 = 100.0;
const R: f64 = 0.05;
const Q: f64 = 0.02;
const TAU: f64 = 1.0;
const TYPES: [OptionType; 2] = [OptionType::Call, OptionType::Put];

fn cgmysv(s: f64, r: f64) -> CgmysvPricer<Deterministic> {
  CgmysvPricer {
    params: CgmysvParams {
      alpha: 0.5184,
      lambda_plus: 25.4592,
      lambda_minus: 4.6040,
      kappa: 1.0029,
      eta: 0.0711,
      zeta: 0.3443,
      rho: -2.0283,
      v0: 0.01115,
    },
    s,
    r,
    q: Q,
    n_paths: 16,
    n_steps: 8,
    n_jumps: 16,
    seed: Deterministic::new(7),
  }
}

/// §14: a `0.0` is never a sentinel, so an off-domain query is a NaN estimate from no path.
#[test]
fn the_cgmysv_entry_points_are_nan_for_a_query_off_their_domain() {
  let (nan, inf) = (f64::NAN, f64::INFINITY);
  let mut priced = Vec::new();
  for (s, k, r, tau) in [
    (nan, K, R, TAU),
    (110.0, nan, R, TAU),
    (110.0, K, nan, TAU),
    (0.0, K, R, TAU),
    (-5.0, K, R, TAU),
    (110.0, 0.0, R, TAU),
    (110.0, -5.0, R, TAU),
    (110.0, K, R, inf),
    (110.0, nan, R, 0.0),
  ] {
    for leg in TYPES {
      let model = cgmysv(s, r);
      let estimates: [(&str, McEstimate<f64>); 5] = [
        ("European", model.price_european(k, tau, leg)),
        ("American", model.price_american(k, tau, leg)),
        ("Asian", model.price_asian(k, tau, leg)),
        (
          "up-and-out",
          model.price_barrier(k, tau, 120.0, BarrierType::UpAndOut, leg),
        ),
        (
          "down-and-in",
          model.price_barrier(k, tau, 105.0, BarrierType::DownAndIn, leg),
        ),
      ];
      for (what, estimate) in estimates {
        if !(estimate.mean.is_nan() && estimate.std_err.is_nan() && estimate.n_samples == 0) {
          priced.push(format!(
            "{what} {leg:?} at ({s}, {k}, {r}, {tau}): {estimate}"
          ));
        }
      }
    }
  }
  assert!(priced.is_empty(), "{priced:#?}");
  let barrier = cgmysv(110.0, R).price_barrier(K, TAU, nan, BarrierType::UpAndOut, TYPES[0]);
  assert!(barrier.mean.is_nan() && barrier.n_samples == 0, "{barrier}");
}
