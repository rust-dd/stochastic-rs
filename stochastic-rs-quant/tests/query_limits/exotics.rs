//! The exotic pricers at expiry, past their barriers and outside their domain.

use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::barrier::BarrierPricer;
use stochastic_rs_quant::pricing::barrier::BarrierType;
use stochastic_rs_quant::pricing::barrier::DoubleBarrierPricer;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::pricing::exotic::BarrierOption;
use stochastic_rs_quant::pricing::exotic::DoubleBarrierOption;
use stochastic_rs_quant::pricing::exotic::LookbackOption;
use stochastic_rs_quant::pricing::lookback::LookbackPricer;
use stochastic_rs_quant::pricing::variance_swap::VarianceSwapPricer;
use stochastic_rs_quant::pricing::variance_swap::VolatilitySwapPricer;
use stochastic_rs_quant::pricing::variance_swap::replication_weights;
use stochastic_rs_quant::traits::ModelPricer;

const R: f64 = 0.05;
const Q: f64 = 0.02;
const SIGMA: f64 = 0.25;

fn barrier(barrier_type: BarrierType, option_type: OptionType, barrier: f64) -> BarrierOption {
  BarrierOption {
    strike: 100.0,
    barrier,
    rebate: 3.0,
    barrier_type,
    option_type,
  }
}

fn lookback(
  strike: Option<f64>,
  extremes: (Option<f64>, Option<f64>),
  call: bool,
) -> LookbackOption {
  LookbackOption {
    strike,
    running_min: extremes.0,
    running_max: extremes.1,
    option_type: if call {
      OptionType::Call
    } else {
      OptionType::Put
    },
  }
}

#[test]
fn an_expired_barrier_pays_its_payoff_or_never_knocked_in_pays_its_rebate() {
  use BarrierType::*;
  use OptionType::*;
  let pricer = BarrierPricer::new(SIGMA);
  for s in [90.0, 100.0, 110.0] {
    for option_type in [Call, Put] {
      let payoff = match option_type {
        Call => (s - 100.0_f64).max(0.0),
        Put => (100.0_f64 - s).max(0.0),
      };
      for (barrier_type, h) in [(UpAndOut, 130.0), (DownAndOut, 80.0)] {
        let option = barrier(barrier_type, option_type, h);
        assert_eq!(
          pricer.price(&option, s, R, Q, 0.0),
          payoff,
          "{barrier_type:?} at {s}"
        );
      }
      for (barrier_type, h) in [(UpAndIn, 130.0), (DownAndIn, 80.0)] {
        let option = barrier(barrier_type, option_type, h);
        assert_eq!(
          pricer.price(&option, s, R, Q, 0.0),
          3.0,
          "{barrier_type:?} at {s}"
        );
      }
    }
  }
}

#[test]
fn a_breached_barrier_has_knocked_out_to_its_rebate_or_in_to_the_vanilla() {
  use BarrierType::*;
  use OptionType::*;
  let pricer = BarrierPricer::new(SIGMA);
  let vanilla = BsmPricer::new(SIGMA, BsmCoc::Merton1973);
  for tau in [0.0, 0.5, 1.0] {
    for (barrier_type, h, s) in [(DownAndOut, 80.0, 75.0), (UpAndOut, 130.0, 135.0)] {
      for option_type in [Call, Put] {
        let option = BarrierOption {
          rebate: 0.0,
          ..barrier(barrier_type, option_type, h)
        };
        assert_eq!(
          pricer.price(&option, s, R, Q, tau),
          0.0,
          "{barrier_type:?} {tau}"
        );
        let with_rebate = barrier(barrier_type, option_type, h);
        assert_eq!(pricer.price(&with_rebate, s, R, Q, tau), 3.0);
        assert_eq!(
          pricer.price(&with_rebate, h, R, Q, tau),
          3.0,
          "on the barrier"
        );
      }
    }
    for (barrier_type, h, s) in [(DownAndIn, 80.0, 75.0), (UpAndIn, 130.0, 135.0)] {
      for option_type in [Call, Put] {
        let option = barrier(barrier_type, option_type, h);
        let knocked_in = vanilla.price_option(s, 100.0, R, Q, tau, option_type);
        assert_eq!(
          pricer.price(&option, s, R, Q, tau),
          knocked_in,
          "{barrier_type:?}"
        );
      }
    }
  }
}

#[test]
fn the_double_barrier_is_dead_outside_its_corridor_and_pays_its_payoff_at_expiry() {
  let pricer = DoubleBarrierPricer::new(SIGMA);
  for option_type in [OptionType::Call, OptionType::Put] {
    let option = DoubleBarrierOption {
      strike: 100.0,
      lower: 70.0,
      upper: 130.0,
      option_type,
    };
    for s in [60.0, 70.0, 130.0, 140.0] {
      for tau in [0.0, 0.25, 1.0] {
        assert_eq!(
          pricer.price(&option, s, R, Q, tau),
          0.0,
          "{option_type:?} {s} {tau}"
        );
      }
    }
    for s in [90.0, 100.0, 110.0] {
      let payoff = match option_type {
        OptionType::Call => (s - 100.0_f64).max(0.0),
        OptionType::Put => (100.0 - s).max(0.0),
      };
      assert_eq!(
        pricer.price(&option, s, R, Q, 0.0),
        payoff,
        "{option_type:?} {s}"
      );
    }
  }
}

#[test]
fn an_expired_lookback_pays_off_on_its_running_extremes() {
  let pricer = LookbackPricer::new(SIGMA);
  for (option, payoff) in [
    (lookback(None, (None, None), true), 0.0),
    (lookback(None, (Some(90.0), None), true), 10.0),
    (lookback(None, (None, None), false), 0.0),
    (lookback(None, (None, Some(110.0)), false), 10.0),
    (lookback(Some(90.0), (None, None), true), 10.0),
    (lookback(Some(105.0), (None, Some(110.0)), true), 5.0),
    (lookback(Some(110.0), (None, None), false), 10.0),
    (lookback(Some(95.0), (Some(90.0), None), false), 5.0),
  ] {
    assert_eq!(
      pricer.price(&option, 100.0, R, Q, 0.0),
      payoff,
      "{option:?}"
    );
  }
}

#[test]
fn an_exotic_answers_a_negative_or_nan_tau_or_a_nonpositive_spot_with_nan() {
  let barrier_option = barrier(BarrierType::DownAndOut, OptionType::Put, 80.0);
  let barrier_pricer = BarrierPricer::new(SIGMA);
  let corridor = DoubleBarrierOption {
    strike: 100.0,
    lower: 70.0,
    upper: 130.0,
    option_type: OptionType::Call,
  };
  let double = DoubleBarrierPricer::new(SIGMA);
  let lookback_pricer = LookbackPricer::new(SIGMA);
  let floating = lookback(None, (None, None), true);
  for (s, tau) in [
    (100.0, -1.0),
    (100.0, f64::NAN),
    (0.0, 1.0),
    (-1.0, 1.0),
    (f64::NAN, 1.0),
  ] {
    assert!(
      barrier_pricer.price(&barrier_option, s, R, Q, tau).is_nan(),
      "{s} {tau}"
    );
    assert!(double.price(&corridor, s, R, Q, tau).is_nan(), "{s} {tau}");
    assert!(
      lookback_pricer.price(&floating, s, R, Q, tau).is_nan(),
      "{s} {tau}"
    );
  }
  assert!(
    barrier_pricer
      .price(&barrier_option, 75.0, f64::NAN, Q, 1.0)
      .is_nan()
  );
  assert!(double.price(&corridor, 100.0, R, f64::NAN, 1.0).is_nan());
}

#[test]
fn the_variance_swap_strikes_answer_a_maturity_outside_their_domain_with_nan() {
  let strikes = [90.0, 100.0, 110.0];
  let prices = [1.0, 2.0, 1.0];
  for tau in [0.0, -1.0, f64::NAN] {
    let replicated =
      VarianceSwapPricer::fair_strike_replication(&strikes, &prices, 100.0, R, Q, tau);
    assert!(replicated.is_nan(), "{tau}");
    assert!(
      replication_weights(&strikes, tau)
        .iter()
        .all(|w| w.is_nan()),
      "{tau}"
    );
  }
  for tau in [-1.0, f64::NAN] {
    assert!(VarianceSwapPricer::fair_strike_heston(0.04, 1.5, 0.06, tau).is_nan());
    assert!(VolatilitySwapPricer::fair_strike_heston(0.04, 1.5, 0.06, 0.3, tau).is_nan());
  }
  assert_eq!(
    VarianceSwapPricer::fair_strike_heston(0.04, 1.5, 0.06, 0.0),
    0.04
  );
}
