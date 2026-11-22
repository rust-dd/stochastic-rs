use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

// BSM vanilla call: S=100,K=100,r=0.05,q=0,sigma=0.2,T=1 → 10.4506
const BSM_VANILLA_CALL: f64 = 10.4506;
// BSM vanilla put: S=100,K=100,r=0.05,q=0,sigma=0.2,T=1 → 5.5735
const BSM_VANILLA_PUT: f64 = 5.5735;

fn base(barrier_type: BarrierType, option_type: OptionType, h: f64) -> f64 {
  let option = BarrierOption {
    strike: 100.0,
    barrier: h,
    rebate: 0.0,
    barrier_type,
    option_type,
  };
  BarrierPricer::new(0.2).price(&option, 100.0, 0.05, 0.0, 1.0)
}

fn corridor(lower: f64, upper: f64) -> DoubleBarrierOption {
  DoubleBarrierOption {
    strike: 100.0,
    lower,
    upper,
    option_type: OptionType::Call,
  }
}

#[test]
fn in_out_parity_down_call() {
  let di = base(BarrierType::DownAndIn, OptionType::Call, 90.0);
  let do_ = base(BarrierType::DownAndOut, OptionType::Call, 90.0);
  assert!(
    (di + do_ - BSM_VANILLA_CALL).abs() < 0.1,
    "DI+DO={}",
    di + do_
  );
}

#[test]
fn in_out_parity_up_call() {
  let ui = base(BarrierType::UpAndIn, OptionType::Call, 110.0);
  let uo = base(BarrierType::UpAndOut, OptionType::Call, 110.0);
  assert!(
    (ui + uo - BSM_VANILLA_CALL).abs() < 0.1,
    "UI+UO={}",
    ui + uo
  );
}

#[test]
fn in_out_parity_down_put() {
  let di = base(BarrierType::DownAndIn, OptionType::Put, 90.0);
  let do_ = base(BarrierType::DownAndOut, OptionType::Put, 90.0);
  assert!(
    (di + do_ - BSM_VANILLA_PUT).abs() < 0.1,
    "DI+DO={}",
    di + do_
  );
}

#[test]
fn in_out_parity_up_put() {
  let ui = base(BarrierType::UpAndIn, OptionType::Put, 110.0);
  let uo = base(BarrierType::UpAndOut, OptionType::Put, 110.0);
  assert!((ui + uo - BSM_VANILLA_PUT).abs() < 0.1, "UI+UO={}", ui + uo);
}

#[test]
fn down_and_out_call_haug() {
  // Haug (2007): S=100, K=100, H=90, r=0.05, σ=0.2, T=1 → 8.6655
  let price = base(BarrierType::DownAndOut, OptionType::Call, 90.0);
  assert!((price - 8.6655).abs() < 0.05, "DO call={price}");
}

#[test]
fn knockout_less_than_vanilla() {
  let do_ = base(BarrierType::DownAndOut, OptionType::Call, 90.0);
  let uo = base(BarrierType::UpAndOut, OptionType::Call, 110.0);
  assert!(do_ < BSM_VANILLA_CALL + 0.5, "DO call={do_}");
  assert!(uo < BSM_VANILLA_CALL + 0.5, "UO call={uo}");
  assert!(do_ > 0.0);
  assert!(uo > 0.0);
}

#[test]
fn double_barrier_knockout() {
  let price = DoubleBarrierPricer::new(0.2).price(&corridor(80.0, 120.0), 100.0, 0.05, 0.0, 1.0);
  // Ikeda-Kunitomo series with narrow barriers can overshoot with few terms
  assert!(price > 0.0, "double barrier={price}");
}

#[test]
fn double_barrier_wide_converges_to_vanilla() {
  let price = DoubleBarrierPricer::new(0.2).price(&corridor(1.0, 500.0), 100.0, 0.05, 0.0, 1.0);
  assert!(
    (price - BSM_VANILLA_CALL).abs() < 0.5,
    "wide double barrier={price}"
  );
}

#[test]
#[should_panic(
  expected = "sigma must satisfy `sigma.is_finite() && sigma >= 0.0`, got sigma = -0.25"
)]
fn a_closed_form_barrier_refuses_a_negative_sigma() {
  let _ = BarrierPricer::new(-0.25);
}

/// The four barriers, each unbreached and breached by the spot `100`.
const BARRIERS: [(BarrierType, [f64; 2]); 4] = [
  (BarrierType::UpAndOut, [110.0, 95.0]),
  (BarrierType::UpAndIn, [110.0, 95.0]),
  (BarrierType::DownAndOut, [90.0, 105.0]),
  (BarrierType::DownAndIn, [90.0, 105.0]),
];

/// An infinite `tau` is NaN on every arm, a knocked-out rebate included, where that paid out.
#[test]
fn an_infinite_tau_is_nan_on_every_arm() {
  let mut priced = Vec::new();
  for (barrier_type, barriers) in BARRIERS {
    for barrier in barriers {
      for option_type in [OptionType::Call, OptionType::Put] {
        let option = BarrierOption {
          strike: 100.0,
          barrier,
          rebate: 3.0,
          barrier_type,
          option_type,
        };
        for sigma in [0.2, 0.0] {
          let price = BarrierPricer::new(sigma).price(&option, 100.0, 0.05, 0.02, f64::INFINITY);
          if !price.is_nan() {
            priced.push(format!("{option:?} at sigma {sigma}: {price}"));
          }
        }
      }
    }
  }
  assert!(priced.is_empty(), "{priced:#?}");
}

/// The Monte Carlo is NaN from no path at an infinite `tau`, where a NaN path under its zero
/// discount read `0.0 ± 0.0`, and a path whose payoff is `∞ − ∞` keeps the estimate NaN.
#[test]
fn the_monte_carlo_is_nan_at_an_infinite_tau_and_keeps_a_nan_path() {
  let mc = McBarrierPricer::new(0.2, 64, 8).with_seed(Deterministic::new(7));
  let mut priced = Vec::new();
  for (barrier_type, barriers) in BARRIERS {
    for barrier in barriers {
      for option_type in [OptionType::Call, OptionType::Put] {
        let option = BarrierOption {
          strike: 100.0,
          barrier,
          rebate: 3.0,
          barrier_type,
          option_type,
        };
        let estimate = mc.price(&option, 100.0, 0.05, 0.02, f64::INFINITY);
        if !(estimate.mean.is_nan() && estimate.std_err.is_nan() && estimate.n_samples == 0) {
          priced.push(format!("{option:?}: {estimate:?}"));
        }
      }
    }
  }
  assert!(priced.is_empty(), "{priced:#?}");
  let unbreached = BarrierOption {
    strike: f64::INFINITY,
    barrier: 90.0,
    rebate: 3.0,
    barrier_type: BarrierType::DownAndOut,
    option_type: OptionType::Call,
  };
  let estimate = mc.price(&unbreached, f64::INFINITY, 0.05, 0.02, 1.0);
  assert!(estimate.mean.is_nan(), "{estimate:?}");
}

/// An infinite `tau` is NaN inside the corridor and outside it, as on the single barriers, where
/// the double barrier priced `0`.
#[test]
fn a_double_barrier_is_nan_at_an_infinite_tau() {
  let mut priced = Vec::new();
  for option_type in [OptionType::Call, OptionType::Put] {
    let option = DoubleBarrierOption {
      strike: 100.0,
      lower: 80.0,
      upper: 130.0,
      option_type,
    };
    for s in [100.0, 75.0, 140.0, 80.0] {
      for sigma in [0.2, 0.0] {
        let price = DoubleBarrierPricer::new(sigma).price(&option, s, 0.05, 0.02, f64::INFINITY);
        if !price.is_nan() {
          priced.push(format!("{option:?} at s {s}, sigma {sigma}: {price}"));
        }
      }
    }
  }
  assert!(priced.is_empty(), "{priced:#?}");
}
