//! Query limits: NaN outside a pricer's domain and the payoff at `tau = 0`, never a sentinel.

use chrono::NaiveDate;
use ndarray::Array1;
use stochastic_rs_quant::CallPut;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::QuantError;
use stochastic_rs_quant::calendar::Frequency;
use stochastic_rs_quant::cashflows::RateTenor;
use stochastic_rs_quant::credit::HazardInterpolation;
use stochastic_rs_quant::credit::SurvivalCurve;
use stochastic_rs_quant::credit::SurvivalPoint;
use stochastic_rs_quant::curves::BootstrapInstrument;
use stochastic_rs_quant::curves::BootstrapSpec;
use stochastic_rs_quant::curves::CurvePoint;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::curves::IterativeOptions;
use stochastic_rs_quant::curves::MultiCurve;
use stochastic_rs_quant::curves::bootstrap;
use stochastic_rs_quant::market::RateHelper;
use stochastic_rs_quant::market::build_curve;
use stochastic_rs_quant::pricing::AssetOrNothingPricer;
use stochastic_rs_quant::pricing::CashOrNothingPricer;
use stochastic_rs_quant::pricing::GapPricer;
use stochastic_rs_quant::pricing::SuperSharePricer;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::heston::HestonPricer;
use stochastic_rs_quant::pricing::merton_jump::Merton1976Pricer;
use stochastic_rs_quant::pricing::sabr::SabrParams;
use stochastic_rs_quant::pricing::sabr::alpha_from_atm_vol;
use stochastic_rs_quant::pricing::sabr::hagan_implied_vol;
use stochastic_rs_quant::pricing::snell_envelope::SnellEnvelopePricer;
use stochastic_rs_quant::traits::Greeks;
use stochastic_rs_quant::traits::ModelPricer;

const R: f64 = 0.05;
const Q: f64 = 0.01;
const SPOTS: [f64; 3] = [90.0, 100.0, 110.0];
const OPTION_TYPES: [OptionType; 2] = [OptionType::Call, OptionType::Put];

fn heston() -> HestonPricer {
  HestonPricer::new(HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  })
}

fn assert_all_nan(greeks: Greeks, what: &str) {
  for (name, value) in Greeks::COMPONENT_NAMES.iter().zip(greeks.as_array()) {
    assert!(value.is_nan(), "{what}: {name} must be NaN, got {value}");
  }
}

#[test]
fn the_snell_lattice_answers_nan_when_the_crr_probability_leaves_the_unit_interval() {
  // dt = 0.5, u = e^{0.2 sqrt(0.5)} = 1.152, d = 0.868, growth = e^{0.25} = 1.284: p = 1.46.
  let coarse = SnellEnvelopePricer::new(0.2, 2);
  assert!(coarse.price_call(100.0, 100.0, 0.5, 0.0, 1.0).is_nan());
  let detailed = coarse.price_detailed(100.0, 100.0, 0.5, 0.0, 1.0, OptionType::Call);
  assert!(detailed.price.is_nan() && detailed.european_price.is_nan());
  assert!(detailed.exercise_boundary.is_empty());
  let fine = SnellEnvelopePricer::new(0.2, 400);
  assert!(fine.price_call(100.0, 100.0, 0.5, 0.0, 1.0).is_finite());
}

#[test]
#[should_panic(expected = "steps must satisfy `steps >= 1`, got steps = 0")]
fn a_snell_lattice_without_a_step_is_a_precondition() {
  SnellEnvelopePricer::new(0.2, 0);
}

#[test]
fn the_discount_curve_has_no_zero_sentinels() {
  let curve = DiscountCurve::<f64>::from_zero_rates(
    Array1::from_vec(vec![1.0, 2.0, 5.0]).view(),
    Array1::from_vec(vec![0.03; 3]).view(),
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  assert!(curve.simple_forward_rate(2.0, 1.0).is_nan());
  assert!(curve.simple_forward_rate(1.0, 1.0).is_nan());
  assert!((curve.simple_forward_rate(1.0, 2.0) - (0.03_f64.exp() - 1.0)).abs() < 1e-12);
  assert!(curve.forward_rate(2.0, 1.0).is_nan());
  assert!(curve.forward_rate(1.0, 1.0).is_nan());
  assert!((curve.forward_rate(1.0, 2.0) - 0.03).abs() < 1e-12);
  assert!(curve.forward_rate(f64::NAN, 2.0).is_nan() && curve.forward_rate(1.0, f64::NAN).is_nan());
  assert_eq!(
    curve
      .forward_rates(Array1::from_vec(vec![1.0]).view())
      .len(),
    0
  );
  assert_eq!(
    curve.forward_rates(Array1::from_vec(vec![]).view()).len(),
    0
  );
  let forwards = curve.forward_rates(Array1::from_vec(vec![1.0, 2.0, 2.0]).view());
  assert!((forwards[0] - 0.03).abs() < 1e-12 && forwards[1].is_nan());
  assert!(curve.par_rate(0.0, Frequency::SemiAnnual).is_nan());
  assert!(curve.par_rate(-1.0, Frequency::SemiAnnual).is_nan());
  assert!(
    curve
      .par_rate(f64::INFINITY, Frequency::SemiAnnual)
      .is_nan()
  );
}

#[test]
#[should_panic(expected = "points must satisfy `p.time.is_finite()` for every p, got p.time = NaN")]
fn a_discount_curve_pillar_at_a_nan_time_is_a_precondition() {
  let point = |time, discount_factor| CurvePoint {
    time,
    discount_factor,
  };
  let method = InterpolationMethod::LogLinearOnDiscountFactors;
  DiscountCurve::new(vec![point(1.0, 0.97), point(f64::NAN, 0.95)], method);
}

#[test]
#[should_panic(expected = "points must satisfy `p.time.is_finite()` for every p, got p.time = NaN")]
fn a_survival_curve_node_at_a_nan_time_is_a_precondition() {
  let point = |time, survival_probability| SurvivalPoint {
    time,
    survival_probability,
  };
  let method = HazardInterpolation::PiecewiseConstantHazard;
  SurvivalCurve::new(vec![point(1.0, 0.98), point(f64::NAN, 0.95)], method);
}

#[test]
fn alpha_from_atm_vol_is_nan_for_query_data_outside_its_domain_and_panics_only_on_rho() {
  assert!(alpha_from_atm_vol(0.2, -1.0, 1.0, 0.5, -0.3, 0.6).is_nan());
  assert!(alpha_from_atm_vol(0.0, 100.0, 1.0, 0.5, -0.3, 0.6).is_nan());
  assert!(alpha_from_atm_vol(0.2, 100.0, -1.0, 0.5, -0.3, 0.6).is_nan());
  assert!(alpha_from_atm_vol(0.2, 100.0, f64::INFINITY, 0.5, -0.3, 0.6).is_nan());
  assert!(alpha_from_atm_vol(0.2, 100.0, f64::NAN, 0.5, -0.3, 0.6).is_nan());
  assert!(alpha_from_atm_vol(0.2, 100.0, 1.0, 0.5, -0.3, 0.6).is_finite());
  // β = 1, ν = 0: α is σ_ATM itself, however small.
  assert_eq!(alpha_from_atm_vol(1e-9, 1.0, 1.0, 1.0, 0.0, 0.0), 1e-9);
  let err = std::panic::catch_unwind(|| alpha_from_atm_vol(0.2, 100.0, 1.0, 0.5, 1.0, 0.6));
  let err = err.unwrap_err();
  let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
  assert_eq!(
    msg,
    "rho must satisfy `rho > -1.0 && rho < 1.0`, got rho = 1.0"
  );
}

#[test]
fn alpha_from_atm_vol_is_the_smallest_positive_root_and_nan_without_one() {
  let close = |got: f64, want: f64| (got - want).abs() <= 8.0 * f64::EPSILON * want;
  // Newton from α₀ ends at a negative root of 12.375α² − 18.59α − 0.2 and of a β = 0.5 cubic.
  let alpha = alpha_from_atm_vol(0.2, 100.0, 5.0, 1.0, 0.99, 10.0);
  assert!(close(alpha, 1.512871295794398), "{alpha}");
  let alpha = alpha_from_atm_vol(0.2, 100.0, 30.0, 0.5, 0.95, 5.0);
  assert!(close(alpha, 11.706371823748876), "{alpha}");
  // Roots 0.05 and 1/15, where Newton from α₀ = 0.1 converges to the larger.
  let alpha = alpha_from_atm_vol(0.1, 0.05, 30.0, 1.0, -0.8, 5.0);
  assert!(close(alpha, 0.05000000000000115), "{alpha}");
  // Newton stops short of a root: at the cubic's one positive root, and NaN with none.
  let alpha = alpha_from_atm_vol(0.5, 1.0, 30.0, 0.2, -0.99, 1.0);
  assert!(close(alpha, 2.1019952276218388), "{alpha}");
  assert!(alpha_from_atm_vol(0.5, 0.05, 30.0, 1.0, -0.99, 0.1).is_nan());
  assert!(alpha_from_atm_vol(0.2, 100.0, 5.0, 1.0, -0.99, 10.0).is_nan());
}

#[test]
fn alpha_from_atm_vol_is_nan_where_its_root_does_not_reprice_the_atm_vol() {
  // β = 0.99: the cubic's one positive root, near 5.9e5, sums terms that cancel to σ_ATM.
  assert!(alpha_from_atm_vol(0.05, 100.0, 30.0, 0.99, -0.95, 10.0).is_nan());
  for (v, f, tau, beta, rho, nu) in [
    (0.2, 100.0, 5.0, 1.0, 0.99, 10.0),
    (0.2, 100.0, 30.0, 0.5, 0.95, 5.0),
  ] {
    let alpha = alpha_from_atm_vol(v, f, tau, beta, rho, nu);
    let params = SabrParams {
      alpha,
      beta,
      rho,
      nu,
    };
    let atm = hagan_implied_vol(f, f, tau, &params);
    assert!(
      (atm - v).abs() <= f64::EPSILON.sqrt() * v,
      "{atm} at {alpha}"
    );
  }
}

#[test]
fn prices_at_zero_maturity_are_intrinsic_and_greeks_are_nan() {
  let bsm = BsmPricer::new(0.2, BsmCoc::Merton1973);
  let merton = Merton1976Pricer::new(0.2, 0.5, 0.3, 50, BsmCoc::Merton1973);
  let heston = heston();
  let k = 100.0;
  for s in SPOTS {
    let bsm_pair = CallPut {
      call: bsm.price_call(s, k, R, Q, 0.0),
      put: bsm.price_put(s, k, R, Q, 0.0),
    };
    let prices = [
      ("bsm", bsm_pair),
      ("merton", merton.price_call_put(s, k, R, Q, 0.0)),
      ("heston", heston.price_call_put(s, k, R, Q, 0.0)),
    ];
    for (name, CallPut { call, put }) in prices {
      assert_eq!(call, (s - k).max(0.0), "{name} call at s = {s}");
      assert_eq!(put, (k - s).max(0.0), "{name} put at s = {s}");
    }
    for ot in OPTION_TYPES {
      assert_all_nan(bsm.greeks(s, k, R, Q, 0.0, ot), "bsm");
      assert_all_nan(merton.greeks(s, k, R, Q, 0.0, ot), "merton");
      assert_all_nan(heston.greeks(s, k, R, Q, 0.0, ot), "heston");
      let bsm_typed = [
        bsm.delta(s, k, R, Q, 0.0, ot),
        bsm.rho(s, k, R, Q, 0.0, ot),
        bsm.phi(s, k, R, Q, 0.0, ot),
        bsm.strike_delta(s, k, R, Q, 0.0, ot),
        bsm.lambda(s, k, R, Q, 0.0, ot),
        heston.delta(s, k, R, Q, 0.0, ot),
        heston.rho(s, k, R, Q, 0.0, ot),
        merton.delta(s, k, R, Q, 0.0, ot),
        merton.rho(s, k, R, Q, 0.0, ot),
        merton.theta(s, k, R, Q, 0.0, ot),
        merton.charm(s, k, R, Q, 0.0, ot),
      ];
      assert!(
        bsm_typed.iter().all(|g| g.is_nan()),
        "{ot:?} s = {s}: {bsm_typed:?}"
      );
    }
    let untyped = [
      bsm.vega(s, k, R, Q, 0.0),
      bsm.gamma(s, k, R, Q, 0.0),
      bsm.strike_gamma(s, k, R, Q, 0.0),
      heston.gamma(s, k, R, Q, 0.0),
      heston.vega(s, k, R, Q, 0.0),
      heston.vanna(s, k, R, Q, 0.0),
      heston.volga(s, k, R, Q, 0.0),
      heston.initial_variance_vega(s, k, R, Q, 0.0),
      merton.gamma(s, k, R, Q, 0.0),
      merton.vega(s, k, R, Q, 0.0),
      merton.vanna(s, k, R, Q, 0.0),
      merton.volga(s, k, R, Q, 0.0),
      merton.veta(s, k, R, Q, 0.0),
    ];
    assert!(untyped.iter().all(|g| g.is_nan()), "s = {s}: {untyped:?}");
  }
  let unlinked = BsmPricer::new(f64::NAN, BsmCoc::Merton1973);
  assert!(unlinked.price_call(110.0, k, R, Q, 0.0).is_nan());
  assert!(bsm.price_call(110.0, k, f64::NAN, Q, 0.0).is_nan());
  assert!(bsm.price_call(f64::NAN, k, R, Q, 0.0).is_nan());
}

#[test]
fn an_expired_digital_pays_its_payoff_and_nothing_at_the_strike() {
  let cash = CashOrNothingPricer::new(10.0, 0.35);
  let asset = AssetOrNothingPricer::new(0.35);
  let k = 100.0;
  for (s, call_pays, put_pays) in [(90.0, 0.0, 1.0), (100.0, 0.0, 0.0), (110.0, 1.0, 0.0)] {
    assert_eq!(
      cash.price_call(s, k, R, Q, 0.0),
      10.0 * call_pays,
      "cash call at s = {s}"
    );
    assert_eq!(
      cash.price_put(s, k, R, Q, 0.0),
      10.0 * put_pays,
      "cash put at s = {s}"
    );
    assert_eq!(
      asset.price_call(s, k, R, Q, 0.0),
      s * call_pays,
      "asset call at s = {s}"
    );
    assert_eq!(
      asset.price_put(s, k, R, Q, 0.0),
      s * put_pays,
      "asset put at s = {s}"
    );
    for ot in OPTION_TYPES {
      assert_all_nan(cash.greeks(s, k, R, Q, 0.0, ot), "cash-or-nothing");
      assert_all_nan(asset.greeks(s, k, R, Q, 0.0, ot), "asset-or-nothing");
      let accessors = [
        cash.delta(s, k, R, Q, 0.0, ot),
        cash.gamma(s, k, R, Q, 0.0, ot),
        cash.vega(s, k, R, Q, 0.0, ot),
        asset.delta(s, k, R, Q, 0.0, ot),
      ];
      assert!(
        accessors.iter().all(|g| g.is_nan()),
        "{ot:?} s = {s}: {accessors:?}"
      );
    }
  }
  assert!(
    CashOrNothingPricer::new(10.0, f64::NAN)
      .price_call(110.0, k, R, Q, 0.0)
      .is_nan()
  );
  assert!(cash.price_call(f64::NAN, k, R, Q, 0.0).is_nan());
  assert!(cash.price_call(110.0, k, R, f64::NAN, 0.0).is_nan());
  assert!(asset.price_put(90.0, f64::NAN, R, Q, 0.0).is_nan());
}

#[test]
fn an_expired_american_option_is_worth_its_payoff() {
  let snell = SnellEnvelopePricer::new(0.2, 200);
  let k = 100.0;
  for s in SPOTS {
    let (call, put) = ((s - k).max(0.0), (k - s).max(0.0));
    assert_eq!(snell.price_call(s, k, R, Q, 0.0), call, "call at s = {s}");
    assert_eq!(snell.price_put(s, k, R, Q, 0.0), put, "put at s = {s}");
    let detailed = snell.price_detailed(s, k, R, Q, 0.0, OptionType::Put);
    let values = (
      detailed.price,
      detailed.european_price,
      detailed.early_exercise_premium,
    );
    assert_eq!(values, (put, put, 0.0), "detailed put at s = {s}");
    assert!(detailed.exercise_boundary.is_empty());
  }
  assert!(snell.price_call(f64::NAN, k, R, Q, 0.0).is_nan());
  assert!(snell.price_call(110.0, k, f64::NAN, Q, 0.0).is_nan());
  assert!(snell.price_call(110.0, k, R, Q, -1.0).is_nan());
}

#[test]
#[should_panic(
  expected = "sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = -0.2"
)]
fn a_snell_lattice_with_a_non_positive_volatility_is_a_precondition() {
  SnellEnvelopePricer::new(-0.2, 100);
}

#[test]
fn zeta_is_the_exercise_probability_and_the_payoff_indicator_at_expiry() {
  let bsm = BsmPricer::new(0.25, BsmCoc::Merton1973);
  let call = bsm.zeta(100.0, 105.0, R, Q, 0.75, OptionType::Call);
  let put = bsm.zeta(100.0, 105.0, R, Q, 0.75, OptionType::Put);
  assert!(
    put > 0.0 && (call + put - 1.0).abs() < 1e-15,
    "{call} {put}"
  );
  let digital = CashOrNothingPricer::new(1.0, 0.25);
  for s in SPOTS {
    for ot in OPTION_TYPES {
      let indicator = digital.price_option(s, 100.0, R, Q, 0.0, ot);
      assert_eq!(
        bsm.zeta(s, 100.0, R, Q, 0.0, ot),
        indicator,
        "{ot:?} at s = {s}"
      );
    }
  }
  let unlinked = BsmPricer::new(f64::NAN, BsmCoc::Merton1973);
  assert!(
    unlinked
      .zeta(110.0, 100.0, R, Q, 0.0, OptionType::Call)
      .is_nan()
  );
  assert!(
    bsm
      .zeta(f64::NAN, 100.0, R, Q, 0.0, OptionType::Put)
      .is_nan()
  );
}

#[test]
fn an_expired_gap_pays_strictly_past_its_trigger_and_a_supershare_on_its_closed_band() {
  let gap = GapPricer::new(95.0, 0.2);
  for (s, call, put) in [(90.0, 0.0, 5.0), (100.0, 0.0, 0.0), (110.0, 15.0, 0.0)] {
    assert_eq!(
      gap.price_call(s, 100.0, R, Q, 0.0),
      call,
      "gap call at s = {s}"
    );
    assert_eq!(
      gap.price_put(s, 100.0, R, Q, 0.0),
      put,
      "gap put at s = {s}"
    );
  }
  let supershare = SuperSharePricer::new(110.0, 0.2);
  let band = [
    (80.0, 0.0),
    (90.0, 1.0),
    (100.0, 100.0 / 90.0),
    (110.0, 110.0 / 90.0),
    (120.0, 0.0),
  ];
  for (s, pays) in band {
    assert_eq!(
      supershare.price_call(s, 90.0, R, Q, 0.0),
      pays,
      "supershare at s = {s}"
    );
  }
  assert!(gap.price_put(f64::NAN, 100.0, R, Q, 0.0).is_nan());
  assert!(supershare.price_call(f64::NAN, 90.0, R, Q, 0.0).is_nan());
  let unlinked = GapPricer::new(95.0, f64::NAN);
  assert!(unlinked.price_call(110.0, 100.0, R, Q, 0.0).is_nan());
}

#[test]
fn heston_answers_nan_before_expiry_and_at_a_nan_tau() {
  let heston = heston();
  for tau in [-1e-9, -1.0, f64::NAN] {
    let CallPut { call, put } = heston.price_call_put(110.0, 100.0, R, Q, tau);
    assert!(
      call.is_nan() && put.is_nan(),
      "tau = {tau}: ({call}, {put})"
    );
    let vega = heston.initial_variance_vega(110.0, 100.0, R, Q, tau);
    assert!(vega.is_nan(), "tau = {tau}: v0 vega {vega}");
    assert_all_nan(
      heston.greeks(110.0, 100.0, R, Q, tau, OptionType::Call),
      "heston",
    );
  }
}

#[test]
fn a_multi_curve_leg_skips_an_empty_period_and_a_swap_rate_needs_two_dates() {
  let flat = |rate: f64| {
    DiscountCurve::<f64>::from_zero_rates(
      Array1::from_vec(vec![0.5, 1.0, 2.0, 5.0]).view(),
      Array1::from_vec(vec![rate; 4]).view(),
      InterpolationMethod::LogLinearOnDiscountFactors,
    )
  };
  let mut curves = MultiCurve::new(flat(0.04));
  curves.add_forecast(RateTenor::ThreeMonths, flat(0.045));
  let schedule = Array1::from_vec(vec![0.5, 1.0, 2.0]);
  let repeated = Array1::from_vec(vec![0.5, 1.0, 1.0, 2.0]);
  let pv = curves.pv_floating_leg(RateTenor::ThreeMonths, schedule.view(), 1.0);
  assert!(pv.is_some_and(f64::is_finite), "{pv:?}");
  assert_eq!(
    curves.pv_floating_leg(RateTenor::ThreeMonths, repeated.view(), 1.0),
    pv
  );
  let rate = curves.fair_swap_rate(RateTenor::ThreeMonths, schedule.view());
  assert!(rate.is_some_and(f64::is_finite), "{rate:?}");
  assert_eq!(
    curves.fair_swap_rate(RateTenor::ThreeMonths, repeated.view()),
    rate
  );
  assert_eq!(
    curves.fair_swap_rate(RateTenor::ThreeMonths, Array1::from_vec(vec![1.0]).view()),
    None
  );
  assert_eq!(
    curves.fair_swap_rate(RateTenor::ThreeMonths, Array1::from_vec(vec![]).view()),
    None
  );
  let empty_period = curves.fair_swap_rate(
    RateTenor::ThreeMonths,
    Array1::from_vec(vec![1.0, 1.0]).view(),
  );
  assert!(empty_period.is_some_and(f64::is_nan), "{empty_period:?}");
}

#[test]
fn a_multi_curve_leg_over_a_backward_period_is_nan() {
  let flat = DiscountCurve::<f64>::from_zero_rates(
    Array1::from_vec(vec![0.5, 1.0, 2.0, 5.0]).view(),
    Array1::from_vec(vec![0.04; 4]).view(),
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  let mut curves = MultiCurve::new(flat.clone());
  curves.add_forecast(RateTenor::ThreeMonths, flat);
  let backward = Array1::from_vec(vec![0.5, 2.0, 1.0]);
  let pv = curves.pv_floating_leg(RateTenor::ThreeMonths, backward.view(), 1.0);
  assert!(pv.is_some_and(f64::is_nan), "{pv:?}");
  let rate = curves.fair_swap_rate(RateTenor::ThreeMonths, backward.view());
  assert!(rate.is_some_and(f64::is_nan), "{rate:?}");
}

#[test]
fn a_bootstrap_quote_at_a_non_finite_maturity_is_an_invalid_quote() {
  let method = InterpolationMethod::LogLinearOnDiscountFactors;
  let deposit = |maturity| BootstrapInstrument::Deposit {
    maturity,
    rate: 0.03,
  };
  let rejects_quote_1 = |result: Result<DiscountCurve<f64>, QuantError>| {
    matches!(result, Err(QuantError::InvalidQuote { index: 1, .. }))
  };
  for bad in [f64::NAN, f64::INFINITY] {
    let quotes = [deposit(0.5), deposit(bad), deposit(1.0)];
    let spec = BootstrapSpec::new(&quotes, method);
    assert!(rejects_quote_1(bootstrap(spec)), "{bad}");
    assert!(
      rejects_quote_1(bootstrap(spec.iterative(IterativeOptions::default()))),
      "{bad}"
    );
    let swap = BootstrapInstrument::Swap {
      maturity: bad,
      rate: 0.03,
      frequency: Frequency::SemiAnnual,
    };
    assert!(
      rejects_quote_1(bootstrap(BootstrapSpec::new(&[deposit(0.5), swap], method))),
      "{bad}"
    );
  }
  let ois = DiscountCurve::<f64>::from_zero_rates(
    Array1::from_vec(vec![0.5, 1.0, 2.0]).view(),
    Array1::from_vec(vec![0.02; 3]).view(),
    method,
  );
  let swap = BootstrapInstrument::SwapWithSchedule {
    rate: 0.03,
    fixed_times: vec![1.0, f64::NAN],
    float_times: vec![0.5, 1.0, f64::NAN],
  };
  let quotes = [
    BootstrapInstrument::Deposit {
      maturity: 0.25,
      rate: 0.03,
    },
    swap,
  ];
  assert!(rejects_quote_1(bootstrap(
    BootstrapSpec::new(&quotes, method).discounting_with(&ois)
  )));
  assert!(bootstrap(BootstrapSpec::new(&[deposit(0.5), deposit(1.0)], method)).is_ok());
}

#[test]
fn a_rate_helper_at_a_non_finite_maturity_is_an_invalid_quote_at_its_own_index() {
  struct Pillar(f64, bool);

  impl RateHelper<f64> for Pillar {
    fn maturity(&self, _: NaiveDate) -> f64 {
      self.0
    }

    fn to_instrument(&self, _: NaiveDate) -> Option<BootstrapInstrument<f64>> {
      self.1.then_some(BootstrapInstrument::Deposit {
        maturity: self.0,
        rate: 0.03,
      })
    }
  }

  let date = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();
  let (unquoted, short, broken) = (
    Pillar(0.25, false),
    Pillar(0.5, true),
    Pillar(f64::NAN, true),
  );
  let helpers: [&dyn RateHelper<f64>; 3] = [&unquoted, &short, &broken];
  let method = InterpolationMethod::LogLinearOnDiscountFactors;
  let result = build_curve(&helpers, date, method);
  assert!(
    matches!(result, Err(QuantError::InvalidQuote { index: 2, .. })),
    "{result:?}"
  );
  assert!(build_curve(&helpers[..2], date, method).is_ok());
}
