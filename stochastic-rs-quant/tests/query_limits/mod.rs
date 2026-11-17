//! Query limits: NaN outside a pricer's domain and the payoff at `tau = 0`, never a sentinel.

use ndarray::Array1;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::credit::HazardInterpolation;
use stochastic_rs_quant::credit::SurvivalCurve;
use stochastic_rs_quant::credit::SurvivalPoint;
use stochastic_rs_quant::curves::CurvePoint;
use stochastic_rs_quant::curves::DiscountCurve;
use stochastic_rs_quant::curves::InterpolationMethod;
use stochastic_rs_quant::pricing::AssetOrNothingPricer;
use stochastic_rs_quant::pricing::CashOrNothingPricer;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::heston::HestonPricer;
use stochastic_rs_quant::pricing::merton_jump::Merton1976Pricer;
use stochastic_rs_quant::pricing::sabr::alpha_from_atm_vol;
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
    assert!(
      value.is_nan(),
      "{what}: {name} must be NaN at tau = 0, got {value}"
    );
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
    &Array1::from_vec(vec![1.0, 2.0, 5.0]),
    &Array1::from_vec(vec![0.03; 3]),
    InterpolationMethod::LogLinearOnDiscountFactors,
  );
  assert!(curve.simple_forward_rate(2.0, 1.0).is_nan());
  assert!(curve.simple_forward_rate(1.0, 1.0).is_nan());
  assert!((curve.simple_forward_rate(1.0, 2.0) - (0.03_f64.exp() - 1.0)).abs() < 1e-12);
  assert!(curve.forward_rate(2.0, 1.0).is_nan());
  assert!(curve.forward_rate(1.0, 1.0).is_nan());
  assert!((curve.forward_rate(1.0, 2.0) - 0.03).abs() < 1e-12);
  assert!(curve.forward_rate(f64::NAN, 2.0).is_nan() && curve.forward_rate(1.0, f64::NAN).is_nan());
  assert_eq!(curve.forward_rates(&Array1::from_vec(vec![1.0])).len(), 0);
  assert_eq!(curve.forward_rates(&Array1::from_vec(vec![])).len(), 0);
  let forwards = curve.forward_rates(&Array1::from_vec(vec![1.0, 2.0, 2.0]));
  assert!((forwards[0] - 0.03).abs() < 1e-12 && forwards[1].is_nan());
  assert!(curve.par_rate(0.0, 2).is_nan());
  assert!(curve.par_rate(-1.0, 2).is_nan());
  assert!(curve.par_rate(1.0, 0).is_nan());
  assert!(curve.par_rate(f64::INFINITY, 2).is_nan());
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
fn prices_at_zero_maturity_are_intrinsic_and_greeks_are_nan() {
  let bsm = BsmPricer::new(0.2, BsmCoc::Merton1973);
  let merton = Merton1976Pricer::new(0.2, 0.5, 0.3, 50, BsmCoc::Merton1973);
  let heston = heston();
  let k = 100.0;
  for s in SPOTS {
    let prices = [
      ("bsm", bsm.call_put(s, k, R, Q, 0.0)),
      ("merton", merton.call_put(s, k, R, Q, 0.0)),
      ("heston", heston.call_put(s, k, R, Q, 0.0)),
    ];
    for (name, (call, put)) in prices {
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
        bsm.zeta(s, k, R, Q, 0.0, ot),
        bsm.strike_delta(s, k, R, Q, 0.0, ot),
        bsm.lambda(s, k, R, Q, 0.0, ot).0,
        bsm.lambda(s, k, R, Q, 0.0, ot).1,
        heston.delta(s, k, R, Q, 0.0, ot),
        heston.rho(s, k, R, Q, 0.0, ot),
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
      heston.call_put_initial_variance_vega(s, k, R, Q, 0.0).0,
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
