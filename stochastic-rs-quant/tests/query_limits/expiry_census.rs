//! The non-exotic pricers at and past expiry: the payoff at `tau = 0`, NaN for `tau < 0` or NaN.

use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_quant::OptionStyle;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::calibration::rbergomi::RBergomiParams;
use stochastic_rs_quant::calibration::rbergomi::RBergomiXi0;
use stochastic_rs_quant::calibration::rbergomi::simulate_rbergomi_terminal_samples;
use stochastic_rs_quant::lattice::CrrModel;
use stochastic_rs_quant::pricing::asian::AsianPricer;
use stochastic_rs_quant::pricing::bachelier::BachelierPricer;
use stochastic_rs_quant::pricing::basket::ArithmeticBasketLevyPricer;
use stochastic_rs_quant::pricing::basket::GeometricBasketPricer;
use stochastic_rs_quant::pricing::bjerksund_stensland::BjerksundStensland2002Pricer;
use stochastic_rs_quant::pricing::cgmysv::CgmysvParams;
use stochastic_rs_quant::pricing::cgmysv::CgmysvPricer;
use stochastic_rs_quant::pricing::finite_difference::FiniteDifferenceMethod;
use stochastic_rs_quant::pricing::finite_difference::FiniteDifferencePricer;
use stochastic_rs_quant::pricing::fourier::BsmFourier;
use stochastic_rs_quant::pricing::fourier::CosEngine;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::heston_adi::HestonAdiPricer;
use stochastic_rs_quant::pricing::heston_stoch_corr::HestonStochCorrPricer;
use stochastic_rs_quant::pricing::kirk::KirkSpreadPricer;
use stochastic_rs_quant::pricing::quanto::QuantoPricer;
use stochastic_rs_quant::pricing::rainbow::RainbowPayoff;
use stochastic_rs_quant::pricing::rainbow::StulzRainbowPricer;
use stochastic_rs_quant::pricing::rbergomi::RBergomiPricer;
use stochastic_rs_quant::pricing::regime_switching::CosPricer;
use stochastic_rs_quant::pricing::spread::MargrabePricer;
use stochastic_rs_quant::traits::ModelPricer;

const K: f64 = 100.0;
const R: f64 = 0.05;
const Q: f64 = 0.02;
const SPOTS: [f64; 3] = [90.0, 100.0, 110.0];
const TYPES: [OptionType; 2] = [OptionType::Call, OptionType::Put];

fn vanilla(option_type: OptionType, s: f64) -> f64 {
  match option_type {
    OptionType::Call => (s - K).max(0.0),
    OptionType::Put => (K - s).max(0.0),
  }
}

/// `price(type, s, tau)` is `payoff(type, s)` at `tau = 0` and NaN at `tau = −1`, NaN and `+∞`.
fn assert_expiry_rule(
  what: &str,
  price: impl Fn(OptionType, f64, f64) -> f64,
  payoff: impl Fn(OptionType, f64) -> f64,
) {
  for option_type in TYPES {
    for s in SPOTS {
      let (got, want) = (price(option_type, s, 0.0), payoff(option_type, s));
      assert_eq!(got, want, "{what} {option_type:?} at s = {s}, tau = 0");
      for tau in [-1.0, f64::NAN, f64::INFINITY] {
        let got = price(option_type, s, tau);
        assert!(
          got.is_nan(),
          "{what} {option_type:?} at s = {s}, tau = {tau}: {got}"
        );
      }
    }
  }
}

fn of_model(model: &impl ModelPricer) -> impl Fn(OptionType, f64, f64) -> f64 + '_ {
  move |option_type, s, tau| model.price_option(s, K, R, Q, tau, option_type)
}

fn heston_params() -> HestonParams {
  HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  }
}

/// Its own test: before the rule, the `tau = −1` quadrature never returned.
#[test]
fn the_stochastic_correlation_pricer_follows_the_expiry_rule() {
  let hscm = HestonStochCorrPricer::new(0.04, 1.5, 0.04, 0.3, -0.5, 2.0, -0.4, 0.2, 0.3);
  assert_expiry_rule("HSCM", of_model(&hscm), vanilla);
}

#[test]
fn the_single_asset_model_pricers_follow_the_expiry_rule() {
  let adi = HestonAdiPricer::new(heston_params()).with_grid(20, 10, 10);
  assert_expiry_rule("ADI", of_model(&adi), vanilla);
  assert_expiry_rule("Asian", of_model(&AsianPricer::new(0.25)), vanilla);
  let american = BjerksundStensland2002Pricer::new(0.25);
  assert_expiry_rule("Bjerksund-Stensland", of_model(&american), vanilla);
  assert_expiry_rule("Bachelier", of_model(&BachelierPricer::new(20.0)), vanilla);
  let quanto = QuantoPricer::new(0.2, 0.12, 0.3, 0.05, 1.5);
  assert_expiry_rule("quanto", of_model(&quanto), |t, s| 1.5 * vanilla(t, s));
  for style in [OptionStyle::European, OptionStyle::American] {
    for method in [
      FiniteDifferenceMethod::Explicit,
      FiniteDifferenceMethod::Implicit,
      FiniteDifferenceMethod::CrankNicolson,
    ] {
      let fd = FiniteDifferencePricer::new(0.2, 100, 100, style, method);
      assert_expiry_rule(&format!("FD {style:?} {method:?}"), of_model(&fd), vanilla);
    }
  }
  let crr = CrrModel::new(0.2, 50);
  assert_expiry_rule("CRR European", of_model(&crr), vanilla);
  let american = |option_type, s, tau| crr.price_american(s, K, R, Q, tau, option_type);
  assert_expiry_rule("CRR American", american, vanilla);
}

/// No query panics: a NaN coordinate before expiry is NaN in every single-asset pricer here.
#[test]
fn a_nan_query_before_expiry_is_nan() {
  let hscm = HestonStochCorrPricer::new(0.04, 1.5, 0.04, 0.3, -0.5, 2.0, -0.4, 0.2, 0.3);
  let adi = HestonAdiPricer::new(heston_params()).with_grid(20, 10, 10);
  let fd =
    |style| FiniteDifferencePricer::new(0.2, 20, 20, style, FiniteDifferenceMethod::CrankNicolson);
  let rbergomi = RBergomiPricer::new(rbergomi_params()).with_paths(16);
  let models: [(&str, &dyn ModelPricer); 9] = [
    ("HSCM", &hscm),
    ("ADI", &adi),
    ("Asian", &AsianPricer::new(0.25)),
    (
      "Bjerksund-Stensland",
      &BjerksundStensland2002Pricer::new(0.25),
    ),
    ("Bachelier", &BachelierPricer::new(20.0)),
    ("quanto", &QuantoPricer::new(0.2, 0.12, 0.3, 0.05, 1.5)),
    ("FD European", &fd(OptionStyle::European)),
    ("FD American", &fd(OptionStyle::American)),
    ("rBergomi", &rbergomi),
  ];
  let nan = f64::NAN;
  for (what, model) in models {
    for (s, k, r, q) in [
      (nan, K, R, Q),
      (110.0, nan, R, Q),
      (110.0, K, nan, Q),
      (110.0, K, R, nan),
    ] {
      for option_type in TYPES {
        let got = model.price_option(s, k, r, q, 1.0, option_type);
        assert!(
          got.is_nan(),
          "{what} {option_type:?} at ({s}, {k}, {r}, {q}): {got}"
        );
      }
    }
  }
}

#[test]
fn the_adi_down_and_out_call_at_expiry_is_its_payoff_above_the_barrier() {
  let adi = HestonAdiPricer::new(heston_params())
    .with_grid(20, 10, 10)
    .with_barrier(95.0);
  assert_eq!(adi.price_call(90.0, K, R, Q, 0.0), 0.0);
  assert_eq!(adi.price_call(110.0, K, R, Q, 0.0), 10.0);
  assert!(adi.price_call(110.0, K, R, Q, -1.0).is_nan());
  assert!(adi.price_call(f64::NAN, K, R, Q, 0.0).is_nan());
  assert!(adi.price_put(110.0, K, R, Q, 0.0).is_nan());
}

#[test]
fn the_cos_pricers_follow_the_expiry_rule() {
  let bsm = BsmFourier { sigma: 0.2 };
  let cos = |option_type, s, tau| CosEngine::default().price(&bsm, s, K, R, Q, tau, option_type);
  assert_expiry_rule("CosEngine", cos, vanilla);
  let (generator, vols) = (array![[-0.5, 0.5], [1.0, -1.0]], array![0.15, 0.3]);
  let regime = |option_type, s, tau| {
    let pricer = CosPricer::default();
    match option_type {
      OptionType::Call => pricer.price_call(s, K, R, Q, tau, generator.view(), vols.view(), 0),
      OptionType::Put => pricer.price_put(s, K, R, Q, tau, generator.view(), vols.view(), 0),
    }
  };
  assert_expiry_rule("CosPricer", regime, vanilla);
}

#[test]
fn the_two_asset_closed_forms_follow_the_expiry_rule() {
  let margrabe = MargrabePricer::new(0.25, 0.2, 0.4);
  let exchange = |option_type, s, tau| match option_type {
    OptionType::Call => margrabe.price(s, K, Q, 0.01, tau),
    OptionType::Put => margrabe.price(K, s, 0.01, Q, tau),
  };
  assert_expiry_rule("Margrabe", exchange, vanilla);
  let kirk = KirkSpreadPricer::new(0.25, 0.2, 0.4);
  let spread = |option_type, s, tau| match option_type {
    OptionType::Call => kirk.spread_call(s, 95.0, 5.0, R, tau),
    OptionType::Put => kirk.spread_put(s, 95.0, 5.0, R, tau),
  };
  assert_expiry_rule("Kirk", spread, vanilla);
  for (call, put, on_max) in [
    (RainbowPayoff::CallOnMax, RainbowPayoff::PutOnMax, true),
    (RainbowPayoff::CallOnMin, RainbowPayoff::PutOnMin, false),
  ] {
    let other = 95.0_f64;
    let price = |option_type, s, tau| {
      let payoff = if option_type == OptionType::Call {
        call
      } else {
        put
      };
      StulzRainbowPricer::new(payoff, 0.25, 0.2, 0.4).price(s, other, K, R, Q, 0.01, tau)
    };
    let extreme = |s: f64| if on_max { s.max(other) } else { s.min(other) };
    assert_expiry_rule("Stulz", price, |option_type, s| {
      vanilla(option_type, extreme(s))
    });
  }
}

#[test]
fn the_basket_closed_forms_follow_the_expiry_rule() {
  let (w, sigma) = (array![0.5, 0.5], array![0.25, 0.2]);
  let rho = array![[1.0, 0.4], [0.4, 1.0]];
  let q = array![Q, 0.01];
  let levy = ArithmeticBasketLevyPricer::new(w.clone(), sigma.clone(), rho.clone());
  let arithmetic =
    |option_type, s, tau| levy.price_option(array![s, K].view(), K, R, q.view(), tau, option_type);
  assert_expiry_rule("Levy basket", arithmetic, |t, s| {
    vanilla(t, 0.5 * s + 0.5 * K)
  });
  let geometric = GeometricBasketPricer::new(w, sigma, rho);
  let price = |option_type, s, tau| {
    geometric.price_option(array![s, K].view(), K, R, q.view(), tau, option_type)
  };
  let mean = |s: f64| (0.5 * s.ln() + 0.5 * K.ln()).exp();
  assert_expiry_rule("geometric basket", price, |t, s| vanilla(t, mean(s)));
}

fn rbergomi_params() -> RBergomiParams {
  RBergomiParams {
    hurst: 0.12,
    rho: -0.72,
    eta: 1.6,
    xi0: RBergomiXi0::Constant(0.04),
  }
}

#[test]
fn the_monte_carlo_model_pricers_pay_out_at_expiry_without_a_path() {
  let rbergomi = RBergomiPricer::new(rbergomi_params())
    .with_paths(64)
    .with_seed(Deterministic::new(7));
  assert_expiry_rule("rBergomi", of_model(&rbergomi), vanilla);
  let at_expiry = rbergomi.price_call_estimate(110.0, K, R, Q, 0.0);
  assert_eq!(
    (at_expiry.mean, at_expiry.std_err, at_expiry.n_samples),
    (10.0, 0.0, 0)
  );
  let cgmysv = |s: f64| CgmysvPricer {
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
    r: R,
    q: Q,
    n_paths: 16,
    n_steps: 8,
    n_jumps: 16,
    seed: Deterministic::new(7),
  };
  let european = |option_type, s, tau| cgmysv(s).price_european(K, tau, option_type).mean;
  assert_expiry_rule("CGMYSV European", european, vanilla);
  let american = |option_type, s, tau| cgmysv(s).price_american(K, tau, option_type).mean;
  assert_expiry_rule("CGMYSV American", american, vanilla);
  let asian = |option_type, s, tau| cgmysv(s).price_asian(K, tau, option_type).mean;
  assert_expiry_rule("CGMYSV Asian", asian, vanilla);
  let expired = cgmysv(110.0).price_european(K, 0.0, OptionType::Call);
  assert_eq!((expired.std_err, expired.n_samples), (0.0, 0));
}

#[test]
fn the_rbergomi_sampler_answers_its_spot_at_zero_maturity_and_nan_past_it() {
  let seed = Deterministic::new(7);
  let draw = |s0, tau| {
    simulate_rbergomi_terminal_samples(&rbergomi_params(), s0, R, Q, tau, 4, 50, 8, &seed)
  };
  assert_eq!(draw(100.0, 0.0), vec![100.0; 4]);
  for (s0, tau) in [
    (100.0, -1.0),
    (100.0, f64::NAN),
    (f64::NAN, 1.0),
    (0.0, 1.0),
    (-5.0, 1.0),
  ] {
    let samples = draw(s0, tau);
    assert!(
      samples.len() == 4 && samples.iter().all(|x| x.is_nan()),
      "{s0} {tau}: {samples:?}"
    );
  }
}

#[test]
#[should_panic(expected = "paths must satisfy `paths > 0`, got paths = 0")]
fn an_rbergomi_sample_without_a_path_is_a_precondition() {
  let seed = Deterministic::new(7);
  simulate_rbergomi_terminal_samples(&rbergomi_params(), 100.0, R, Q, 1.0, 0, 50, 8, &seed);
}
