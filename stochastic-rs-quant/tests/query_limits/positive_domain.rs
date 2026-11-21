//! Before expiry a lognormal pricer has no value at a spot or strike that is not positive.

use ndarray::array;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_quant::OptionStyle;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::calibration::rbergomi::RBergomiParams;
use stochastic_rs_quant::calibration::rbergomi::RBergomiXi0;
use stochastic_rs_quant::lattice::CrrModel;
use stochastic_rs_quant::pricing::asian::AsianPricer;
use stochastic_rs_quant::pricing::bachelier::BachelierPricer;
use stochastic_rs_quant::pricing::basket::ArithmeticBasketLevyPricer;
use stochastic_rs_quant::pricing::basket::GeometricBasketPricer;
use stochastic_rs_quant::pricing::bjerksund_stensland::BjerksundStensland2002Pricer;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::pricing::finite_difference::FiniteDifferenceMethod;
use stochastic_rs_quant::pricing::finite_difference::FiniteDifferencePricer;
use stochastic_rs_quant::pricing::fourier::BsmFourier;
use stochastic_rs_quant::pricing::fourier::CosEngine;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::heston_adi::HestonAdiPricer;
use stochastic_rs_quant::pricing::heston_stoch_corr::HestonStochCorrPricer;
use stochastic_rs_quant::pricing::kirk::KirkSpreadPricer;
use stochastic_rs_quant::pricing::malliavin_gbm::GbmMalliavinPricer;
use stochastic_rs_quant::pricing::merton_jump::Merton1976Pricer;
use stochastic_rs_quant::pricing::quanto::QuantoPricer;
use stochastic_rs_quant::pricing::rainbow::RainbowPayoff;
use stochastic_rs_quant::pricing::rainbow::StulzRainbowPricer;
use stochastic_rs_quant::pricing::rbergomi::RBergomiPricer;
use stochastic_rs_quant::pricing::regime_switching::CosPricer;
use stochastic_rs_quant::pricing::sabr::SabrParams;
use stochastic_rs_quant::pricing::sabr::SabrPricer;
use stochastic_rs_quant::pricing::snell_envelope::SnellEnvelopePricer;
use stochastic_rs_quant::pricing::spread::MargrabePricer;
use stochastic_rs_quant::traits::ModelPricer;

const K: f64 = 100.0;
const R: f64 = 0.05;
const Q: f64 = 0.02;
const TAU: f64 = 1.0;
const TYPES: [OptionType; 2] = [OptionType::Call, OptionType::Put];
/// `(s, k)` with one coordinate zero or negative.
const OFF: [(f64, f64); 4] = [(0.0, K), (-5.0, K), (100.0, 0.0), (100.0, -5.0)];

/// Every `(what, value)` that is a number, where each must be NaN.
fn numbers(cells: impl IntoIterator<Item = (String, f64)>) -> Vec<String> {
  let priced = cells.into_iter().filter(|(_, value)| !value.is_nan());
  priced
    .map(|(what, value)| format!("{what}: {value}"))
    .collect()
}

fn sweep(what: &str, model: &dyn ModelPricer) -> Vec<(String, f64)> {
  let cell = |(s, k), leg| {
    let price = model.price_option(s, k, R, Q, TAU, leg);
    (format!("{what} {leg:?} at s = {s}, k = {k}"), price)
  };
  let cells = OFF.iter().flat_map(|&at| TYPES.map(|leg| cell(at, leg)));
  cells.collect()
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

#[test]
fn a_lognormal_model_pricer_is_nan_for_a_spot_or_strike_that_is_not_positive() {
  let fd = |style, method| FiniteDifferencePricer::new(0.2, 40, 40, style, method);
  let rbergomi = RBergomiPricer::new(RBergomiParams {
    hurst: 0.12,
    rho: -0.72,
    eta: 1.6,
    xi0: RBergomiXi0::Constant(0.04),
  });
  let rbergomi = rbergomi.with_paths(16).with_seed(Deterministic::new(7));
  let malliavin = GbmMalliavinPricer::new(0.2, 64, 8, 0.5).with_seed(Deterministic::new(3));
  let models: [(&str, &dyn ModelPricer); 18] = [
    ("BSM", &BsmPricer::new(0.2, BsmCoc::Merton1973)),
    ("Black 76", &BsmPricer::new(0.2, BsmCoc::Black1976)),
    (
      "Merton 1976",
      &Merton1976Pricer::new(0.2, 0.5, 0.1, 20, BsmCoc::Merton1973),
    ),
    ("Heston", &super::heston()),
    (
      "SABR",
      &SabrPricer::new(SabrParams::new(0.2, 0.7, -0.3, 0.4)),
    ),
    ("Asian", &AsianPricer::new(0.25)),
    ("quanto", &QuantoPricer::new(0.2, 0.12, 0.3, 0.05, 1.5)),
    (
      "Bjerksund-Stensland",
      &BjerksundStensland2002Pricer::new(0.25),
    ),
    ("CRR", &CrrModel::new(0.2, 50)),
    ("Snell", &SnellEnvelopePricer::new(0.2, 50)),
    (
      "FD European explicit",
      &fd(OptionStyle::European, FiniteDifferenceMethod::Explicit),
    ),
    (
      "FD European implicit",
      &fd(OptionStyle::European, FiniteDifferenceMethod::Implicit),
    ),
    (
      "FD European Crank-Nicolson",
      &fd(OptionStyle::European, FiniteDifferenceMethod::CrankNicolson),
    ),
    (
      "FD American Crank-Nicolson",
      &fd(OptionStyle::American, FiniteDifferenceMethod::CrankNicolson),
    ),
    (
      "ADI",
      &HestonAdiPricer::new(heston_params()).with_grid(20, 10, 10),
    ),
    ("rBergomi", &rbergomi),
    ("Malliavin", &malliavin),
    (
      "ADI down-and-out",
      &HestonAdiPricer::new(heston_params())
        .with_grid(20, 10, 10)
        .with_barrier(95.0),
    ),
  ];
  let mut cells = Vec::new();
  for (what, model) in models {
    cells.extend(sweep(what, model));
  }
  let crr = CrrModel::new(0.2, 50);
  for (s, k) in OFF {
    for leg in TYPES {
      let american = crr.price_american(s, k, R, Q, TAU, leg);
      cells.push((format!("CRR American {leg:?} at {s}, {k}"), american));
      let cos = CosEngine::default().price(&BsmFourier { sigma: 0.2 }, s, k, R, Q, TAU, leg);
      cells.push((format!("CosEngine {leg:?} at {s}, {k}"), cos));
    }
  }
  let priced = numbers(cells);
  assert!(priced.is_empty(), "{priced:#?}");
}

/// Its own test: the answer comes before the quadrature, whose walk at `s = 0` has no bound.
#[test]
fn the_stochastic_correlation_pricer_is_nan_for_a_spot_or_strike_that_is_not_positive() {
  let hscm = HestonStochCorrPricer::new(0.04, 1.5, 0.04, 0.3, -0.5, 2.0, -0.4, 0.2, 0.3);
  let mut cells = sweep("HSCM", &hscm);
  for (s, k) in OFF {
    let direct = hscm.price_call_carr_madan(s, k, R, Q, TAU);
    cells.push((format!("HSCM Carr-Madan at {s}, {k}"), direct));
  }
  let priced = numbers(cells);
  assert!(priced.is_empty(), "{priced:#?}");
}

#[test]
fn the_payoff_stands_at_expiry_for_a_spot_or_strike_that_is_not_positive() {
  let fd = FiniteDifferenceMethod::CrankNicolson;
  let models: [(&str, &dyn ModelPricer); 7] = [
    ("BSM", &BsmPricer::new(0.2, BsmCoc::Merton1973)),
    (
      "Merton 1976",
      &Merton1976Pricer::new(0.2, 0.5, 0.1, 20, BsmCoc::Merton1973),
    ),
    ("Asian", &AsianPricer::new(0.25)),
    (
      "Bjerksund-Stensland",
      &BjerksundStensland2002Pricer::new(0.25),
    ),
    ("CRR", &CrrModel::new(0.2, 50)),
    (
      "FD",
      &FiniteDifferencePricer::new(0.2, 40, 40, OptionStyle::European, fd),
    ),
    (
      "HSCM",
      &HestonStochCorrPricer::new(0.04, 1.5, 0.04, 0.3, -0.5, 2.0, -0.4, 0.2, 0.3),
    ),
  ];
  for (what, model) in models {
    for (s, k) in OFF {
      let (call, put) = (
        model.price_call(s, k, R, Q, 0.0),
        model.price_put(s, k, R, Q, 0.0),
      );
      assert_eq!(
        (call, put),
        ((s - k).max(0.0), (k - s).max(0.0)),
        "{what} at s = {s}, k = {k}"
      );
    }
  }
}

/// A normal model: the forward and the strike range over the whole line.
#[test]
fn bachelier_prices_a_negative_spot_and_strike() {
  let bachelier = BachelierPricer::new(20.0);
  for (s, k) in [(-5.0, 3.0), (0.0, 0.0), (4.0, -3.0), (-2.0, -6.0)] {
    let (call, put) = (
      bachelier.price_call(s, k, R, Q, TAU),
      bachelier.price_put(s, k, R, Q, TAU),
    );
    assert!(call > 0.0 && put > 0.0, "{s} {k}: {call} {put}");
    let parity = s * (-Q * TAU).exp() - k * (-R * TAU).exp();
    assert!((call - put - parity).abs() < 1e-12, "{s} {k}");
  }
}

#[test]
fn the_multi_asset_closed_forms_are_nan_for_a_spot_that_is_not_positive() {
  let mut cells = Vec::new();
  let margrabe = MargrabePricer::new(0.25, 0.2, 0.4);
  let kirk = KirkSpreadPricer::new(0.25, 0.2, 0.4);
  let (w, sigma) = (array![0.5, 0.5], array![0.25, 0.2]);
  let rho = array![[1.0, 0.4], [0.4, 1.0]];
  let q = array![Q, 0.01];
  let levy = ArithmeticBasketLevyPricer::new(w.clone(), sigma.clone(), rho.clone());
  let geometric = GeometricBasketPricer::new(w, sigma, rho);
  let (generator, vols) = (array![[-0.5, 0.5], [1.0, -1.0]], array![0.15, 0.3]);
  for (s1, s2) in [(0.0, 100.0), (-5.0, 100.0), (100.0, 0.0), (100.0, -5.0)] {
    let at = format!("at ({s1}, {s2})");
    cells.push((
      format!("Margrabe {at}"),
      margrabe.price(s1, s2, Q, 0.01, TAU),
    ));
    cells.push((
      format!("Margrabe delta1 {at}"),
      margrabe.delta1(s1, s2, Q, 0.01, TAU),
    ));
    cells.push((
      format!("Margrabe delta2 {at}"),
      margrabe.delta2(s1, s2, Q, 0.01, TAU),
    ));
    let spread = kirk.spread_call_put(s1, s2, 5.0, R, TAU);
    cells.push((format!("Kirk call {at}"), spread.call));
    cells.push((format!("Kirk put {at}"), spread.put));
    for leg in TYPES {
      let spots = array![s1, s2];
      let price = levy.price_option(spots.view(), K, R, q.view(), TAU, leg);
      cells.push((format!("Levy {leg:?} {at}"), price));
      let price = geometric.price_option(spots.view(), K, R, q.view(), TAU, leg);
      cells.push((format!("geometric {leg:?} {at}"), price));
    }
    for payoff in [
      RainbowPayoff::CallOnMax,
      RainbowPayoff::CallOnMin,
      RainbowPayoff::PutOnMax,
      RainbowPayoff::PutOnMin,
    ] {
      let stulz = StulzRainbowPricer::new(payoff, 0.25, 0.2, 0.4);
      let price = stulz.price(s1, s2, K, R, Q, 0.01, TAU);
      cells.push((format!("Stulz {payoff:?} {at}"), price));
    }
  }
  for (s, k) in OFF {
    let pricer = CosPricer::default();
    let call = pricer.price_call(s, k, R, Q, TAU, generator.view(), vols.view(), 0);
    let put = pricer.price_put(s, k, R, Q, TAU, generator.view(), vols.view(), 0);
    cells.push((format!("regime COS call at {s}, {k}"), call));
    cells.push((format!("regime COS put at {s}, {k}"), put));
  }
  for k in [0.0, -5.0] {
    for leg in TYPES {
      let spots = array![100.0, 100.0];
      let price = levy.price_option(spots.view(), k, R, q.view(), TAU, leg);
      cells.push((format!("Levy {leg:?} at k = {k}"), price));
      let price = geometric.price_option(spots.view(), k, R, q.view(), TAU, leg);
      cells.push((format!("geometric {leg:?} at k = {k}"), price));
    }
  }
  let priced = numbers(cells);
  assert!(priced.is_empty(), "{priced:#?}");
  assert_eq!(margrabe.price(0.0, 100.0, Q, 0.01, 0.0), 0.0);
  assert_eq!(kirk.spread_call_put(0.0, 95.0, 5.0, R, 0.0).put, 100.0);
  // A spread strike is a difference of prices, so it may be zero or negative.
  let spread = kirk.spread_call_put(100.0, 95.0, -5.0, R, TAU);
  assert!(spread.call > 10.0 && spread.put > 0.0, "{spread:?}");
}
