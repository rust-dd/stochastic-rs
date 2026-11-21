//! One shape per concept: a named pair where a pricer yields both legs, structs from the order
//! book, and an enum for the option type.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_quant::CallPut;
use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::calibration::SwaptionQuote;
use stochastic_rs_quant::fx::CurrencyPair;
use stochastic_rs_quant::fx::FxForward;
use stochastic_rs_quant::fx::currency;
use stochastic_rs_quant::instruments::SwaptionDirection;
use stochastic_rs_quant::market::provider::ChainQuote;
use stochastic_rs_quant::order_book::Depth;
use stochastic_rs_quant::order_book::Execution;
use stochastic_rs_quant::order_book::Level;
use stochastic_rs_quant::order_book::OrderBook;
use stochastic_rs_quant::order_book::Placement;
use stochastic_rs_quant::order_book::Side;
use stochastic_rs_quant::pricing::asian::AsianPricer;
use stochastic_rs_quant::pricing::bachelier::BachelierPricer;
use stochastic_rs_quant::pricing::bjerksund_stensland::BjerksundStensland2002Pricer;
use stochastic_rs_quant::pricing::bsm::BsmCoc;
use stochastic_rs_quant::pricing::bsm::BsmPricer;
use stochastic_rs_quant::pricing::fourier::BatesFourier;
use stochastic_rs_quant::pricing::fourier::HkdeFourier;
use stochastic_rs_quant::pricing::fourier::HkdeParams;
use stochastic_rs_quant::pricing::fourier::SvjParams;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::pricing::heston::HestonPricer;
use stochastic_rs_quant::pricing::heston_stoch_corr::HestonStochCorrPricer;
use stochastic_rs_quant::pricing::kirk::KirkSpreadPricer;
use stochastic_rs_quant::pricing::malliavin_gbm::GbmMalliavinPricer;
use stochastic_rs_quant::pricing::merton_jump::Merton1976Pricer;
use stochastic_rs_quant::pricing::quanto::QuantoPricer;
use stochastic_rs_quant::pricing::sabr::SabrParams;
use stochastic_rs_quant::pricing::sabr::SabrPricer;
use stochastic_rs_quant::pricing::snell_envelope::SnellEnvelopePricer;
use stochastic_rs_quant::traits::ModelPricer;
use stochastic_rs_quant::vol_surface::ImpliedVolSurface;
use stochastic_rs_quant::vol_surface::build_surface;

const S: f64 = 100.0;
const K: f64 = 105.0;
const R: f64 = 0.05;
const Q: f64 = 0.02;
const TAU: f64 = 0.75;

const fn pair(call: f64, put: f64) -> CallPut {
  CallPut { call, put }
}

const HESTON_LAMBDA: CallPut = pair(5.081933443026998, 7.706153343407024);
const HESTON_VEGA: f64 = 51.66113350364657;
const MERTON: CallPut = pair(7.15321711897312, 9.777437019353142);
const HSCM: CallPut = pair(5.383063358956783, 8.007283259336816);
const MALLIAVIN: CallPut = pair(5.747134860337733, 8.371354760717765);
const MALLIAVIN_LOCALIZED: CallPut = pair(6.4057386359588495, 9.029958536338881);
const KIRK: CallPut = pair(8.547483131757428, 3.6709335716157754);
const SABR: CallPut = pair(5.633121359954366, 8.257341260334393);
const SABR_BETA: CallPut = pair(5.529815233702159, 8.154035134082179);
const BACHELIER: CallPut = pair(5.425620481356887, 8.049840381736892);
const ASIAN: CallPut = pair(3.2776906533992545, 7.067347664422634);
const QUANTO: CallPut = pair(8.111646452569918, 12.84376640716852);
const BS2002: CallPut = pair(7.35628449811837, 10.292928328280922);
const LAMBDA_CALL: f64 = 6.625327774385489;
const LAMBDA_PUT: f64 = -4.987062420615437;
const BSM: [(BsmCoc, CallPut, CallPut); 5] = [
  (
    BsmCoc::Bsm1973,
    pair(8.113484530210712, 9.248898390896997),
    pair(0.0, 5.0),
  ),
  (
    BsmCoc::Merton1973,
    pair(7.356284457997674, 9.980504358377694),
    pair(0.0, 5.0),
  ),
  (
    BsmCoc::Black1976,
    pair(6.317127602016328, 11.133099690620426),
    pair(0.0, 5.0),
  ),
  (
    BsmCoc::Asay1982,
    pair(6.317127602016328, 11.133099690620426),
    pair(0.0, 5.0),
  ),
  (
    BsmCoc::GarmanKohlhagen1983,
    pair(7.356284457997674, 9.980504358377694),
    pair(0.0, 5.0),
  ),
];

fn heston() -> HestonPricer {
  HestonPricer::new(HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  })
}

fn legs(model: &impl ModelPricer, tau: f64) -> CallPut {
  let call = model.price_call(S, K, R, Q, tau);
  assert_eq!(model.price_option(S, K, R, Q, tau, OptionType::Call), call);
  let put = model.price_put(S, K, R, Q, tau);
  assert_eq!(model.price_option(S, K, R, Q, tau, OptionType::Put), put);
  pair(call, put)
}

/// A pinned value went through libm, whose last bits differ between platforms; each leg is held at
/// its own scale.
fn assert_pinned(got: CallPut, pinned: CallPut, what: &str) {
  let close = |a: f64, b: f64| (a - b).abs() <= 1e-13 * b.abs().max(1.0);
  let both = close(got.call, pinned.call) && close(got.put, pinned.put);
  assert!(both, "{what}: {got:?} against {pinned:?}");
}

fn assert_pinned_value(got: f64, pinned: f64, what: &str) {
  let close = (got - pinned).abs() <= 1e-13 * pinned.abs().max(1.0);
  assert!(close, "{what}: {got} against {pinned}");
}

#[test]
fn the_order_book_answers_with_structs() {
  let mut book = OrderBook::new();
  let placed: Placement = book.add_order(Side::Sell, 101.0, 5.0);
  assert!(placed.trades.is_empty());
  let hit: Execution = book.execute_order(Side::Buy, 7.0);
  assert_eq!(hit.trades.len(), 1);
  assert_eq!(hit.unfilled, 2.0);
  assert_eq!(hit.trades[0].maker_id, placed.order_id);
  assert!(book.best_ask().is_none());
  book.add_order(Side::Buy, 99.0, 1.0);
  book.add_order(Side::Buy, 98.5, 2.0);
  book.add_order(Side::Sell, 102.0, 3.0);
  let level = |price, size| Level { price, size };
  assert_eq!(book.best_bid(), Some(level(99.0, 1.0)));
  assert_eq!(book.best_ask(), Some(level(102.0, 3.0)));
  let depth: Depth = book.depth();
  assert_eq!(depth.bids, [level(99.0, 1.0), level(98.5, 2.0)]);
  assert_eq!(depth.asks, [level(102.0, 3.0)]);
  let crossing = book.add_order(Side::Buy, 102.0, 4.0);
  assert_eq!(crossing.trades[0].size, 3.0);
  assert_eq!(book.best_bid(), Some(level(102.0, 1.0)));
}

#[test]
fn the_surface_takes_an_option_type() {
  let model = heston();
  let strikes = vec![90.0, 100.0, 110.0];
  let maturities = vec![0.5, 1.0];
  let prices = ndarray::Array2::from_shape_fn((2, 3), |(j, i)| {
    model.price_put(S, strikes[i], R, Q, maturities[j]) * (R * maturities[j]).exp()
  });
  let forwards: Vec<f64> = maturities
    .iter()
    .map(|&t| S * ((R - Q) * t).exp())
    .collect();
  let surface = ImpliedVolSurface::from_prices(
    strikes.clone(),
    maturities.clone(),
    forwards.clone(),
    prices.view(),
    OptionType::Put,
  );
  assert!(
    surface.ivs.iter().all(|v| *v > 0.1 && *v < 0.4),
    "{:?}",
    surface.ivs
  );
  let built = build_surface(
    strikes,
    maturities,
    forwards,
    prices.view(),
    OptionType::Put,
  );
  assert_eq!(built.iv_surface.ivs, surface.ivs);
}

#[test]
fn a_chain_quote_names_its_option_type() {
  let quote = ChainQuote {
    strike: 100.0,
    tau: 0.5,
    last: 4.0,
    bid: 3.9,
    ask: 4.1,
    implied_vol: 0.2,
    option_type: OptionType::Put,
  };
  assert_eq!(quote.option_type, OptionType::Put);
}

#[test]
fn one_computation_answers_a_named_pair() {
  let heston = heston();
  let heston_pair = pair(5.164008255606973, 7.7882281559869995);
  let named = heston.price_call_put(S, K, R, Q, TAU);
  assert_eq!(named, legs(&heston, TAU));
  assert_pinned(named, heston_pair, "heston");
  assert_pinned(
    heston.price_call_put(S, K, R, Q, 0.0),
    pair(0.0, 5.0),
    "heston at expiry",
  );
  let CallPut { call, put } = heston
    .with_vol_risk_premium(0.1)
    .price_call_put(S, K, R, Q, TAU);
  assert_pinned(pair(call, put), HESTON_LAMBDA, "heston lambda");
  let vega = heston.initial_variance_vega(S, K, R, Q, TAU);
  assert_pinned_value(vega, HESTON_VEGA, "heston vega");

  let merton = Merton1976Pricer::new(0.25, 1.0, 0.4, 50, BsmCoc::Merton1973);
  let named = merton.price_call_put(S, K, R, Q, TAU);
  assert_eq!(named, legs(&merton, TAU));
  assert_pinned(named, MERTON, "merton");

  let hscm = HestonStochCorrPricer::new(0.04, 1.5, 0.04, 0.3, -0.5, 2.0, -0.4, 0.2, 0.3);
  let named = hscm.price_call_put(S, K, R, Q, TAU);
  assert_eq!(named, legs(&hscm, TAU));
  assert_pinned(named, HSCM, "hscm");

  let malliavin = GbmMalliavinPricer::new(0.2, 400, 64, 0.25, Deterministic::new(7));
  let named = malliavin.price_call_put(S, K, R, Q, TAU);
  assert_eq!(named, legs(&malliavin, TAU));
  assert_pinned(named, MALLIAVIN, "malliavin");
  let localized = malliavin.price_call_put_localized(S, K, R, Q, TAU);
  assert_pinned(localized, MALLIAVIN_LOCALIZED, "malliavin localized");
  let expired = malliavin.price_call_put(S, K, R, Q, 0.0);
  assert_pinned(expired, pair(0.0, 5.0), "malliavin at expiry");

  let kirk = KirkSpreadPricer::new(0.30, 0.25, 0.7);
  let named = kirk.spread_call_put(100.0, 90.0, 5.0, 0.05, 0.5);
  assert_pinned(named, KIRK, "kirk");
  assert_eq!(kirk.spread_call(100.0, 90.0, 5.0, 0.05, 0.5), named.call);
  assert_eq!(kirk.spread_put(100.0, 90.0, 5.0, 0.05, 0.5), named.put);
}

#[test]
fn a_closed_form_prices_each_leg_through_the_trait() {
  for (coc, at_tau, at_expiry) in BSM {
    let model = BsmPricer::new(0.25, coc);
    assert_pinned(legs(&model, TAU), at_tau, &format!("{coc:?}"));
    assert_pinned(legs(&model, 0.0), at_expiry, &format!("{coc:?} at expiry"));
  }
  let sabr = SabrPricer::new(SabrParams::new(0.2, 1.0, -0.3, 0.4));
  assert_pinned(legs(&sabr, TAU), SABR, "sabr");
  let sabr_beta = SabrPricer::new(SabrParams::new(2.0, 0.5, -0.3, 0.4));
  assert_pinned(legs(&sabr_beta, TAU), SABR_BETA, "sabr beta");
  assert_pinned(
    legs(&BachelierPricer::new(20.0), TAU),
    BACHELIER,
    "bachelier",
  );
  assert_pinned(legs(&AsianPricer::new(0.25), TAU), ASIAN, "asian");
  let quanto = QuantoPricer::new(0.2, 0.12, 0.3, 0.05, 1.5);
  assert_pinned(legs(&quanto, TAU), QUANTO, "quanto");
  let bs2002 = BjerksundStensland2002Pricer::new(0.25);
  assert_pinned(legs(&bs2002, TAU), BS2002, "bs2002");
}

#[test]
fn the_other_carries_and_the_edges_keep_their_values() {
  let at = |model: &dyn ModelPricer, s, k, r, q, tau| {
    pair(
      model.price_call(s, k, r, q, tau),
      model.price_put(s, k, r, q, tau),
    )
  };
  let no_dividend = pair(7.905150880530156, 9.040564741216432);
  let futures = pair(6.127093624624524, 10.943065713228627);
  for (coc, pinned) in [
    (BsmCoc::Bsm1973, no_dividend),
    (BsmCoc::Black1976, futures),
    (BsmCoc::Asay1982, futures),
    (BsmCoc::GarmanKohlhagen1983, MERTON),
  ] {
    let merton = Merton1976Pricer::new(0.25, 1.0, 0.4, 50, coc);
    let named = merton.price_call_put(S, K, R, Q, TAU);
    assert_eq!(named, legs(&merton, TAU), "{coc:?}");
    assert_pinned(named, pinned, &format!("{coc:?}"));
  }
  let merton = Merton1976Pricer::new(0.25, 1.0, 0.4, 50, BsmCoc::Merton1973);
  let expired = merton.price_call_put(S, K, R, Q, 0.0);
  assert_pinned(expired, pair(0.0, 5.0), "merton at expiry");
  let no_jumps = Merton1976Pricer::new(0.25, 0.0, 0.4, 50, BsmCoc::Merton1973);
  let black_scholes = pair(7.356284457997674, 9.980504358377694);
  assert_pinned(
    no_jumps.price_call_put(S, K, R, Q, TAU),
    black_scholes,
    "merton without jumps",
  );

  let below_zero = at(&BachelierPricer::new(20.0), -5.0, 2.0, R, Q, TAU);
  assert_pinned(
    below_zero,
    pair(3.7831765246338898, 10.635125058090846),
    "bachelier",
  );
  let in_the_money = at(
    &BjerksundStensland2002Pricer::new(0.35),
    42.0,
    40.0,
    0.04,
    0.08,
    0.75,
  );
  assert_pinned(
    in_the_money,
    pair(5.286858855569344, 4.361461686435224),
    "bs2002",
  );
  let malliavin = GbmMalliavinPricer::new(0.2, 400, 64, 0.25, Deterministic::new(7));
  let zero_spot = malliavin.price_call_put_localized(0.0, K, R, Q, TAU);
  assert_pinned(
    zero_spot,
    pair(0.0, 101.13541386068628),
    "malliavin at zero spot",
  );
  let hscm = HestonStochCorrPricer::new(0.04, 1.5, 0.04, 0.3, -0.5, 2.0, -0.4, 0.2, 0.3);
  let far_out = hscm.price_call_put(S, 400.0, R, Q, 0.05);
  assert_pinned(
    far_out,
    pair(3.3428110535475566e-13, 299.10119897564687),
    "hscm",
  );
}

#[test]
fn lambda_is_the_elasticity_of_the_option_it_is_asked_for() {
  let bsm = BsmPricer::new(0.25, BsmCoc::Merton1973);
  let call = bsm.lambda(S, K, R, Q, TAU, OptionType::Call);
  assert_pinned_value(call, LAMBDA_CALL, "call");
  let lambda_put = bsm.lambda(S, K, R, Q, TAU, OptionType::Put);
  assert_pinned_value(lambda_put, LAMBDA_PUT, "put");
  let put = bsm.price_put(S, K, R, Q, TAU);
  let delta = bsm.delta(S, K, R, Q, TAU, OptionType::Put);
  assert_eq!(lambda_put, delta * S / put);
}

#[test]
fn the_vocabulary_is_one_word_per_concept() {
  let bsm = BsmPricer {
    sigma: 0.2,
    b: BsmCoc::Merton1973,
  };
  let greeks = bsm.greeks(S, K, R, Q, TAU, OptionType::Call);
  assert_eq!(greeks.volga, bsm.volga(S, K, R, Q, TAU));
  assert_eq!(greeks.veta, bsm.veta(S, K, R, Q, TAU));
  let merton = Merton1976Pricer::new(0.25, 1.0, 0.4, 50, BsmCoc::Merton1973);
  assert_eq!(merton.sigma, 0.25);
  assert_eq!(
    merton.greeks(S, K, R, Q, TAU, OptionType::Put).volga,
    merton.volga(S, K, R, Q, TAU)
  );
  assert_eq!(BachelierPricer::new(20.0).sigma, 20.0);
  assert_eq!(AsianPricer::new(0.25).sigma, 0.25);
  assert_eq!(BjerksundStensland2002Pricer::new(0.25).sigma, 0.25);
  assert_eq!(SnellEnvelopePricer::new(0.25, 50).sigma, 0.25);
  assert_eq!(GbmMalliavinPricer::new(0.2, 8, 4, 0.5, Unseeded).sigma, 0.2);
  let quanto = QuantoPricer {
    sigma: 0.2,
    sigma_fx: 0.12,
    rho: 0.3,
    r_f: 0.05,
    fixed_rate: 1.5,
  };
  assert_eq!(quanto, QuantoPricer::new(0.2, 0.12, 0.3, 0.05, 1.5));

  let forward = FxForward::new(
    CurrencyPair::new(currency::EUR, currency::USD),
    1.1,
    0.03,
    0.01,
    1.0,
  );
  assert_eq!((forward.r_d, forward.r_f), (0.03, 0.01));
  let quote = SwaptionQuote {
    tau: 1.0,
    tenor: 2.0,
    black_vol: 0.2,
    fixed_accrual: 0.5,
    direction: SwaptionDirection::Payer,
    weight: None,
  };
  assert_eq!(quote.tau, 1.0);
}

#[test]
fn the_jump_models_hold_their_params_privately() {
  let svj = SvjParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    lambda: 0.5,
    mu_j: -0.05,
    sigma_j: 0.1,
  };
  let bates = BatesFourier::new(svj);
  assert_eq!(bates.params(), svj);
  assert_pinned_value(
    bates.price_call(S, K, R, Q, TAU),
    5.6848296939312775,
    "bates",
  );
  let hkde_params = HkdeParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
    lambda: 0.5,
    p_up: 0.4,
    eta1: 10.0,
    eta2: 5.0,
  };
  let hkde = HkdeFourier::new(hkde_params);
  assert_eq!(hkde.params(), hkde_params);
  assert_pinned_value(hkde.price_call(S, K, R, Q, TAU), 6.559356757760838, "hkde");
}

#[test]
#[should_panic(expected = "sigma must satisfy `sigma.is_finite() && sigma > 0.0`, got sigma = 0.0")]
fn a_jump_model_names_its_vol_of_variance_sigma() {
  let _ = BatesFourier::new(SvjParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.0,
    rho: -0.7,
    lambda: 0.5,
    mu_j: -0.05,
    sigma_j: 0.1,
  });
}

#[test]
fn a_monte_carlo_estimate_is_a_plain_value() {
  fn plain<T: Copy + PartialEq>() {}
  plain::<stochastic_rs_stochastic::mc::McEstimate<f64>>();
}
