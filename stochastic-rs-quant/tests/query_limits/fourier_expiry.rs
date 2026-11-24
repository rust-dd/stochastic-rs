//! The Fourier entry points at and past expiry: the payoff at `tau = 0`, NaN for a `tau` that is
//! negative, NaN or infinite.

use stochastic_rs_quant::OptionType;
use stochastic_rs_quant::pricing::fourier::BsmFourier;
use stochastic_rs_quant::pricing::fourier::CarrMadanPricer;
use stochastic_rs_quant::pricing::fourier::FrftCarrMadanPricer;
use stochastic_rs_quant::pricing::fourier::GilPelaezPricer;
use stochastic_rs_quant::pricing::fourier::HestonFourier;
use stochastic_rs_quant::pricing::fourier::LewisPricer;
use stochastic_rs_quant::pricing::fourier::RoughHeston;
use stochastic_rs_quant::pricing::fourier::RoughHestonCf;
use stochastic_rs_quant::pricing::fourier::RoughHestonParams;
use stochastic_rs_quant::pricing::heston::HestonParams;
use stochastic_rs_quant::traits::ModelPricer;

const K: f64 = 100.0;
const R: f64 = 0.05;
const Q: f64 = 0.02;
const SPOTS: [f64; 3] = [90.0, 100.0, 110.0];
const PAST: [f64; 3] = [-1.0, f64::NAN, f64::INFINITY];
const BSM: BsmFourier = BsmFourier { sigma: 0.2 };

fn payoff(option_type: OptionType, s: f64, k: f64) -> f64 {
  match option_type {
    OptionType::Call => (s - k).max(0.0),
    OptionType::Put => (k - s).max(0.0),
  }
}

fn assert_expiry_rule(what: &str, price: impl Fn(OptionType, f64, f64) -> f64) {
  for option_type in [OptionType::Call, OptionType::Put] {
    for s in SPOTS {
      let (got, want) = (price(option_type, s, 0.0), payoff(option_type, s, K));
      assert_eq!(got, want, "{what} {option_type:?} at s = {s}, tau = 0");
      for tau in PAST {
        let got = price(option_type, s, tau);
        assert!(
          got.is_nan(),
          "{what} {option_type:?} at s = {s}, tau = {tau}: {got}"
        );
      }
    }
  }
}

/// The put a caller forms by parity on `call`, which is exact at these spots.
fn by_parity(call: f64, option_type: OptionType, s: f64, tau: f64) -> f64 {
  match option_type {
    OptionType::Call => call,
    OptionType::Put => call - s * (-Q * tau).exp() + K * (-R * tau).exp(),
  }
}

#[test]
fn the_blanket_model_pricer_follows_the_expiry_rule() {
  assert_expiry_rule("blanket", |option_type, s, tau| {
    BSM.price_option(s, K, R, Q, tau, option_type)
  });
  let heston = HestonFourier::new(HestonParams {
    v0: 0.04,
    kappa: 2.0,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  });
  assert_expiry_rule("blanket, Heston", |option_type, s, tau| {
    heston.price_option(s, K, R, Q, tau, option_type)
  });
  let rough = RoughHestonParams::new(0.04, 1.0, 0.04, 0.4, -0.65, 0.1);
  for cf in [
    RoughHestonCf::Adams { steps: 8 },
    RoughHestonCf::default(),
    RoughHestonCf::LIFTED,
  ] {
    let model = RoughHeston::new(rough, cf);
    assert_expiry_rule("blanket, rough Heston", |option_type, s, tau| {
      model.price_option(s, K, R, Q, tau, option_type)
    });
  }
}

#[test]
fn the_quadrature_pricers_follow_the_expiry_rule() {
  assert_expiry_rule("Lewis", |option_type, s, tau| {
    let call = LewisPricer::price_call(&BSM, s, K, R, Q, tau);
    by_parity(call, option_type, s, tau)
  });
  assert_expiry_rule("Gil-Pelaez", |option_type, s, tau| {
    let call = GilPelaezPricer::price_call(&BSM, s, K, R, Q, tau);
    by_parity(call, option_type, s, tau)
  });
}

#[test]
fn the_transform_pricers_follow_the_expiry_rule() {
  let carr_madan = CarrMadanPricer::new(4096, 1.5);
  assert_expiry_rule("Carr-Madan", |option_type, s, tau| match option_type {
    OptionType::Call => carr_madan.price_call(&BSM, s, K, R, Q, tau),
    OptionType::Put => carr_madan.price_put(&BSM, s, K, R, Q, tau),
  });
  let fractional = FrftCarrMadanPricer::new(512, 1.5, 0.25, 0.01);
  assert_expiry_rule("FRFT", |option_type, s, tau| match option_type {
    OptionType::Call => fractional.price_call(&BSM, s, K, R, Q, tau),
    OptionType::Put => fractional.price_put(&BSM, s, K, R, Q, tau),
  });
}

#[test]
fn a_transform_surface_is_its_nodes_payoffs_at_expiry_and_nan_past_it() {
  let carr_madan = CarrMadanPricer::new(256, 1.5);
  let fractional = FrftCarrMadanPricer::new(256, 1.5, 0.25, 0.01);
  let surface = |which: usize, s: f64, tau: f64| match which {
    0 => carr_madan.price_call_surface(&BSM, s, R, Q, tau),
    _ => fractional.price_call_surface(&BSM, s, R, Q, tau),
  };
  for which in [0, 1] {
    let s = if which == 0 { 1.1 } else { 110.0 };
    let (log_strikes, prices) = surface(which, s, 0.0);
    assert_eq!((log_strikes.len(), prices.len()), (256, 256));
    let (below, above) = (log_strikes[0].exp(), log_strikes[255].exp());
    assert!(below < s && s < above, "{which}: {below} {s} {above}");
    for (log_strike, price) in log_strikes.iter().zip(&prices) {
      assert_eq!(*price, (s - log_strike.exp()).max(0.0), "{which}");
    }
    for tau in PAST {
      let (log_strikes, prices) = surface(which, s, tau);
      assert_eq!(log_strikes.len(), 256);
      assert!(prices.iter().all(|p| p.is_nan()), "{which} at tau = {tau}");
    }
  }
}

/// The fractional grid is centred on `ln F_T`, which a spot that is not positive or a maturity
/// that is not finite leaves without a finite value: no node has a strike, and none a price.
#[test]
fn a_fractional_surface_off_the_domain_has_no_finite_log_strike() {
  let fractional = FrftCarrMadanPricer::new(256, 1.5, 0.25, 0.01);
  for (s, tau) in [
    (0.0, 1.0),
    (-5.0, 1.0),
    (f64::NAN, 1.0),
    (f64::INFINITY, 1.0),
    (f64::INFINITY, 0.0),
    (110.0, f64::NAN),
    (110.0, f64::INFINITY),
  ] {
    let (log_strikes, prices) = fractional.price_call_surface(&BSM, s, R, Q, tau);
    assert_eq!((log_strikes.len(), prices.len()), (256, 256));
    assert!(
      log_strikes.iter().all(|k| !k.is_finite()) && prices.iter().all(|p| p.is_nan()),
      "({s}, {tau}): {} to {}, {}",
      log_strikes[0],
      log_strikes[255],
      prices[128]
    );
  }
}

/// At expiry a spot of 0 puts every node at the strike `e^{−∞} = 0`, whose payoff `(0 − 0)⁺`
/// stands as any payoff does.
#[test]
fn a_fractional_surface_at_expiry_pays_a_zero_spot_its_payoff() {
  let fractional = FrftCarrMadanPricer::new(256, 1.5, 0.25, 0.01);
  let (log_strikes, prices) = fractional.price_call_surface(&BSM, 0.0, R, Q, 0.0);
  assert!(log_strikes.iter().all(|k| *k == f64::NEG_INFINITY));
  assert!(prices.iter().all(|p| *p == 0.0));
  assert_eq!((log_strikes.len(), prices.len()), (256, 256));
}

/// Parity on a payoff rounds: `(1.1 − 0.3) − 1.1 + 0.3` is `−5.6e-17`, and the put is 0.
#[test]
fn an_expired_put_is_its_payoff_and_not_parity_on_the_calls() {
  let (s, k) = (1.1, 0.3);
  let carr_madan = CarrMadanPricer::new(4096, 1.5);
  let fractional = FrftCarrMadanPricer::new(512, 1.5, 0.25, 0.01);
  let puts = [
    BSM.price_put(s, k, R, Q, 0.0),
    carr_madan.price_put(&BSM, s, k, R, Q, 0.0),
    fractional.price_put(&BSM, s, k, R, Q, 0.0),
  ];
  assert_eq!(puts, [0.0; 3]);
  assert_eq!(BSM.price_put(k, s, R, Q, 0.0), s - k);
}

#[test]
fn an_expired_fourier_price_is_nan_for_a_nan_rate_spot_or_strike() {
  let carr_madan = CarrMadanPricer::new(4096, 1.5);
  let fractional = FrftCarrMadanPricer::new(512, 1.5, 0.25, 0.01);
  for (s, k, r, q) in [
    (f64::NAN, K, R, Q),
    (110.0, f64::NAN, R, Q),
    (110.0, K, f64::NAN, Q),
    (110.0, K, R, f64::NAN),
  ] {
    let prices = [
      BSM.price_call(s, k, r, q, 0.0),
      BSM.price_put(s, k, r, q, 0.0),
      LewisPricer::price_call(&BSM, s, k, r, q, 0.0),
      GilPelaezPricer::price_call(&BSM, s, k, r, q, 0.0),
      carr_madan.price_call(&BSM, s, k, r, q, 0.0),
      carr_madan.price_put(&BSM, s, k, r, q, 0.0),
      fractional.price_call(&BSM, s, k, r, q, 0.0),
      fractional.price_put(&BSM, s, k, r, q, 0.0),
    ];
    assert!(
      prices.iter().all(|p| p.is_nan()),
      "({s}, {k}, {r}, {q}): {prices:?}"
    );
  }
}

/// Before expiry a lognormal price has no value at a spot or strike that is zero, negative or
/// infinite; at expiry the payoff is still defined.
#[test]
fn a_fourier_price_is_nan_before_expiry_for_a_spot_or_strike_that_is_not_positive() {
  let carr_madan = CarrMadanPricer::new(4096, 1.5);
  let fractional = FrftCarrMadanPricer::new(512, 1.5, 0.25, 0.01);
  for (s, k) in [
    (0.0, K),
    (-5.0, K),
    (f64::INFINITY, K),
    (110.0, 0.0),
    (110.0, -5.0),
    (110.0, f64::INFINITY),
  ] {
    let prices = [
      BSM.price_call(s, k, R, Q, 1.0),
      BSM.price_put(s, k, R, Q, 1.0),
      LewisPricer::price_call(&BSM, s, k, R, Q, 1.0),
      GilPelaezPricer::price_call(&BSM, s, k, R, Q, 1.0),
      carr_madan.price_call(&BSM, s, k, R, Q, 1.0),
      carr_madan.price_put(&BSM, s, k, R, Q, 1.0),
      fractional.price_call(&BSM, s, k, R, Q, 1.0),
      fractional.price_put(&BSM, s, k, R, Q, 1.0),
    ];
    assert!(prices.iter().all(|p| p.is_nan()), "({s}, {k}): {prices:?}");
  }
  for s in [0.0, -5.0] {
    let (_, prices) = carr_madan.price_call_surface(&BSM, s, R, Q, 1.0);
    assert!(prices.iter().all(|p| p.is_nan()), "Carr-Madan at s = {s}");
    let (_, prices) = fractional.price_call_surface(&BSM, s, R, Q, 1.0);
    assert!(prices.iter().all(|p| p.is_nan()), "FRFT at s = {s}");
  }
  assert_eq!(BSM.price_call(110.0, -5.0, R, Q, 0.0), 115.0);
  assert_eq!(BSM.price_put(-5.0, K, R, Q, 0.0), 105.0);
}

/// Before expiry a rate or yield that is not finite has no price, as in COS, and an infinite spot
/// no surface.
#[test]
fn a_fourier_price_is_nan_before_expiry_for_a_rate_that_is_not_finite() {
  let carr_madan = CarrMadanPricer::new(4096, 1.5);
  let fractional = FrftCarrMadanPricer::new(512, 1.5, 0.25, 0.01);
  let (inf, nan) = (f64::INFINITY, f64::NAN);
  let mut live = Vec::new();
  for (s, r, q) in [
    (110.0, inf, Q),
    (110.0, -inf, Q),
    (110.0, nan, Q),
    (110.0, R, inf),
    (110.0, R, -inf),
    (110.0, R, nan),
    (inf, R, Q),
  ] {
    let prices = [
      ("blanket call", BSM.price_call(s, K, r, q, 1.0)),
      ("blanket put", BSM.price_put(s, K, r, q, 1.0)),
      ("Lewis", LewisPricer::price_call(&BSM, s, K, r, q, 1.0)),
      (
        "Gil-Pelaez",
        GilPelaezPricer::price_call(&BSM, s, K, r, q, 1.0),
      ),
      (
        "Carr-Madan call",
        carr_madan.price_call(&BSM, s, K, r, q, 1.0),
      ),
      (
        "Carr-Madan put",
        carr_madan.price_put(&BSM, s, K, r, q, 1.0),
      ),
      ("FRFT call", fractional.price_call(&BSM, s, K, r, q, 1.0)),
      ("FRFT put", fractional.price_put(&BSM, s, K, r, q, 1.0)),
    ];
    for (name, price) in prices.into_iter().filter(|(_, p)| !p.is_nan()) {
      live.push(format!("{name} at ({s}, {r}, {q}): {price}"));
    }
    for (name, (_, surface)) in [
      (
        "Carr-Madan",
        carr_madan.price_call_surface(&BSM, s, r, q, 1.0),
      ),
      ("FRFT", fractional.price_call_surface(&BSM, s, r, q, 1.0)),
    ] {
      let nodes = surface.iter().filter(|p| !p.is_nan()).count();
      if nodes > 0 {
        live.push(format!("{name} surface at ({s}, {r}, {q}): {nodes} nodes"));
      }
    }
  }
  assert!(live.is_empty(), "{live:#?}");
}
