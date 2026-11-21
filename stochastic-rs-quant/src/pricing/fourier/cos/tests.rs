use super::*;
use crate::pricing::bsm::BsmCoc;
use crate::pricing::bsm::BsmPricer;
use crate::pricing::fourier::BsmFourier;
use crate::pricing::fourier::GilPelaezPricer;
use crate::pricing::fourier::HestonFourier;
use crate::pricing::fourier::KouFourier;
use crate::pricing::heston::HestonParams;
use crate::pricing::heston::HestonPricer;
use crate::traits::ModelPricer;

/// Fang & Oosterlee (2008) §5.1: COS at `N = 256, L = 10` reproduces the closed-form BSM call.
#[test]
fn cos_bsm_matches_analytic() {
  let model = BsmFourier { sigma: 0.25 };
  let expected = BsmPricer::new(0.25, BsmCoc::Bsm1973).price_call(100.0, 110.0, 0.05, 0.0, 1.0);
  let price = CosEngine::default().price(&model, 100.0, 110.0, 0.05, 0.0, 1.0, OptionType::Call);
  assert!(
    (price - expected).abs() < 1e-10,
    "COS BSM: got={price}, expected={expected}"
  );
}

#[test]
fn cos_heston_matches_quadrature() {
  let model = HestonFourier::new(HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  });
  let reference = HestonPricer::new(HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  })
  .price_call(100.0, 100.0, 0.05, 0.0, 1.0);
  // A wider `L` under-resolves the density at `N = 256` (`L = 40` misses by 1.5e-4).
  let price = CosEngine::default().price(&model, 100.0, 100.0, 0.05, 0.0, 1.0, OptionType::Call);
  let gil_pelaez = GilPelaezPricer::price_call(&model, 100.0, 100.0, 0.05, 0.0, 1.0);
  assert!(
    (price - gil_pelaez).abs() < 1e-6,
    "COS vs Gil-Pelaez (same chf): cos={price}, gil_pelaez={gil_pelaez}"
  );
  assert!(
    (price - reference).abs() < 1e-6,
    "COS Heston: got={price}, expected={reference}"
  );
}

/// A chf still above `1e-9` at term 256 (`σ = 1`, `ρ = −0.9`, widened by `√c4`) takes more
/// terms by default; an explicit `n` is the sum of exactly `n` terms.
#[test]
fn the_default_adds_terms_until_the_chf_has_decayed() {
  let model = HestonFourier::new(HestonParams {
    v0: 0.02,
    kappa: 0.5,
    theta: 0.09,
    sigma: 1.0,
    rho: -0.9,
  });
  for k in [80.0, 100.0, 120.0] {
    let want = model.price_put(100.0, k, 0.05, 0.01, 2.0);
    let adaptive = CosEngine::default().price(&model, 100.0, k, 0.05, 0.01, 2.0, OptionType::Put);
    let fixed = CosEngine::new(256, 10.0).price(&model, 100.0, k, 0.05, 0.01, 2.0, OptionType::Put);
    assert!(
      (adaptive - want).abs() < 5e-6,
      "K={k}: {adaptive} vs {want}"
    );
    assert!((fixed - want).abs() > 1e-3, "K={k}: {fixed} vs {want}");
  }
}

#[test]
fn the_default_stops_at_256_terms_where_the_chf_has_decayed() {
  let model = HestonFourier::new(HestonParams {
    v0: 0.04,
    kappa: 1.5,
    theta: 0.04,
    sigma: 0.3,
    rho: -0.7,
  });
  let adaptive = CosEngine::default().price(&model, 100.0, 100.0, 0.05, 0.0, 1.0, OptionType::Put);
  let fixed =
    CosEngine::new(256, 10.0).price(&model, 100.0, 100.0, 0.05, 0.0, 1.0, OptionType::Put);
  assert_eq!(adaptive, fixed);
}

#[test]
fn cos_kou_matches_gil_pelaez() {
  let model = KouFourier {
    sigma: 0.15,
    lambda: 3.0,
    p_up: 0.2,
    eta1: 25.0,
    eta2: 10.0,
  };
  let reference = GilPelaezPricer::price_call(&model, 100.0, 100.0, 0.05, 0.01, 1.0);
  let price = CosEngine::default().price(&model, 100.0, 100.0, 0.05, 0.01, 1.0, OptionType::Call);
  assert!(
    (price - reference).abs() < 1e-4,
    "COS Kou: got={price}, expected={reference}"
  );
}

#[test]
fn cos_put_call_parity() {
  let model = BsmFourier { sigma: 0.2 };
  let engine = CosEngine::default();
  let call = engine.price(&model, 100.0, 95.0, 0.05, 0.03, 0.5, OptionType::Call);
  let put = engine.price(&model, 100.0, 95.0, 0.05, 0.03, 0.5, OptionType::Put);
  let parity = 100.0 * (-0.03_f64 * 0.5).exp() - 95.0 * (-0.05_f64 * 0.5).exp();
  assert!(
    (call - put - parity).abs() < 1e-8,
    "put-call parity: call={call}, put={put}, parity={parity}"
  );
}

#[test]
fn cos_converges_in_n() {
  let model = BsmFourier { sigma: 0.25 };
  let expected = BsmPricer::new(0.25, BsmCoc::Bsm1973).price_call(100.0, 110.0, 0.05, 0.0, 1.0);
  let err_small =
    (CosEngine::new(64, 10.0).price(&model, 100.0, 110.0, 0.05, 0.0, 1.0, OptionType::Call)
      - expected)
      .abs();
  let err_large =
    (CosEngine::new(512, 10.0).price(&model, 100.0, 110.0, 0.05, 0.0, 1.0, OptionType::Call)
      - expected)
      .abs();
  assert!(
    err_small >= err_large,
    "convergence in N: err(64)={err_small}, err(512)={err_large}"
  );
}

/// Degenerate maturity `t=0` collapses `c2` (and `c4`) to zero, so the
/// truncation half-width is zero — the crate convention is `NaN`, never a
/// silently-wrong `0.0` (see `carr_madan_out_of_grid_returns_nan_not_zero`
/// in `super::super::tests` for the same convention elsewhere in this module).
#[test]
fn cos_invalid_cumulants_gives_nan() {
  let model = BsmFourier { sigma: 0.25 };
  let price = CosEngine::default().price(&model, 100.0, 100.0, 0.05, 0.0, 0.0, OptionType::Call);
  assert!(
    price.is_nan(),
    "degenerate t=0 cumulants must give NaN, got {price}"
  );
}

#[test]
#[should_panic(expected = "CosEngine requires l > 0 and n >= 2")]
fn cos_engine_new_rejects_nonpositive_l() {
  CosEngine::new(256, 0.0);
}

#[test]
#[should_panic(expected = "CosEngine requires l > 0 and n >= 2")]
fn cos_engine_new_rejects_too_few_terms() {
  CosEngine::new(1, 10.0);
}

/// A characteristic function that returns `NaN` (standing in for a
/// user-supplied model overflowing its own `chf`) must exit `price` as
/// `NaN`, not the `0.0` `f64::max` would silently produce by picking its
/// non-`NaN` operand — see [`CosEngine::price`]'s doc.
#[test]
fn cos_price_preserves_nan_from_chf_blowup() {
  struct NanModel;
  impl FourierModelExt for NanModel {
    fn chf(&self, _t: f64, _xi: Complex64) -> Complex64 {
      Complex64::new(f64::NAN, f64::NAN)
    }

    fn cumulants(&self, _t: f64) -> super::super::Cumulants {
      super::super::Cumulants {
        c1: 0.0,
        c2: 0.04,
        c4: Some(0.0),
      }
    }
  }
  let price = CosEngine::default().price(&NanModel, 100.0, 100.0, 0.05, 0.0, 1.0, OptionType::Call);
  assert!(
    price.is_nan(),
    "NaN chf must propagate to a NaN price, got {price}"
  );
}

/// Le Floc'h (2020), arXiv:2005.13248, §3: a call's coefficients grow like `e^b`, so over a wide
/// range a call stays at the closed form only through the put and parity.
#[test]
fn cos_call_holds_over_a_wide_truncation_range() {
  let model = BsmFourier { sigma: 0.2 };
  let bsm = BsmPricer::new(0.2, BsmCoc::Merton1973);
  let engine = CosEngine::new(4096, 100.0);
  for k in [80.0, 100.0, 120.0] {
    let call = engine.price(&model, 100.0, k, 0.05, 0.01, 1.0, OptionType::Call);
    let want = bsm.price_call(100.0, k, 0.05, 0.01, 1.0);
    assert!((call - want).abs() < 1e-10, "K={k}: {call} vs {want}");
  }
}

/// Where 0 falls outside [a, b] the payoff coefficients must clamp to the payoff's support.
#[test]
fn cos_matches_closed_form_at_wing_strikes() {
  let model = BsmFourier { sigma: 0.2 };
  let bsm = BsmPricer::new(0.2, BsmCoc::Merton1973);
  let engine = CosEngine::default();
  for k in [40.0, 250.0] {
    let call = engine.price(&model, 100.0, k, 0.05, 0.01, 0.05, OptionType::Call);
    let put = engine.price(&model, 100.0, k, 0.05, 0.01, 0.05, OptionType::Put);
    let (want_c, want_p) = (
      bsm.price_call(100.0, k, 0.05, 0.01, 0.05),
      bsm.price_put(100.0, k, 0.05, 0.01, 0.05),
    );
    assert!(
      (call - want_c).abs() < 1e-9,
      "K={k} call: {call} vs {want_c}"
    );
    assert!((put - want_p).abs() < 1e-9, "K={k} put: {put} vs {want_p}");
  }
}

fn fang_oosterlee_heston() -> HestonFourier {
  HestonFourier::new(HestonParams {
    v0: 0.0175,
    kappa: 1.5768,
    theta: 0.0398,
    sigma: 0.5751,
    rho: -0.5711,
  })
}

/// A strike array is `price` at each strike, bit for bit: both sum the same terms in one order.
fn assert_strikes_are_prices<M: FourierModelExt>(model: &M) {
  let strikes = ndarray::Array1::linspace(40.0, 250.0, 43);
  for engine in [CosEngine::default(), CosEngine::new(512, 12.0)] {
    for option_type in [OptionType::Call, OptionType::Put] {
      for t in [0.05, 1.0, 10.0] {
        let prices = engine.price_strikes(model, 100.0, strikes.view(), 0.05, 0.01, t, option_type);
        for (&price, &k) in prices.iter().zip(&strikes) {
          assert_eq!(
            price,
            engine.price(model, 100.0, k, 0.05, 0.01, t, option_type),
            "K={k}, T={t}"
          );
        }
      }
    }
  }
}

#[test]
fn price_strikes_is_price_at_every_strike() {
  assert_strikes_are_prices(&BsmFourier { sigma: 0.2 });
  assert_strikes_are_prices(&fang_oosterlee_heston());
  assert_strikes_are_prices(&KouFourier {
    sigma: 0.15,
    lambda: 3.0,
    p_up: 0.2,
    eta1: 25.0,
    eta2: 10.0,
  });
}

/// Fang & Oosterlee (2008) Table 2: three GBM calls in one computation, the reference values
/// printed to nine decimals.
#[test]
fn price_strikes_reproduces_the_gbm_strike_vector_of_fang_oosterlee() {
  let prices = CosEngine::default().price_strikes(
    &BsmFourier { sigma: 0.25 },
    100.0,
    ndarray::array![80.0, 100.0, 120.0].view(),
    0.1,
    0.0,
    0.1,
    OptionType::Call,
  );
  for (price, want) in prices.iter().zip([20.799226309, 3.659968453, 0.044577814]) {
    assert!((price - want).abs() < 1e-9, "{price} vs {want}");
  }
}

/// Fang & Oosterlee (2008) Tables 4–6: the 21 Heston strikes of Table 6, with `K = 100` at the
/// printed references; the `T = 1` one is a Carr–Madan value, held here to seven decimals.
#[test]
fn price_strikes_holds_the_heston_references_of_fang_oosterlee() {
  let strikes = ndarray::Array1::linspace(50.0, 150.0, 21);
  for (t, want, tolerance) in [(1.0, 5.785155450, 1e-7), (10.0, 22.318945791, 1e-9)] {
    let prices = CosEngine::default().price_strikes(
      &fang_oosterlee_heston(),
      100.0,
      strikes.view(),
      0.0,
      0.0,
      t,
      OptionType::Call,
    );
    assert_eq!(strikes[10], 100.0);
    assert!(
      (prices[10] - want).abs() < tolerance,
      "T={t}: {} vs {want}",
      prices[10]
    );
  }
}

/// One strike outside the domain is NaN alone; a range with no width is NaN at every strike.
#[test]
fn price_strikes_keeps_nan_to_the_strikes_it_belongs_to() {
  let model = BsmFourier { sigma: 0.25 };
  let engine = CosEngine::default();
  let strikes = ndarray::array![90.0, -1.0, 110.0];
  let prices = engine.price_strikes(
    &model,
    100.0,
    strikes.view(),
    0.05,
    0.0,
    1.0,
    OptionType::Put,
  );
  assert!(prices[1].is_nan());
  assert_eq!(
    prices[0],
    engine.price(&model, 100.0, 90.0, 0.05, 0.0, 1.0, OptionType::Put)
  );
  assert_eq!(
    prices[2],
    engine.price(&model, 100.0, 110.0, 0.05, 0.0, 1.0, OptionType::Put)
  );

  let degenerate = engine.price_strikes(
    &model,
    100.0,
    strikes.view(),
    0.05,
    0.0,
    0.0,
    OptionType::Put,
  );
  assert!(degenerate.iter().all(|price| price.is_nan()));
  let none = engine.price_strikes(
    &model,
    100.0,
    ndarray::Array1::<f64>::zeros(0).view(),
    0.05,
    0.0,
    1.0,
    OptionType::Put,
  );
  assert!(none.is_empty());
}
