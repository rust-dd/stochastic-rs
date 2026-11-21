use super::*;
use crate::pricing::bsm::BsmCoc;
use crate::pricing::bsm::BsmPricer;
use crate::pricing::fourier::BsmFourier;
use crate::pricing::fourier::GilPelaezPricer;
use crate::pricing::fourier::HestonFourier;
use crate::pricing::fourier::KouFourier;
use crate::pricing::fourier::MertonJdFourier;
use crate::pricing::fourier::NigFourier;
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

/// A zero volatility collapses `c2` and `c4` to zero and the truncation half-width with them:
/// NaN, never a silently wrong `0.0`.
#[test]
fn cos_invalid_cumulants_gives_nan() {
  let model = BsmFourier { sigma: 0.0 };
  let price = CosEngine::default().price(&model, 100.0, 100.0, 0.05, 0.0, 1.0, OptionType::Call);
  assert!(
    price.is_nan(),
    "degenerate cumulants must give NaN, got {price}"
  );
}

/// The payoff at `t = 0`; NaN past expiry, and before it for a rate that is not finite or a spot
/// or strike that is not finite and positive.
#[test]
fn a_query_off_the_domain_is_nan_and_an_expired_one_its_payoff() {
  let model = BsmFourier { sigma: 0.2 };
  let price = |s, k, r, q, t, ty| CosEngine::default().price(&model, s, k, r, q, t, ty);
  for (s, call, put) in [(90.0, 0.0, 10.0), (100.0, 0.0, 0.0), (110.0, 10.0, 0.0)] {
    assert_eq!(
      price(s, 100.0, 0.05, 0.02, 0.0, OptionType::Call),
      call,
      "s = {s}"
    );
    assert_eq!(
      price(s, 100.0, 0.05, 0.02, 0.0, OptionType::Put),
      put,
      "s = {s}"
    );
  }
  let (inf, nan) = (f64::INFINITY, f64::NAN);
  let off = [
    (100.0, 100.0, 0.05, 0.02, -1.0),
    (100.0, 100.0, 0.05, 0.02, nan),
    (100.0, 100.0, 0.05, 0.02, inf),
    (100.0, 100.0, nan, 0.02, 0.0),
    (100.0, 0.0, 0.05, 0.02, 1.0),
    (100.0, -5.0, 0.05, 0.02, 1.0),
    (100.0, inf, 0.05, 0.02, 1.0),
    (100.0, nan, 0.05, 0.02, 1.0),
    (0.0, 100.0, 0.05, 0.02, 1.0),
    (-5.0, 100.0, 0.05, 0.02, 1.0),
    (inf, 100.0, 0.05, 0.02, 1.0),
    (nan, 100.0, 0.05, 0.02, 1.0),
    (100.0, 100.0, inf, 0.02, 1.0),
    (100.0, 100.0, -inf, 0.02, 1.0),
    (100.0, 100.0, nan, 0.02, 1.0),
    (100.0, 100.0, 0.05, inf, 1.0),
    (100.0, 100.0, 0.05, nan, 1.0),
  ];
  for (s, k, r, q, t) in off {
    for ty in [OptionType::Call, OptionType::Put] {
      let value = price(s, k, r, q, t, ty);
      assert!(
        value.is_nan(),
        "{ty:?} at s {s}, k {k}, r {r}, q {q}, t {t}: {value}"
      );
    }
  }
}

#[test]
#[should_panic(expected = "l must satisfy `l.is_finite() && l > 0.0`, got l = 0.0")]
fn cos_engine_new_rejects_nonpositive_l() {
  CosEngine::new(256, 0.0);
}

#[test]
#[should_panic(expected = "l must satisfy `l.is_finite() && l > 0.0`, got l = inf")]
fn cos_engine_new_rejects_an_infinite_l() {
  CosEngine::new(256, f64::INFINITY);
}

#[test]
#[should_panic(expected = "n must satisfy `n >= 2`, got n = 1")]
fn cos_engine_new_rejects_too_few_terms() {
  CosEngine::new(1, 10.0);
}

#[test]
#[should_panic(expected = "l must satisfy `l.is_finite() && l > 0.0`, got l = NaN")]
fn cos_engine_adaptive_rejects_a_nan_l() {
  CosEngine::adaptive(f64::NAN);
}

/// `adaptive(l)` is the default's term rule at its own `l`, where 256 terms fall short.
#[test]
fn adaptive_adds_terms_at_its_own_width_as_the_default_does() {
  let model = HestonFourier::new(HestonParams {
    v0: 0.02,
    kappa: 0.5,
    theta: 0.09,
    sigma: 1.0,
    rho: -0.9,
  });
  let at = |engine: CosEngine, k| engine.price(&model, 100.0, k, 0.05, 0.01, 2.0, OptionType::Put);
  for k in [80.0, 100.0, 120.0] {
    let want = model.price_put(100.0, k, 0.05, 0.01, 2.0);
    let adaptive = at(CosEngine::adaptive(12.0), k);
    let fixed = at(CosEngine::new(256, 12.0), k);
    assert!(
      (adaptive - want).abs() < 5e-6,
      "K={k}: {adaptive} vs {want}"
    );
    assert!((fixed - want).abs() > 1e-3, "K={k}: {fixed} vs {want}");
    assert_eq!(
      at(CosEngine::adaptive(10.0), k),
      at(CosEngine::default(), k)
    );
  }
}

/// A NaN `chf`, as from a model overflowing its own, prices NaN, not the `0.0` that `f64::max`
/// would pick over it.
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
  let model = std::hint::black_box(model);
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
  assert_strikes_are_prices(&MertonJdFourier {
    sigma: 0.2,
    lambda: 0.5,
    mu_j: -0.1,
    sigma_j: 0.15,
  });
  assert_strikes_are_prices(&NigFourier {
    alpha: 15.0,
    beta: -5.0,
    delta: 0.5,
  });
}

/// Fang & Oosterlee (2008) Table 2: three GBM calls in one computation, held to the values printed
/// to nine decimals and to the closed form (Merton 1973).
#[test]
fn price_strikes_reproduces_the_gbm_strike_vector_of_fang_oosterlee() {
  let strikes = ndarray::array![80.0, 100.0, 120.0];
  let prices = CosEngine::default().price_strikes(
    &BsmFourier { sigma: 0.25 },
    100.0,
    strikes.view(),
    0.1,
    0.0,
    0.1,
    OptionType::Call,
  );
  let printed = [20.799226309, 3.659968453, 0.044577814];
  let bsm = BsmPricer::new(0.25, BsmCoc::Bsm1973);
  for ((&price, &k), want) in prices.iter().zip(&strikes).zip(printed) {
    assert!((price - want).abs() < 1e-9, "K={k}: {price} vs {want}");
    let closed = bsm.price_call(100.0, k, 0.1, 0.0, 0.1);
    assert!((price - closed).abs() < 1e-12, "K={k}: {price} vs {closed}");
  }
}

/// Table 6's strikes 50, 55, …, 150, Heston set (53), `T = 1`: Lewis's integral of the same `chf`
/// from `python3 mp_refs.py` (mpmath, 40 digits, `heston_call_lewis`).
const TABLE_6: [f64; 21] = [
  50.07053913971511,
  45.12410854150659,
  40.20880117230948,
  35.33869482461895,
  30.5332869929249,
  25.819775173024123,
  21.236638756516854,
  16.839368496216263,
  12.70953177475374,
  8.967794318649112,
  5.78515543437619,
  3.3592018895317572,
  1.787135001945827,
  0.9211483314582412,
  0.48282813789152784,
  0.2621235686061102,
  0.1475936526090662,
  0.08587840764229675,
  0.05141485251512249,
  0.03155321757076034,
  0.019788382207635565,
];

/// Reference: Fang & Oosterlee (2008) Tables 4–6: `TABLE_6`, and `T = 10` at `K = 100`.
/// The default engine is within 3e-8 of mpmath and the printed 5.785155450 is 1.6e-8 high; 4,096
/// terms over 16 deviations hold 1e-12, and at `T = 10` Table 5's 22.318945791 stands to 1e-9.
#[test]
fn price_strikes_holds_the_heston_references_of_fang_oosterlee() {
  let model = fang_oosterlee_heston();
  let strikes = ndarray::Array1::linspace(50.0, 150.0, 21);
  for (engine, tolerance) in [
    (CosEngine::default(), 3e-8),
    (CosEngine::new(4096, 16.0), 1e-12),
  ] {
    let prices = engine.price_strikes(
      &model,
      100.0,
      strikes.view(),
      0.0,
      0.0,
      1.0,
      OptionType::Call,
    );
    for ((&price, &k), want) in prices.iter().zip(&strikes).zip(TABLE_6) {
      assert!((price - want).abs() < tolerance, "K={k}: {price} vs {want}");
    }
  }
  let long = CosEngine::default().price(&model, 100.0, 100.0, 0.0, 0.0, 10.0, OptionType::Call);
  assert!((long - 22.31894579115449).abs() < 1e-11, "{long}");
  assert!((long - 22.318945791).abs() < 1e-9, "{long}");
}

/// A strike off the domain is NaN alone and a spot, rate or maturity off it every strike; each
/// value is `price`'s own, the payoff at expiry included.
#[test]
fn price_strikes_keeps_nan_to_the_strikes_it_belongs_to() {
  let model = BsmFourier { sigma: 0.25 };
  let engine = CosEngine::default();
  let (inf, nan) = (f64::INFINITY, f64::NAN);
  let strikes = ndarray::array![90.0, -1.0, 0.0, inf, nan, 110.0];
  let queries = [
    (100.0, 0.05, 0.0, 1.0),
    (100.0, 0.05, 0.0, 0.0),
    (100.0, 0.05, 0.0, -1.0),
    (100.0, 0.05, 0.0, nan),
    (100.0, 0.05, 0.0, inf),
    (0.0, 0.05, 0.0, 1.0),
    (inf, 0.05, 0.0, 1.0),
    (100.0, inf, 0.0, 1.0),
    (100.0, 0.05, nan, 1.0),
  ];
  for (s, r, q, t) in queries {
    for ty in [OptionType::Call, OptionType::Put] {
      let prices = engine.price_strikes(&model, s, strikes.view(), r, q, t, ty);
      for (&got, &k) in prices.iter().zip(&strikes) {
        let want = engine.price(&model, s, k, r, q, t, ty);
        let same = got == want || (got.is_nan() && want.is_nan());
        assert!(
          same,
          "{ty:?} at s {s}, k {k}, r {r}, q {q}, t {t}: {got} vs {want}"
        );
      }
    }
  }
  let live = engine.price_strikes(
    &model,
    100.0,
    strikes.view(),
    0.05,
    0.0,
    1.0,
    OptionType::Put,
  );
  assert!(
    live[0] > 0.0 && live[5] > 0.0 && live.slice(ndarray::s![1..5]).iter().all(|p| p.is_nan())
  );
  let expired = engine.price_strikes(
    &model,
    100.0,
    strikes.view(),
    0.05,
    0.0,
    0.0,
    OptionType::Call,
  );
  assert_eq!(
    expired.slice(ndarray::s![..3]).to_vec(),
    [10.0, 101.0, 100.0]
  );
  for (s, r, q) in [(inf, 0.05, 0.0), (100.0, inf, 0.0), (100.0, 0.05, nan)] {
    let off = engine.price_strikes(&model, s, strikes.view(), r, q, 1.0, OptionType::Put);
    assert!(off.iter().all(|p| p.is_nan()), "s {s}, r {r}, q {q}: {off}");
  }
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
