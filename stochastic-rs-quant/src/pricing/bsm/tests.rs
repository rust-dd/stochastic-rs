use super::*;
use crate::OptionType;
use crate::traits::ModelPricer;

/// Query point every golden below is pinned at: `(s, k, r, q, tau)`.
const Q: (f64, f64, f64, f64, f64) = (100.0, 105.0, 0.05, 0.02, 0.75);
/// Volatility every golden below is pinned at.
const V: f64 = 0.25;

#[test]
fn bsm_price() {
  let (s, k, r, q, tau) = Q;
  let model = BSMPricer::new(0.2, BSMCoc::Bsm1973);
  let (call, put) = model.call_put(s, k, r, q, tau);
  assert!(call > 0.0 && put > 0.0);
  assert_eq!(call, model.price_call(s, k, r, q, tau));
  assert_eq!(put, model.price_put(s, k, r, q, tau));
}

/// Every `BSMCoc` variant priced through [`ModelPricer`] must reproduce the
/// price the pre-query `BSMPricer` produced at the same market data.
///
/// Values captured from the pre-refactor `calculate_call_put()` before any
/// change was made — the same technique
/// `merton_price_m10_matches_pre_refactor_value` uses — not recomputed from
/// the new code. `(r, q)` stands in for `(r_d, r_f)` under
/// `GarmanKohlhagen1983`, which is what makes that variant's golden equal
/// `Merton1973`'s; the pre-refactor pricer was fed `r_d = r`, `r_f = q`.
#[test]
fn bsm_model_pricer_matches_pre_refactor_goldens() {
  let (s, k, r, q, tau) = Q;
  let cases = [
    (
      BSMCoc::Bsm1973,
      8.113_492_237_591_075,
      9.248_906_098_277_367,
    ),
    (
      BSMCoc::Merton1973,
      7.356_277_072_859_285,
      9.980_496_973_239_312,
    ),
    (
      BSMCoc::Black1976,
      6.317_116_399_721_527,
      11.133_088_488_325_626,
    ),
    (
      BSMCoc::Asay1982,
      6.317_116_399_721_527,
      11.133_088_488_325_626,
    ),
    (
      BSMCoc::GarmanKohlhagen1983,
      7.356_277_072_859_285,
      9.980_496_973_239_312,
    ),
  ];

  for (coc, want_call, want_put) in cases {
    let model = BSMPricer::new(V, coc);
    let call = model.price_call(s, k, r, q, tau);
    let put = model.price_put(s, k, r, q, tau);
    assert_eq!(call, want_call, "{coc:?} call");
    assert_eq!(put, want_put, "{coc:?} put");
    assert_eq!(
      model.price_option(s, k, r, q, tau, OptionType::Call),
      want_call
    );
    assert_eq!(
      model.price_option(s, k, r, q, tau, OptionType::Put),
      want_put
    );
  }
}

/// `Asay1982` and `Black1976` are the same model in this crate — both carry
/// at `b = 0` and discount at `exp(-r * tau)`. Asay (1982)'s margined
/// futures convention (no discounting at all) is *not* implemented; this
/// pins the equality so a future Asay implementation is a deliberate,
/// visible change rather than a silent one.
#[test]
fn bsm_asay_currently_equals_black76() {
  let (s, k, r, q, tau) = Q;
  let black = BSMPricer::new(V, BSMCoc::Black1976);
  let asay = BSMPricer::new(V, BSMCoc::Asay1982);
  assert_eq!(
    black.call_put(s, k, r, q, tau),
    asay.call_put(s, k, r, q, tau)
  );
}

/// [`ModelPricer::price_put`]'s default is vanilla parity
/// (`C - S e^{-q tau} + K e^{-r tau}`), which assumes `b = r - q`. It is
/// therefore correct for `Merton1973` / `GarmanKohlhagen1983` and wrong for
/// the other three — the reason `BSMPricer` overrides it. This test is the
/// evidence for that override.
#[test]
fn bsm_price_put_overrides_vanilla_parity() {
  let (s, k, r, q, tau) = Q;
  let vanilla = |call: f64| call - s * (-q * tau).exp() + k * (-r * tau).exp();

  for coc in [BSMCoc::Merton1973, BSMCoc::GarmanKohlhagen1983] {
    let model = BSMPricer::new(V, coc);
    let (call, put) = model.call_put(s, k, r, q, tau);
    assert!(
      (put - vanilla(call)).abs() < 1e-12,
      "{coc:?}: b = r - q, so vanilla parity must hold"
    );
  }

  for coc in [BSMCoc::Bsm1973, BSMCoc::Black1976, BSMCoc::Asay1982] {
    let model = BSMPricer::new(V, coc);
    let (call, put) = model.call_put(s, k, r, q, tau);
    assert!(
      (put - vanilla(call)).abs() > 1e-3,
      "{coc:?}: b != r - q, so the trait default would be wrong (put={put}, default={})",
      vanilla(call)
    );
    // The parity that does hold carries at `exp((b - r) * tau)`.
    let carry = ((model.b(r, q) - r) * tau).exp();
    assert!(
      (call - put - (s * carry - k * (-r * tau).exp())).abs() < 1e-12,
      "{coc:?}: generalised parity must hold"
    );
  }
}

#[test]
fn bsm_implied_volatility() {
  let (s, k, r, q, tau) = Q;
  let model = BSMPricer::new(0.2, BSMCoc::Bsm1973);
  let call = model.price_call(s, k, r, q, tau);
  let iv = model.implied_volatility(call, s, k, r, q, tau, OptionType::Call);
  assert!(
    (iv - 0.2).abs() < 1e-6,
    "IV round-trip failed: input sigma=0.2, recovered iv={iv}"
  );
}

#[test]
fn bsm_iv_round_trip_across_strikes_and_maturities() {
  for &tau in &[0.25_f64, 1.0, 2.0] {
    for &k in &[90.0_f64, 100.0, 110.0] {
      for &sigma in &[0.1_f64, 0.2, 0.4] {
        let model = BSMPricer::new(sigma, BSMCoc::Bsm1973);
        let call = model.price_call(100.0, k, 0.03, 0.0, tau);
        let iv = model.implied_volatility(call, 100.0, k, 0.03, 0.0, tau, OptionType::Call);
        assert!(
          (iv - sigma).abs() < 1e-4,
          "IV round-trip mismatch: tau={tau}, k={k}, sigma_in={sigma}, sigma_out={iv}"
        );
      }
    }
  }
}

/// The date pair moved off the pricer: the caller resolves `(eval, expiry)`
/// to `tau` through [`DayCountConvention`](crate::calendar::DayCountConvention)
/// and passes it. Same numbers as when `BSMPricer` did it internally
/// through `TimeExt`.
#[test]
fn bsm_dates_match_tau_pricing() {
  use chrono::NaiveDate;

  use crate::calendar::DayCountConvention;

  let eval = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();
  let expiration = NaiveDate::from_ymd_opt(2027, 1, 2).unwrap();
  let tau = DayCountConvention::Actual365Fixed.year_fraction::<f64>(eval, expiration);

  let model = BSMPricer::new(0.2, BSMCoc::Bsm1973);
  let (c_dates, p_dates) = model.call_put(100.0, 100.0, 0.05, 0.0, tau);
  let (c_tau, p_tau) = model.call_put(100.0, 100.0, 0.05, 0.0, 1.0);
  assert!(
    (c_dates - c_tau).abs() < 1e-2 && (p_dates - p_tau).abs() < 1e-2,
    "365/365 date span must price within a day of tau=1: dates=({c_dates},{p_dates}), tau=({c_tau},{p_tau})"
  );
  let iv = model.implied_volatility(c_dates, 100.0, 100.0, 0.05, 0.0, tau, OptionType::Call);
  assert!((iv - 0.2).abs() < 1e-6, "IV from date-derived tau: {iv}");
}

/// The full Greek set stays finite at the money and keeps the signs the
/// pre-query methods produced.
#[test]
fn bsm_greeks_second_order_are_finite() {
  let (s, k, r, tau) = (100.0, 100.0, 0.05, 1.0);
  let model = BSMPricer::new(0.2, BSMCoc::Bsm1973);
  let vanna = model.vanna(s, k, r, 0.0, tau);
  let charm = model.charm(s, k, r, 0.0, tau, OptionType::Call);
  let volga = model.vomma(s, k, r, 0.0, tau);
  let veta = model.dvega_dtime(s, k, r, 0.0, tau);
  assert!(
    vanna.is_finite() && charm.is_finite() && volga.is_finite() && veta.is_finite(),
    "second-order Greeks should be finite at-the-money"
  );

  let delta = model.delta(s, k, r, 0.0, tau, OptionType::Call);
  assert!((0.0..=1.0).contains(&delta), "call delta in [0,1]: {delta}");
  assert!(model.gamma(s, k, r, 0.0, tau) > 0.0);
  assert!(model.vega(s, k, r, 0.0, tau) > 0.0);
}

/// Greeks pinned to the values the pre-query methods produced at [`Q`].
#[test]
fn bsm_greeks_match_pre_refactor_goldens() {
  let (s, k, r, q, tau) = Q;
  let m = BSMPricer::new(V, BSMCoc::Merton1973);
  let call = OptionType::Call;
  let put = OptionType::Put;
  let cases = [
    (
      "delta",
      m.delta(s, k, r, q, tau, call),
      0.487_377_928_659_916_86,
    ),
    ("gamma", m.gamma(s, k, r, q, tau), 0.018_150_446_393_251_914),
    ("vega", m.vega(s, k, r, q, tau), 34.032_086_987_347_334),
    (
      "theta",
      m.theta(s, k, r, q, tau, call),
      -6.766_334_430_228_008,
    ),
    ("rho", m.rho(s, k, r, q, tau, call), 31.036_136_844_849_302),
    ("vanna", m.vanna(s, k, r, q, tau), 0.361_031_721_107_564_15),
    (
      "charm",
      m.charm(s, k, r, q, tau, call),
      0.104_875_734_124_484_78,
    ),
    ("vomma", m.vomma(s, k, r, q, tau), 0.411_960_901_932_774_2),
    (
      "dvega_dtime",
      m.dvega_dtime(s, k, r, q, tau),
      -22.138_208_955_842_344,
    ),
    (
      "put delta",
      m.delta(s, k, r, q, tau, put),
      -0.497_734_010_943_145_8,
    ),
    (
      "put theta",
      m.theta(s, k, r, q, tau, put),
      -9.655_177_423_155_209,
    ),
    (
      "put rho",
      m.rho(s, k, r, q, tau, put),
      -44.815_423_550_665_42,
    ),
    (
      "put charm",
      m.charm(s, k, r, q, tau, put),
      0.124_577_972_916_546_03,
    ),
    (
      "put phi",
      m.phi(s, k, r, q, tau, put),
      37.330_050_820_735_934,
    ),
    (
      "put zeta",
      m.zeta(s, k, r, q, tau, put),
      -0.590_830_608_058_466,
    ),
    (
      "put strike_delta",
      m.strike_delta(s, k, r, q, tau, put),
      0.569_084_743_500_513_2,
    ),
  ];
  for (name, got, want) in cases {
    assert_eq!(got, want, "{name}");
  }
}

#[test]
fn bsm_iv_round_trip_with_dividend_yield() {
  let model = BSMPricer::new(0.25, BSMCoc::Merton1973);
  let call = model.price_call(100.0, 105.0, 0.04, 0.02, 1.0);
  let iv = model.implied_volatility(call, 100.0, 105.0, 0.04, 0.02, 1.0, OptionType::Call);
  assert!(
    (iv - 0.25).abs() < 1e-6,
    "Merton1973 IV round-trip failed: input sigma=0.25, recovered iv={iv}"
  );
}

/// Two day-count conventions over the same leap-year span give different
/// `tau`, and therefore different prices — the divergence the pricer used
/// to expose through its own `dcc` field, now the caller's to make.
#[test]
fn bsm_dcc_pricing_diverges_from_default() {
  use chrono::NaiveDate;

  use crate::calendar::DayCountConvention;

  // Spans a leap-year (2024 inclusive). Act/365F → 366/365, Act/360 → 366/360.
  let eval = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
  let exp = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
  let tau_default = DayCountConvention::Actual365Fixed.year_fraction::<f64>(eval, exp);
  let tau_act360 = DayCountConvention::Actual360.year_fraction::<f64>(eval, exp);
  assert!(
    (tau_default - 366.0 / 365.0).abs() < 1e-12,
    "Act/365F leap-year fraction"
  );
  assert!(
    (tau_act360 - 366.0 / 360.0).abs() < 1e-12,
    "Act/360 leap-year fraction"
  );

  let model = BSMPricer::new(0.2, BSMCoc::Bsm1973);
  let p_default = model.price_call(100.0, 100.0, 0.05, 0.0, tau_default);
  let p_act360 = model.price_call(100.0, 100.0, 0.05, 0.0, tau_act360);
  assert!(
    (p_default - p_act360).abs() > 1e-3,
    "Act/365F vs Act/360 should produce visibly different ATM prices on a leap-year span (got {p_default} vs {p_act360})"
  );
}

/// One model instance prices a whole strike/maturity grid — the capability
/// [`ModelPricer`] exists for and the pre-query struct could not offer
/// without rebuilding itself per point.
#[test]
fn bsm_one_model_prices_a_grid() {
  let model = BSMPricer::new(0.2, BSMCoc::Merton1973);
  let strikes = [90.0, 100.0, 110.0];
  let maturities = [0.25, 1.0, 2.0];
  for &tau in &maturities {
    let mut prev = f64::INFINITY;
    for &k in &strikes {
      let call = model.price_call(100.0, k, 0.05, 0.01, tau);
      assert!(call.is_finite() && call > 0.0);
      assert!(call < prev, "call must decrease in strike at tau={tau}");
      prev = call;
    }
  }
}
