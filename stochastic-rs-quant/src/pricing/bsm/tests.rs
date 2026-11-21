use super::*;
use crate::OptionType;
use crate::traits::ModelPricer;

/// Query point every golden below is pinned at: `(s, k, r, q, tau)`.
const Q: (f64, f64, f64, f64, f64) = (100.0, 105.0, 0.05, 0.02, 0.75);
/// Volatility every golden below is pinned at.
const V: f64 = 0.25;
/// Absolute tolerance on goldens captured on aarch64-darwin: CI runs x86_64-linux, where FMA and
/// libm differences inside `norm_cdf` rule out bit-exactness.
const TOL: f64 = 1e-12;

#[test]
fn bsm_price() {
  let (s, k, r, q, tau) = Q;
  let model = BsmPricer::new(0.2, BsmCoc::Bsm1973);
  let (call, put) = (
    model.price_call(s, k, r, q, tau),
    model.price_put(s, k, r, q, tau),
  );
  assert!(call > 0.0 && put > 0.0);
  assert_eq!(call, model.price_call(s, k, r, q, tau));
  assert_eq!(put, model.price_put(s, k, r, q, tau));
}

/// Every `BsmCoc` variant reproduces its golden through [`ModelPricer`]; `(r, q)` stands in for
/// `(r_d, r_f)` under `GarmanKohlhagen1983`, which is why its golden equals `Merton1973`'s.
#[test]
fn bsm_model_pricer_matches_pre_refactor_goldens() {
  let (s, k, r, q, tau) = Q;
  let cases = [
    (
      BsmCoc::Bsm1973,
      8.113_484_530_210_712,
      9.248_898_390_896_997,
    ),
    (
      BsmCoc::Merton1973,
      7.356_284_457_997_674,
      9.980_504_358_377_694,
    ),
    (
      BsmCoc::Black1976,
      6.317_127_602_016_328,
      11.133_099_690_620_426,
    ),
    (
      BsmCoc::Asay1982,
      6.317_127_602_016_328,
      11.133_099_690_620_426,
    ),
    (
      BsmCoc::GarmanKohlhagen1983,
      7.356_284_457_997_674,
      9.980_504_358_377_694,
    ),
  ];

  for (coc, want_call, want_put) in cases {
    let model = BsmPricer::new(V, coc);
    let call = model.price_call(s, k, r, q, tau);
    let put = model.price_put(s, k, r, q, tau);
    assert!((call - want_call).abs() < TOL, "{coc:?} call: {call}");
    assert!((put - want_put).abs() < TOL, "{coc:?} put: {put}");
    assert_eq!(
      model.price_option(s, k, r, q, tau, OptionType::Call),
      call,
      "{coc:?}: price_option must dispatch to price_call"
    );
    assert_eq!(
      model.price_option(s, k, r, q, tau, OptionType::Put),
      put,
      "{coc:?}: price_option must dispatch to price_put"
    );
  }
}

/// `Asay1982` is `Black1976` here (`b = 0`, discounted at `e^{−rτ}`): Asay's (1982) undiscounted
/// margined-futures convention is not implemented, and this pins it.
#[test]
fn bsm_asay_currently_equals_black76() {
  let (s, k, r, q, tau) = Q;
  let black = BsmPricer::new(V, BsmCoc::Black1976);
  let asay = BsmPricer::new(V, BsmCoc::Asay1982);
  for option_type in [OptionType::Call, OptionType::Put] {
    assert_eq!(
      black.price_option(s, k, r, q, tau, option_type),
      asay.price_option(s, k, r, q, tau, option_type)
    );
  }
}

/// The parity default assumes `b = r − q`, right for `Merton1973` and `GarmanKohlhagen1983`, wrong
/// for the other three, so `BsmPricer` overrides `price_put`.
#[test]
fn bsm_price_put_overrides_vanilla_parity() {
  let (s, k, r, q, tau) = Q;
  let vanilla = |call: f64| call - s * (-q * tau).exp() + k * (-r * tau).exp();

  for coc in [BsmCoc::Merton1973, BsmCoc::GarmanKohlhagen1983] {
    let model = BsmPricer::new(V, coc);
    let (call, put) = (
      model.price_call(s, k, r, q, tau),
      model.price_put(s, k, r, q, tau),
    );
    assert!(
      (put - vanilla(call)).abs() < 1e-12,
      "{coc:?}: b = r - q, so vanilla parity must hold"
    );
  }

  for coc in [BsmCoc::Bsm1973, BsmCoc::Black1976, BsmCoc::Asay1982] {
    let model = BsmPricer::new(V, coc);
    let (call, put) = (
      model.price_call(s, k, r, q, tau),
      model.price_put(s, k, r, q, tau),
    );
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
  let model = BsmPricer::new(0.2, BsmCoc::Bsm1973);
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
        let model = BsmPricer::new(sigma, BsmCoc::Bsm1973);
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

/// The caller resolves `(eval, expiry)` to `tau` through a `DayCountConvention` and passes it.
#[test]
fn bsm_dates_match_tau_pricing() {
  use chrono::NaiveDate;

  use crate::calendar::DayCountConvention;

  let eval = NaiveDate::from_ymd_opt(2026, 1, 2).unwrap();
  let expiration = NaiveDate::from_ymd_opt(2027, 1, 2).unwrap();
  let tau = DayCountConvention::Actual365Fixed.year_fraction::<f64>(eval, expiration);

  let model = BsmPricer::new(0.2, BsmCoc::Bsm1973);
  let (c_dates, p_dates) = (
    model.price_call(100.0, 100.0, 0.05, 0.0, tau),
    model.price_put(100.0, 100.0, 0.05, 0.0, tau),
  );
  let (c_tau, p_tau) = (
    model.price_call(100.0, 100.0, 0.05, 0.0, 1.0),
    model.price_put(100.0, 100.0, 0.05, 0.0, 1.0),
  );
  // 2026-01-02 → 2027-01-02 is 365 days and 2026 is not a leap year, so
  // Act/365F gives tau == 1.0 exactly and the two prices are identical.
  assert!(
    (c_dates - c_tau).abs() < 1e-12 && (p_dates - p_tau).abs() < 1e-12,
    "date-derived tau diverged from tau=1: dates=({c_dates},{p_dates}), tau=({c_tau},{p_tau})"
  );
  let iv = model.implied_volatility(c_dates, 100.0, 100.0, 0.05, 0.0, tau, OptionType::Call);
  assert!((iv - 0.2).abs() < 1e-6, "IV from date-derived tau: {iv}");
}

/// The full Greek set stays finite at the money, each Greek with its pinned sign.
#[test]
fn bsm_greeks_second_order_are_finite() {
  let (s, k, r, tau) = (100.0, 100.0, 0.05, 1.0);
  let model = BsmPricer::new(0.2, BsmCoc::Bsm1973);
  let vanna = model.vanna(s, k, r, 0.0, tau);
  let charm = model.charm(s, k, r, 0.0, tau, OptionType::Call);
  let volga = model.volga(s, k, r, 0.0, tau);
  let veta = model.veta(s, k, r, 0.0, tau);
  assert!(
    vanna.is_finite() && charm.is_finite() && volga.is_finite() && veta.is_finite(),
    "second-order Greeks should be finite at-the-money"
  );

  let delta = model.delta(s, k, r, 0.0, tau, OptionType::Call);
  assert!((0.0..=1.0).contains(&delta), "call delta in [0,1]: {delta}");
  assert!(model.gamma(s, k, r, 0.0, tau) > 0.0);
  assert!(model.vega(s, k, r, 0.0, tau) > 0.0);
}

/// Greeks pinned at [`Q`].
#[test]
fn bsm_greeks_match_pre_refactor_goldens() {
  let (s, k, r, q, tau) = Q;
  let m = BsmPricer::new(V, BsmCoc::Merton1973);
  let call = OptionType::Call;
  let put = OptionType::Put;
  let cases = [
    (
      "delta",
      m.delta(s, k, r, q, tau, call),
      0.487_377_957_358_522_85,
    ),
    ("gamma", m.gamma(s, k, r, q, tau), 0.018_150_446_393_251_914),
    ("vega", m.vega(s, k, r, q, tau), 34.032_086_987_347_334),
    (
      "theta",
      m.theta(s, k, r, q, tau, call),
      -6.766_334_147_066_908,
    ),
    ("rho", m.rho(s, k, r, q, tau, call), 31.036_133_458_390_957),
    ("vanna", m.vanna(s, k, r, q, tau), 0.361_031_721_107_564_15),
    (
      "charm",
      m.charm(s, k, r, q, tau, call),
      0.104_875_733_550_512_66,
    ),
    ("volga", m.volga(s, k, r, q, tau), 0.411_960_901_932_774_2),
    ("veta", m.veta(s, k, r, q, tau), -22.138_208_955_842_344),
    (
      "put delta",
      m.delta(s, k, r, q, tau, put),
      -0.497_733_982_244_539_8,
    ),
    (
      "put theta",
      m.theta(s, k, r, q, tau, put),
      -9.655_177_591_521_886,
    ),
    (
      "put rho",
      m.rho(s, k, r, q, tau, put),
      -44.815_426_937_123_76,
    ),
    (
      "put charm",
      m.charm(s, k, r, q, tau, put),
      0.124_577_972_342_573_91,
    ),
    (
      "put phi",
      m.phi(s, k, r, q, tau, put),
      37.330_048_668_340_48,
    ),
    (
      "put strike_delta",
      m.strike_delta(s, k, r, q, tau, put),
      0.569_084_786_503_158_9,
    ),
  ];
  for (name, got, want) in cases {
    assert!((got - want).abs() < TOL, "{name}: got {got}, want {want}");
  }
}

#[test]
fn a_put_zeta_is_its_in_the_money_probability() {
  let (s, k, r, q, tau) = Q;
  let m = BsmPricer::new(V, BsmCoc::Merton1973);
  let put = m.zeta(s, k, r, q, tau, OptionType::Put);
  assert!((put - 0.590_830_652_704_328_6).abs() < TOL, "{put}");
  let call = m.zeta(s, k, r, q, tau, OptionType::Call);
  assert!((call + put - 1.0).abs() < TOL, "{call} + {put}");
}

#[test]
fn a_deep_out_of_the_money_put_delta_keeps_its_digits() {
  // mpmath's 4e-14 delta, which Φ(d₁) − 1 would cancel; d₁ = 7.47 rounds it by 2εd₁² ≈ 1.2e-14.
  let m = BsmPricer::new(0.1, BsmCoc::Merton1973);
  let delta = m.delta(100.0, 60.0, 0.05, 0.02, 0.5, OptionType::Put);
  let exact = -3.920_292_531_945_109_5e-14;
  assert!((delta - exact).abs() <= 4e-14 * exact.abs(), "{delta}");
}

#[test]
fn bsm_iv_round_trip_with_dividend_yield() {
  let model = BsmPricer::new(0.25, BsmCoc::Merton1973);
  let call = model.price_call(100.0, 105.0, 0.04, 0.02, 1.0);
  let iv = model.implied_volatility(call, 100.0, 105.0, 0.04, 0.02, 1.0, OptionType::Call);
  assert!(
    (iv - 0.25).abs() < 1e-6,
    "Merton1973 IV round-trip failed: input sigma=0.25, recovered iv={iv}"
  );
}

/// Two day-count conventions over one leap-year span give different `tau`, so different prices.
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

  let model = BsmPricer::new(0.2, BsmCoc::Bsm1973);
  let p_default = model.price_call(100.0, 100.0, 0.05, 0.0, tau_default);
  let p_act360 = model.price_call(100.0, 100.0, 0.05, 0.0, tau_act360);
  assert!(
    (p_default - p_act360).abs() > 1e-3,
    "Act/365F vs Act/360 should produce visibly different ATM prices on a leap-year span (got {p_default} vs {p_act360})"
  );
}

/// One model instance prices a whole strike/maturity grid.
#[test]
fn bsm_one_model_prices_a_grid() {
  let model = BsmPricer::new(0.2, BsmCoc::Merton1973);
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

/// Every member of the aggregate against its accessor: a swapped `volga` and `veta` would look
/// plausible.
#[test]
fn bsm_greeks_aggregate_matches_accessors() {
  let (s, k, r, q, tau) = Q;
  let m = BsmPricer::new(V, BsmCoc::Merton1973);
  for ot in [OptionType::Call, OptionType::Put] {
    let g = m.greeks(s, k, r, q, tau, ot);
    assert_eq!(g.delta, m.delta(s, k, r, q, tau, ot), "delta");
    assert_eq!(g.gamma, m.gamma(s, k, r, q, tau), "gamma");
    assert_eq!(g.vega, m.vega(s, k, r, q, tau), "vega");
    assert_eq!(g.theta, m.theta(s, k, r, q, tau, ot), "theta");
    assert_eq!(g.rho, m.rho(s, k, r, q, tau, ot), "rho");
    assert_eq!(g.vanna, m.vanna(s, k, r, q, tau), "vanna");
    assert_eq!(g.charm, m.charm(s, k, r, q, tau, ot), "charm");
    assert_eq!(g.volga, m.volga(s, k, r, q, tau), "volga");
    assert_eq!(g.veta, m.veta(s, k, r, q, tau), "veta");
    // volga ≠ veta here, so a swapped pair cannot pass the equalities above by accident.
    assert_ne!(
      g.volga, g.veta,
      "volga and veta must not be interchangeable"
    );
  }
}

/// The Python `BSMPricer` and the BSM calibrator price at Merton's carry `r − q`; at `q = 0` that
/// is `r − 0.0 == r`, so every method reads the bits `Bsm1973` reads.
#[test]
fn merton_at_a_zero_yield_is_bsm_1973_bit_for_bit() {
  let methods = |m: &BsmPricer, (s, k, r, tau): (f64, f64, f64, f64), ot: OptionType| {
    let price = m.price_option(s, k, r, 0.0, tau, ot);
    let (call, put) = (
      m.price_call(s, k, r, 0.0, tau),
      m.price_put(s, k, r, 0.0, tau),
    );
    [
      price,
      call,
      put,
      m.delta(s, k, r, 0.0, tau, ot),
      m.gamma(s, k, r, 0.0, tau),
      m.vega(s, k, r, 0.0, tau),
      m.theta(s, k, r, 0.0, tau, ot),
      m.rho(s, k, r, 0.0, tau, ot),
      m.vanna(s, k, r, 0.0, tau),
      m.charm(s, k, r, 0.0, tau, ot),
      m.volga(s, k, r, 0.0, tau),
      m.implied_volatility(price, s, k, r, 0.0, tau, ot),
    ]
    .map(f64::to_bits)
  };
  for v in [0.05, 0.25, 0.8] {
    let (bsm, merton) = (
      BsmPricer::new(v, BsmCoc::Bsm1973),
      BsmPricer::new(v, BsmCoc::Merton1973),
    );
    for query in [
      (100.0, 100.0, 0.03, 1.0),
      (80.0, 140.0, -0.01, 0.02),
      (125.0, 70.0, 0.0, 5.0),
    ] {
      for ot in [OptionType::Call, OptionType::Put] {
        assert_eq!(
          methods(&bsm, query, ot),
          methods(&merton, query, ot),
          "{v} {query:?}"
        );
      }
    }
  }
}

/// Ultima against a central difference of `volga` in sigma, `h = 1e-5 sigma`, over moneyness,
/// maturity and carry.
#[test]
fn ultima_matches_finite_difference_of_volga() {
  let cells = [
    (100.0, 105.0, 0.05, 0.02, 0.75, 0.25),
    (100.0, 100.0, 0.03, 0.0, 1.0, 0.2),
    (100.0, 150.0, 0.02, 0.01, 0.25, 0.3),
    (100.0, 70.0, 0.01, 0.03, 2.0, 0.35),
    (100.0, 95.0, 0.04, 0.0, 0.05, 0.4),
    (80.0, 100.0, 0.0, 0.05, 5.0, 0.15),
  ];
  for (s, k, r, q, tau, v) in cells {
    let h = 1e-5 * v;
    let volga = |sigma: f64| BsmPricer::new(sigma, BsmCoc::Merton1973).volga(s, k, r, q, tau);
    let fd = (volga(v + h) - volga(v - h)) / (2.0 * h);
    let ultima = BsmPricer::new(v, BsmCoc::Merton1973).ultima(s, k, r, q, tau);
    assert!(
      ((ultima - fd) / fd).abs() < 1e-6,
      "{s} {k} {r} {q} {tau} {v}: {ultima} vs {fd}"
    );
  }
}

/// Ultima at the pinned query against `d^3 V/d sigma^3` of the Merton (1973) call from mpmath's
/// numerical third derivative at 60 digits, quoted to 30.
#[test]
fn ultima_matches_mpmath_third_derivative() {
  let (s, k, r, q, tau) = Q;
  let ultima = BsmPricer::new(V, BsmCoc::Merton1973).ultima(s, k, r, q, tau);
  let reference = "-30.4626092468568644460922716058".parse::<f64>().unwrap();
  assert!(((ultima - reference) / reference).abs() < 1e-13, "{ultima}");
}

/// At `d2 = 0` the Volga form of ultima is `0 * inf`; the Vega form stays finite.
#[test]
fn ultima_is_finite_where_d2_vanishes() {
  let (s, r, q, tau, v) = (100.0_f64, 0.05, 0.02, 0.75, 0.25_f64);
  let k = s * ((r - q - 0.5 * v * v) * tau).exp();
  let model = BsmPricer::new(v, BsmCoc::Merton1973);
  let (_, d2) = model.d1_d2(s, k, r, q, tau);
  assert!(d2.abs() < 1e-14, "{d2}");
  assert!(model.ultima(s, k, r, q, tau).is_finite());
}
