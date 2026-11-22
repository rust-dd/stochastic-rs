use super::*;

/// Cash-or-nothing call (S=100, K=80, Q=10, r=b=0.06, σ=0.35, T=0.75) prices 7.3444,
/// `Qe^{−rT}N(d₂)`; the carry `b` enters as the query's `q = r − b = 0`.
#[test]
fn cash_or_nothing_call_closed_form() {
  let p = CashOrNothingPricer::new(10.0, 0.35);
  let price = p.price_call(100.0, 80.0, 0.06, 0.0, 0.75);
  assert!((price - 7.3444).abs() < 0.005, "price={price}");
}

/// Cash-or-nothing put on the same scenario: `Qe^{−rT}N(−d₂) ≈ 0.0303`.
#[test]
fn cash_call_put_parity() {
  // CoN call + CoN put = Q * e^{-rT}
  let p = CashOrNothingPricer::new(10.0, 0.35);
  let call = p.price_call(100.0, 80.0, 0.06, 0.0, 0.75);
  let put = p.price_put(100.0, 80.0, 0.06, 0.0, 0.75);
  let total = call + put;
  let expected = 10.0 * (-0.06_f64 * 0.75).exp();
  assert!((total - expected).abs() < 1e-10, "total={total}");
}

/// Asset-or-nothing call + put = forward $S e^{(b-r)T}$
#[test]
fn aon_call_put_parity() {
  let p = AssetOrNothingPricer::new(0.25);
  let call = p.price_call(100.0, 105.0, 0.05, 0.02, 1.0);
  let put = p.price_put(100.0, 105.0, 0.05, 0.02, 1.0);
  let total = call + put;
  let expected = 100.0 * ((0.03_f64 - 0.05_f64) * 1.0).exp();
  assert!((total - expected).abs() < 1e-9, "total={total}");
}

/// Vanilla call = AoN(call, K) - K * CoN(call, K)/Q (with Q=1).
#[test]
fn vanilla_decomposition() {
  let s = 100.0;
  let k = 100.0;
  let r = 0.05;
  let q = 0.0;
  let sigma = 0.2;
  let tau = 1.0;
  let aon = AssetOrNothingPricer::new(sigma);
  let con = CashOrNothingPricer::new(1.0, sigma);
  // BSM vanilla call ≈ 10.4506
  let vanilla = aon.price_call(s, k, r, q, tau) - k * con.price_call(s, k, r, q, tau);
  assert!((vanilla - 10.4506).abs() < 0.005, "decomposition={vanilla}");
}

/// Gap call with $K_1 = K_2$ equals BSM vanilla call.
#[test]
fn gap_reduces_to_vanilla() {
  let p = GapPricer::new(100.0, 0.2);
  let price = p.price_call(100.0, 100.0, 0.05, 0.0, 1.0);
  assert!((price - 10.4506).abs() < 0.005, "gap={price}");
}

/// Haug 2007, p. 178: Gap call with S=50, K1=50, K2=57, r=0.09, b=0.09,
/// sigma=0.20, T=0.5 → -0.0053
#[test]
fn gap_haug_negative_payoff() {
  let p = GapPricer::new(57.0, 0.20);
  let price = p.price_call(50.0, 50.0, 0.09, 0.0, 0.5);
  assert!(price.abs() < 0.05, "gap call={price}");
  // The option gives a negative cash flow when S is between K1 and K2
  assert!(price < 0.0);
}

/// Supershare must be non-negative and below the spot.
#[test]
fn supershare_positive() {
  let p = SuperSharePricer::new(110.0, 0.2);
  let price = p.price_call(100.0, 90.0, 0.05, 0.05, 0.25);
  assert!(price > 0.0, "supershare={price}");
  assert!(price < 100.0, "supershare must be < S");
}

/// Cash-or-nothing delta against a finite difference for the call and the put, which pins the
/// `option_type` wiring of every digital Greek.
#[test]
fn cash_delta_matches_fd() {
  let h = 0.01;
  let p = CashOrNothingPricer::new(10.0, 0.25);
  let (k, r, q, tau) = (100.0, 0.05, 0.03, 0.5);

  for ot in [OptionType::Call, OptionType::Put] {
    let up = p.price_option(100.0 + h, k, r, q, tau, ot);
    let dn = p.price_option(100.0 - h, k, r, q, tau, ot);
    let fd = (up - dn) / (2.0 * h);
    let analytic = p.delta(100.0, k, r, q, tau, ot);
    assert!(
      (fd - analytic).abs() < 1e-4,
      "{ot:?}: fd={fd}, analytic={analytic}"
    );
  }
}

/// The aggregate maps each exposed Greek to its [`Greeks`] member and leaves `NaN` only where the
/// pricer exposes none.
#[test]
fn digital_greeks_aggregates_match_their_accessors() {
  let (s, k, r, q, tau, ot) = (100.0, 100.0, 0.05, 0.03, 0.5, OptionType::Put);

  let con = CashOrNothingPricer::new(10.0, 0.25);
  let g = con.greeks(s, k, r, q, tau, ot);
  assert_eq!(g.delta, con.delta(s, k, r, q, tau, ot));
  assert_eq!(g.gamma, con.gamma(s, k, r, q, tau, ot));
  assert_eq!(g.vega, con.vega(s, k, r, q, tau, ot));
  for (name, value) in Greeks::COMPONENT_NAMES.iter().zip(g.as_array()) {
    let exposed = matches!(*name, "delta" | "gamma" | "vega");
    assert_eq!(
      value.is_nan(),
      !exposed,
      "cash-or-nothing {name} = {value} (exposed={exposed})"
    );
  }

  let aon = AssetOrNothingPricer::new(0.25);
  let g = aon.greeks(s, k, r, q, tau, ot);
  assert_eq!(g.delta, aon.delta(s, k, r, q, tau, ot));
  for (name, value) in Greeks::COMPONENT_NAMES.iter().zip(g.as_array()) {
    assert_eq!(
      value.is_nan(),
      *name != "delta",
      "asset-or-nothing {name} = {value}"
    );
  }
}

/// The gap parity `C − P = Se^{−qT} − K₂e^{−rT}` (any `K₁`), not the vanilla one: the query strike
/// binds to `K₁` and `self.k2` to the payoff strike.
#[test]
fn gap_put_matches_its_own_parity() {
  let p = GapPricer::new(57.0, 0.20);
  let call = p.price_call(50.0, 50.0, 0.09, 0.0, 0.5);
  let put = p.price_put(50.0, 50.0, 0.09, 0.0, 0.5);
  let parity = 50.0 - 57.0 * (-0.09_f64 * 0.5).exp();
  assert!(
    (call - put - parity).abs() < 1e-9,
    "gap parity: {call} - {put}"
  );

  let vanilla_parity = 50.0 - 50.0 * (-0.09_f64 * 0.5).exp();
  assert!(
    (call - put - vanilla_parity).abs() > 1e-3,
    "K1 != K2, so vanilla parity against the query strike must NOT hold"
  );
}

/// Haug (2007) defines only the supershare payoff, so `price_put` is `NaN`, not the parity default.
#[test]
fn supershare_has_no_put_analogue() {
  let p = SuperSharePricer::new(110.0, 0.2);
  assert!(
    p.price_put(100.0, 90.0, 0.05, 0.05, 0.25).is_nan(),
    "supershare has no put analogue"
  );
}

/// One model instance prices a whole strike/maturity grid.
#[test]
fn digital_one_model_prices_a_grid() {
  let con = CashOrNothingPricer::new(10.0, 0.25);
  let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
  let maturities = [0.25, 0.5, 1.0];

  for tau in maturities {
    let prices = strikes.map(|k| con.price_call(100.0, k, 0.05, 0.01, tau));
    for w in prices.windows(2) {
      assert!(
        w[0] > w[1],
        "cash-or-nothing call must fall in K at tau={tau}: {prices:?}"
      );
    }
  }
}

/// Each of the five query arguments, perturbed alone, moves the price: none is ignored or fixed at
/// construction.
#[test]
fn every_query_argument_drives_the_price() {
  let base = (100.0, 100.0, 0.05, 0.01, 0.5);
  let bump = |i: usize| {
    let mut v = [base.0, base.1, base.2, base.3, base.4];
    v[i] *= 1.10;
    (v[0], v[1], v[2], v[3], v[4])
  };
  let price_all = |(s, k, r, q, tau): (f64, f64, f64, f64, f64)| {
    [
      CashOrNothingPricer::new(10.0, 0.25).price_call(s, k, r, q, tau),
      AssetOrNothingPricer::new(0.25).price_call(s, k, r, q, tau),
      GapPricer::new(105.0, 0.25).price_call(s, k, r, q, tau),
      SuperSharePricer::new(120.0, 0.25).price_call(s, k, r, q, tau),
    ]
  };

  let at_base = price_all(base);
  for (i, name) in ["s", "k", "r", "q", "tau"].iter().enumerate() {
    let moved = price_all(bump(i));
    for (j, (a, b)) in at_base.iter().zip(moved.iter()).enumerate() {
      assert!(
        (a - b).abs() > 1e-9,
        "pricer {j}: bumping {name} left the price at {a}"
      );
    }
  }
}

/// Every digital refuses `σ < 0` and `±∞`, as a negative `σ` flips the shared `d₁` into a finite
/// wrong price; `σ = 0`, a NaN `σ` and a negative `cash` payout stay admissible.
mod construction_validation {
  use super::*;

  const SIGMA: &str = "sigma must satisfy `sigma.is_nan() || (sigma.is_finite() && sigma >= 0.0)`";

  fn panic_text(build: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let err = std::panic::catch_unwind(build).expect_err("must refuse");
    err.downcast_ref::<String>().cloned().unwrap_or_default()
  }

  #[test]
  fn every_digital_refuses_a_negative_or_infinite_sigma() {
    for (bad, shown) in [
      (-0.35, "-0.35"),
      (f64::INFINITY, "inf"),
      (f64::NEG_INFINITY, "-inf"),
    ] {
      let want = format!("{SIGMA}, got sigma = {shown}");
      let texts = [
        panic_text(move || {
          let _ = CashOrNothingPricer::new(10.0, bad);
        }),
        panic_text(move || {
          let _ = AssetOrNothingPricer::new(bad);
        }),
        panic_text(move || {
          let _ = GapPricer::new(100.0, bad);
        }),
        panic_text(move || {
          let _ = SuperSharePricer::new(110.0, bad);
        }),
      ];
      assert!(texts.iter().all(|text| *text == want), "{texts:?}");
    }
  }

  #[test]
  #[should_panic(expected = "cash must satisfy `!cash.is_infinite()`, got cash = inf")]
  fn cash_or_nothing_refuses_an_infinite_cash() {
    let _ = CashOrNothingPricer::new(f64::INFINITY, 0.2);
  }

  #[test]
  #[should_panic(expected = "k2 must satisfy `!k2.is_infinite()`, got k2 = -inf")]
  fn gap_refuses_an_infinite_payoff_strike() {
    let _ = GapPricer::new(f64::NEG_INFINITY, 0.2);
  }

  /// `AnalyticBsEngine` builds these from a market handle, and an unlinked handle reads NaN,
  /// which has to reach the caller as a NaN NPV and not as a panic.
  #[test]
  fn a_nan_sigma_cash_or_payoff_strike_prices_nan() {
    let aon = AssetOrNothingPricer::new(f64::NAN);
    assert!(aon.price_call(100.0, 90.0, 0.05, 0.02, 1.0).is_nan());
    let con = CashOrNothingPricer::new(10.0, f64::NAN);
    assert!(con.price_call(100.0, 80.0, 0.06, 0.0, 0.75).is_nan());
    let con = CashOrNothingPricer::new(f64::NAN, 0.2);
    assert!(con.price_call(100.0, 80.0, 0.06, 0.0, 0.75).is_nan());
    let gap = GapPricer::new(f64::NAN, 0.2);
    assert!(gap.price_call(100.0, 80.0, 0.06, 0.0, 0.75).is_nan());
  }

  /// `x_high` has no handle behind it, and a band edge is a price level: finite and positive.
  #[test]
  fn supershare_refuses_an_upper_band_that_is_not_finite_and_positive() {
    for (bad, shown) in [
      (f64::NAN, "NaN"),
      (-110.0, "-110.0"),
      (f64::INFINITY, "inf"),
    ] {
      let text = panic_text(move || {
        let _ = SuperSharePricer::new(bad, 0.2);
      });
      let want = "x_high must satisfy `x_high.is_finite() && x_high > 0.0`, got x_high = ";
      assert_eq!(text, format!("{want}{shown}"));
    }
  }

  #[test]
  fn contract_terms_and_the_zero_volatility_limit_stay_accepted() {
    assert_eq!(CashOrNothingPricer::new(-10.0, 0.35).cash, -10.0);
    assert_eq!(GapPricer::new(-100.0, 0.2).k2, -100.0);
    assert_eq!(AssetOrNothingPricer::new(0.0).sigma, 0.0);
    assert_eq!(SuperSharePricer::new(80.0, 0.2).x_high, 80.0);
  }
}
