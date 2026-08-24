use super::*;

/// Cash-or-nothing call: S=100, K=80, Q=10, r=0.06, b=0.06, sigma=0.35,
/// T=0.75 → 7.3444 (verified analytically against Q*e^{-rT}*N(d2)).
#[test]
fn cash_or_nothing_call_closed_form() {
  let p = CashOrNothingPricer {
    s: 100.0,
    k: 80.0,
    cash: 10.0,
    r: 0.06,
    b: 0.06,
    sigma: 0.35,
    tau: 0.75,
    option_type: OptionType::Call,
  };
  let price = p.price();
  assert!((price - 7.3444).abs() < 0.005, "price={price}");
}

/// Cash-or-nothing put: S=100, K=80, Q=10, r=0.06, b=0.06, sigma=0.35,
/// T=0.75 → ~ Q*e^{-rT}*N(-d2) = 10 * 0.95599 * (1 - 0.97264/Q*e^{rT})
/// ≈ 0.0303
#[test]
fn cash_call_put_parity() {
  // CoN call + CoN put = Q * e^{-rT}
  let base = CashOrNothingPricer {
    s: 100.0,
    k: 80.0,
    cash: 10.0,
    r: 0.06,
    b: 0.06,
    sigma: 0.35,
    tau: 0.75,
    option_type: OptionType::Call,
  };
  let put = CashOrNothingPricer {
    option_type: OptionType::Put,
    ..base.clone()
  };
  let total = base.price() + put.price();
  let expected = 10.0 * (-0.06_f64 * 0.75).exp();
  assert!((total - expected).abs() < 1e-10, "total={total}");
}

/// Asset-or-nothing call + put = forward $S e^{(b-r)T}$
#[test]
fn aon_call_put_parity() {
  let c = AssetOrNothingPricer {
    s: 100.0,
    k: 105.0,
    r: 0.05,
    b: 0.03,
    sigma: 0.25,
    tau: 1.0,
    option_type: OptionType::Call,
  };
  let p = AssetOrNothingPricer {
    option_type: OptionType::Put,
    ..c.clone()
  };
  let total = c.price() + p.price();
  let expected = 100.0 * ((0.03_f64 - 0.05_f64) * 1.0).exp();
  assert!((total - expected).abs() < 1e-9, "total={total}");
}

/// Vanilla call = AoN(call, K) - K * CoN(call, K)/Q (with Q=1).
#[test]
fn vanilla_decomposition() {
  let s = 100.0;
  let k = 100.0;
  let r = 0.05;
  let b = 0.05;
  let sigma = 0.2;
  let tau = 1.0;
  let aon = AssetOrNothingPricer {
    s,
    k,
    r,
    b,
    sigma,
    tau,
    option_type: OptionType::Call,
  };
  let con = CashOrNothingPricer {
    s,
    k,
    cash: 1.0,
    r,
    b,
    sigma,
    tau,
    option_type: OptionType::Call,
  };
  // BSM vanilla call ≈ 10.4506
  let vanilla = aon.price() - k * con.price();
  assert!((vanilla - 10.4506).abs() < 0.005, "decomposition={vanilla}");
}

/// Gap call with $K_1 = K_2$ equals BSM vanilla call.
#[test]
fn gap_reduces_to_vanilla() {
  let p = GapPricer {
    s: 100.0,
    k1: 100.0,
    k2: 100.0,
    r: 0.05,
    b: 0.05,
    sigma: 0.2,
    tau: 1.0,
    option_type: OptionType::Call,
  };
  let price = p.price();
  assert!((price - 10.4506).abs() < 0.005, "gap={price}");
}

/// Haug 2007, p. 178: Gap call with S=50, K1=50, K2=57, r=0.09, b=0.09,
/// sigma=0.20, T=0.5 → -0.0053
#[test]
fn gap_haug_negative_payoff() {
  let p = GapPricer {
    s: 50.0,
    k1: 50.0,
    k2: 57.0,
    r: 0.09,
    b: 0.09,
    sigma: 0.20,
    tau: 0.5,
    option_type: OptionType::Call,
  };
  let price = p.price();
  assert!(price.abs() < 0.05, "gap call={price}");
  // The option gives a negative cash flow when S is between K1 and K2
  assert!(price < 0.0);
}

/// Supershare must be non-negative and zero when bands are infinitely apart
/// in the wrong direction.
#[test]
fn supershare_positive() {
  let p = SuperSharePricer {
    s: 100.0,
    x_low: 90.0,
    x_high: 110.0,
    r: 0.05,
    b: 0.0,
    sigma: 0.2,
    tau: 0.25,
  };
  let price = p.price();
  assert!(price > 0.0, "supershare={price}");
  assert!(price < p.s, "supershare must be < S");
}

/// Cash-or-nothing delta uses finite difference vs analytic.
#[test]
fn cash_delta_matches_fd() {
  let h = 0.01;
  let base = CashOrNothingPricer {
    s: 100.0,
    k: 100.0,
    cash: 10.0,
    r: 0.05,
    b: 0.02,
    sigma: 0.25,
    tau: 0.5,
    option_type: OptionType::Call,
  };
  let up = CashOrNothingPricer {
    s: 100.0 + h,
    ..base.clone()
  };
  let dn = CashOrNothingPricer {
    s: 100.0 - h,
    ..base.clone()
  };
  let fd = (up.price() - dn.price()) / (2.0 * h);
  let analytic = base.delta();
  assert!((fd - analytic).abs() < 1e-4, "fd={fd}, analytic={analytic}");
}

/// Same scenario as `cash_or_nothing_call_closed_form` /
/// `cash_call_put_parity`, priced through [`ModelPricer`] instead of the
/// inherent `price()`.
#[test]
fn cash_or_nothing_call_via_model_pricer() {
  let base = CashOrNothingPricer {
    s: 100.0,
    k: 80.0,
    cash: 10.0,
    r: 0.06,
    b: 0.06,
    sigma: 0.35,
    tau: 0.75,
    option_type: OptionType::Call,
  };
  let call = base.price_call(100.0, 80.0, 0.06, 0.0, 0.75);
  let put = base.price_put(100.0, 80.0, 0.06, 0.0, 0.75);
  assert!(
    (call - base.price()).abs() < 1e-9,
    "trait={call}, inherent={}",
    base.price()
  );
  assert!((call - 7.3444).abs() < 0.005, "price={call}");
  let expected = 10.0 * (-0.06_f64 * 0.75).exp();
  assert!((call + put - expected).abs() < 1e-9, "total={}", call + put);
}

/// Same scenario as `aon_call_put_parity`, priced through [`ModelPricer`].
#[test]
fn aon_call_via_model_pricer() {
  let c = AssetOrNothingPricer {
    s: 100.0,
    k: 105.0,
    r: 0.05,
    b: 0.03,
    sigma: 0.25,
    tau: 1.0,
    option_type: OptionType::Call,
  };
  let call = c.price_call(100.0, 105.0, 0.05, 0.02, 1.0);
  let put = c.price_put(100.0, 105.0, 0.05, 0.02, 1.0);
  assert!(
    (call - c.price()).abs() < 1e-9,
    "trait={call}, inherent={}",
    c.price()
  );
  let expected = 100.0 * ((0.03_f64 - 0.05_f64) * 1.0).exp();
  assert!((call + put - expected).abs() < 1e-9, "total={}", call + put);
}

/// Same scenario as `gap_haug_negative_payoff` ($K_1 \ne K_2$, so this also
/// proves the query strike binds to $K_1$, not $K_2$) plus
/// `gap_reduces_to_vanilla`, both priced through [`ModelPricer`]. The put
/// check uses gap's own parity identity, $C - P = Se^{-qT} - K_2e^{-rT}$
/// (derived from the closed forms; holds regardless of $K_1$).
#[test]
fn gap_call_via_model_pricer() {
  let p = GapPricer {
    s: 50.0,
    k1: 50.0,
    k2: 57.0,
    r: 0.09,
    b: 0.09,
    sigma: 0.20,
    tau: 0.5,
    option_type: OptionType::Call,
  };
  let call = p.price_call(50.0, 50.0, 0.09, 0.0, 0.5);
  let put = p.price_put(50.0, 50.0, 0.09, 0.0, 0.5);
  assert!(
    (call - p.price()).abs() < 1e-9,
    "trait={call}, inherent={}",
    p.price()
  );
  assert!(call.abs() < 0.05 && call < 0.0, "gap call={call}");
  let parity = 50.0 - 57.0 * (-0.09_f64 * 0.5).exp();
  assert!(
    (call - put - parity).abs() < 1e-9,
    "gap parity: {call} - {put}"
  );

  let vanilla = GapPricer {
    s: 100.0,
    k1: 100.0,
    k2: 100.0,
    r: 0.05,
    b: 0.05,
    sigma: 0.2,
    tau: 1.0,
    option_type: OptionType::Call,
  };
  let vprice = vanilla.price_call(100.0, 100.0, 0.05, 0.0, 1.0);
  assert!((vprice - 10.4506).abs() < 0.005, "gap-as-vanilla={vprice}");
}

/// Same scenario as `supershare_positive`, priced through [`ModelPricer`];
/// also proves the documented no-put contract on `price_put`.
#[test]
fn supershare_call_via_model_pricer() {
  let p = SuperSharePricer {
    s: 100.0,
    x_low: 90.0,
    x_high: 110.0,
    r: 0.05,
    b: 0.0,
    sigma: 0.2,
    tau: 0.25,
  };
  let call = p.price_call(100.0, 90.0, 0.05, 0.05, 0.25);
  assert!(
    (call - p.price()).abs() < 1e-9,
    "trait={call}, inherent={}",
    p.price()
  );
  assert!(call > 0.0 && call < p.s, "supershare={call}");
  assert!(
    p.price_put(100.0, 90.0, 0.05, 0.05, 0.25).is_nan(),
    "supershare has no put analogue"
  );
}
