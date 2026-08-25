use super::*;

/// Cash-or-nothing call: S=100, K=80, Q=10, r=0.06, b=0.06, sigma=0.35,
/// T=0.75 → 7.3444 (verified analytically against Q*e^{-rT}*N(d2)).
///
/// `b = 0.06` is expressed as the query's `q = r - b = 0`, which is how
/// every scenario in this file names its cost of carry now that the model
/// derives $b$ from the query instead of storing it.
#[test]
fn cash_or_nothing_call_closed_form() {
  let p = CashOrNothingPricer::new(10.0, 0.35);
  let price = p.price_call(100.0, 80.0, 0.06, 0.0, 0.75);
  assert!((price - 7.3444).abs() < 0.005, "price={price}");
}

/// Cash-or-nothing put: S=100, K=80, Q=10, r=0.06, b=0.06, sigma=0.35,
/// T=0.75 → ~ Q*e^{-rT}*N(-d2) = 10 * 0.95599 * (1 - 0.97264/Q*e^{rT})
/// ≈ 0.0303
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

/// Cash-or-nothing delta uses finite difference vs analytic, on both option
/// types — the put's delta is the call's negated, and checking only the call
/// would leave the `option_type` wiring of every digital Greek unpinned.
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

/// The aggregate is the only place the exposed Greeks are mapped into
/// [`Greeks`]'s nine members, so it gets the same guard `BSMPricer`'s does:
/// every member either equals its accessor or is `NaN`, and the `NaN`s are
/// the ones the pricer genuinely does not expose (which is what the removed
/// `GreeksExt` defaults produced). A hand-written struct literal at a call
/// site is what loses this pin.
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

/// The gap put has its own parity identity, $C - P = Se^{-qT} - K_2e^{-rT}$
/// (derived from the closed forms; holds regardless of $K_1$), which is not
/// the trait's vanilla parity against the query strike — the two differ
/// exactly because $K_1 \ne K_2$ here. Same scenario as
/// `gap_haug_negative_payoff`, so this doubles as proof that the query
/// strike binds to $K_1$ and `self.k2` to the payoff strike.
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

/// The documented no-put contract: Haug (2007) defines only the supershare
/// payoff, so `price_put` returns `NaN` rather than the trait's
/// vanilla-parity default.
#[test]
fn supershare_has_no_put_analogue() {
  let p = SuperSharePricer::new(110.0, 0.2);
  assert!(
    p.price_put(100.0, 90.0, 0.05, 0.05, 0.25).is_nan(),
    "supershare has no put analogue"
  );
}

/// One model instance prices a whole strike/maturity grid — the capability
/// [`ModelPricer`] exists for and the pre-query struct could not offer
/// without rebuilding itself per point.
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

/// Every argument the trait hands a digital must reach its price. A
/// parameter that is accepted and then ignored — or precomputed at
/// construction from an earlier query — returns a plausible in-range number
/// with nothing to distinguish it, so each of the five is perturbed on its
/// own and required to move the price.
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
