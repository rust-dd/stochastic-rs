#[cfg(feature = "openblas")]
use ndarray::array;

use super::*;

/// Cross-arch tolerance: these goldens come from `biv_norm` and
/// `norm_cdf`, whose last bit is a hostage to FMA contraction and libm
/// differences between the aarch64-darwin dev machine and CI's ubuntu
/// x86_64.
const TOL: f64 = 1e-12;

/// Values captured from the bundled-market-data `StulzRainbowPricer`
/// **before** the model/query reshape. The reshape is an API change only,
/// so these must not move. All four payoffs are pinned, because
/// `PutOnMin` and `PutOnMax` route through `MargrabePricer`, which the
/// same wave reshaped one commit earlier.
#[test]
fn stulz_matches_pre_refactor_goldens() {
  let expected = [
    (RainbowPayoff::CallOnMin, 6.572032430799396),
    (RainbowPayoff::CallOnMax, 21.3836021143453),
    (RainbowPayoff::PutOnMin, 10.164180157272469),
    (RainbowPayoff::PutOnMax, 3.0373392880150476),
  ];
  for (payoff, want) in expected {
    let got = StulzRainbowPricer::new(payoff, 0.20, 0.30, 0.5)
      .price(100.0, 105.0, 100.0, 0.05, 0.0, 0.0, 1.0);
    assert!((got - want).abs() < TOL, "{payoff:?} {got}");
  }

  // Asymmetric: distinct spots, strike, both dividend yields, a negative
  // correlation and a non-unit maturity, so a query field left behind on
  // the struct could not survive by coinciding with a default.
  let asymmetric = [
    (RainbowPayoff::CallOnMin, 4.089461008811323),
    (RainbowPayoff::CallOnMax, 39.80815123244825),
    (RainbowPayoff::PutOnMin, 18.459568337156256),
    (RainbowPayoff::PutOnMax, 0.4223483563151831),
  ];
  for (payoff, want) in asymmetric {
    let got = StulzRainbowPricer::new(payoff, 0.33, 0.19, -0.4)
      .price(88.0, 121.0, 95.0, 0.037, 0.021, 0.013, 1.75);
    assert!((got - want).abs() < TOL, "{payoff:?} {got}");
  }
}

/// One model instance prices a whole strike grid — the point of the split.
#[test]
fn stulz_one_model_prices_a_strike_grid() {
  let model = StulzRainbowPricer::new(RainbowPayoff::CallOnMin, 0.25, 0.30, 0.4);
  let prices = [90.0, 100.0, 110.0].map(|k| model.price(100.0, 100.0, k, 0.05, 0.0, 0.0, 1.0));
  assert!(
    prices[0] > prices[1] && prices[1] > prices[2],
    "worst-of calls must decay in the strike: {prices:?}"
  );
}

/// The maturity is a query argument, so one instance covers a term
/// structure. A `tau` cached at construction would return the same number
/// three times.
#[test]
fn stulz_one_model_prices_a_maturity_grid() {
  let model = StulzRainbowPricer::new(RainbowPayoff::CallOnMax, 0.25, 0.30, 0.4);
  let prices = [0.25, 1.0, 4.0].map(|tau| model.price(100.0, 100.0, 100.0, 0.05, 0.0, 0.0, tau));
  assert!(
    prices[0] < prices[1] && prices[1] < prices[2],
    "best-of calls must rise in tau: {prices:?}"
  );
}

/// Stulz: $C_{\min} + C_{\max} = C_1 + C_2$ (vanilla call sum).
#[test]
fn stulz_min_max_decomposition() {
  use crate::pricing::bsm::BSMCoc;
  use crate::pricing::bsm::BSMPricer;
  use crate::traits::ModelPricer;

  let s1 = 100.0;
  let s2 = 105.0;
  let k = 100.0;
  let v1 = 0.20;
  let v2 = 0.30;
  let rho = 0.5;
  let r = 0.05;
  let q1 = 0.0;
  let q2 = 0.0;
  let tau = 1.0;
  let cmin =
    StulzRainbowPricer::new(RainbowPayoff::CallOnMin, v1, v2, rho).price(s1, s2, k, r, q1, q2, tau);
  let cmax =
    StulzRainbowPricer::new(RainbowPayoff::CallOnMax, v1, v2, rho).price(s1, s2, k, r, q1, q2, tau);
  let c1 = BSMPricer::new(v1, BSMCoc::Merton1973).price_call(s1, k, r, q1, tau);
  let c2 = BSMPricer::new(v2, BSMCoc::Merton1973).price_call(s2, k, r, q2, tau);
  let lhs = cmin + cmax;
  let rhs = c1 + c2;
  assert!((lhs - rhs).abs() < 0.01, "lhs={lhs}, rhs={rhs}");
}

/// Stulz call-on-min should match Monte Carlo within 2%.
#[cfg(feature = "openblas")]
#[test]
fn stulz_min_matches_mc() {
  let stulz = StulzRainbowPricer::new(RainbowPayoff::CallOnMin, 0.25, 0.30, 0.4)
    .price(100.0, 100.0, 100.0, 0.05, 0.0, 0.0, 1.0);
  let s = array![100.0, 100.0];
  let q = array![0.0, 0.0];
  let mc = McRainbowPricer::new(
    RainbowPayoff::CallOnMin,
    array![0.25, 0.30],
    array![[1.0, 0.4], [0.4, 1.0]],
    200_000,
  )
  .price(s.view(), 100.0, 0.05, q.view(), 1.0);
  let rel = (stulz - mc.mean).abs() / stulz.max(1e-10);
  assert!(rel < 0.03, "stulz={stulz}, mc={mc}, rel={rel}");
  assert!(
    mc.std_err > 0.0 && mc.std_err < 0.01 * stulz,
    "std_err out of range for n=200k: {mc}"
  );
}

/// CallOnMax >= each individual vanilla call (always have at least one
/// asset path in the money).
#[test]
fn call_on_max_dominates_vanilla() {
  use crate::pricing::bsm::BSMCoc;
  use crate::pricing::bsm::BSMPricer;
  use crate::traits::ModelPricer;

  let s1 = 100.0;
  let s2 = 100.0;
  let v1 = 0.25;
  let v2 = 0.25;
  let rho = 0.0;
  let cmax = StulzRainbowPricer::new(RainbowPayoff::CallOnMax, v1, v2, rho)
    .price(s1, s2, 100.0, 0.05, 0.0, 0.0, 1.0);
  let c1 = BSMPricer::new(v1, BSMCoc::Merton1973).price_call(s1, 100.0, 0.05, 0.0, 1.0);
  assert!(cmax > c1, "cmax={cmax} should be > c1={c1}");
}

/// 5-asset MC rainbow CallOnMax should be greater than CallOnMin.
#[cfg(feature = "openblas")]
#[test]
fn mc_call_on_max_above_min() {
  let n = 5;
  let s = Array1::from_elem(n, 100.0);
  let sig = Array1::from_elem(n, 0.25);
  let q = Array1::from_elem(n, 0.0);
  let mut rho = Array2::<f64>::from_elem((n, n), 0.3);
  for i in 0..n {
    rho[[i, i]] = 1.0;
  }
  let mc_max = McRainbowPricer::new(RainbowPayoff::CallOnMax, sig.clone(), rho.clone(), 50_000)
    .price(s.view(), 100.0, 0.05, q.view(), 1.0);
  let mc_min = McRainbowPricer::new(RainbowPayoff::CallOnMin, sig, rho, 50_000).price(
    s.view(),
    100.0,
    0.05,
    q.view(),
    1.0,
  );
  assert!(mc_max.mean > mc_min.mean);
}

/// One Monte Carlo model instance prices a whole strike grid. The
/// strikes are far enough apart that the ordering survives the sampling
/// error of independent simulations.
#[cfg(feature = "openblas")]
#[test]
fn mc_rainbow_one_model_prices_a_strike_grid() {
  let s = array![100.0, 100.0];
  let q = array![0.0, 0.0];
  let model = McRainbowPricer::new(
    RainbowPayoff::CallOnMax,
    array![0.25, 0.30],
    array![[1.0, 0.4], [0.4, 1.0]],
    50_000,
  );
  let prices = [80.0, 100.0, 130.0].map(|k| model.price(s.view(), k, 0.05, q.view(), 1.0).mean);
  assert!(
    prices[0] > prices[1] && prices[1] > prices[2],
    "best-of calls must decay in the strike: {prices:?}"
  );
}

/// The model fixes how many assets there are; a query that disagrees is
/// reported by `try_price` as an `Err`, not a panic. Pinned because that
/// is the reason the check did not move to the constructor.
#[cfg(feature = "openblas")]
#[test]
fn mc_rainbow_try_price_reports_a_query_dimension_mismatch() {
  let model = McRainbowPricer::new(
    RainbowPayoff::CallOnMin,
    array![0.25, 0.30],
    array![[1.0, 0.4], [0.4, 1.0]],
    1_000,
  );
  let s = array![100.0, 100.0, 100.0];
  let q = array![0.0, 0.0, 0.0];
  let err = model
    .try_price(s.view(), 100.0, 0.05, q.view(), 1.0)
    .expect_err("a three-asset query against a two-asset model is not priceable");
  assert!(
    err.to_string().contains("does not match n_assets=3"),
    "{err}"
  );
}

/// A correlation matrix that is symmetric but not positive definite is
/// also an `Err` rather than a panic — the other half of what keeps the
/// constructor unguarded.
#[cfg(feature = "openblas")]
#[test]
fn mc_rainbow_try_price_reports_a_non_spd_correlation() {
  let model = McRainbowPricer::new(
    RainbowPayoff::CallOnMin,
    array![0.25, 0.30],
    array![[1.0, 2.0], [2.0, 1.0]],
    1_000,
  );
  let s = array![100.0, 100.0];
  let q = array![0.0, 0.0];
  let err = model
    .try_price(s.view(), 100.0, 0.05, q.view(), 1.0)
    .expect_err("rho = 2 is not a correlation");
  assert!(err.to_string().contains("not positive definite"), "{err}");
}

/// `StulzRainbowPricer::new` rejects a parameter that is not a
/// volatility or not a correlation; `McRainbowPricer::new` rejects only
/// the volatilities and the path count, leaving `rho` and the shapes to
/// `try_price`.
///
/// The two messages are prefixed by their own type, so a `should_panic`
/// anchored on one cannot be satisfied by the other firing.
mod construction_validation {
  use super::*;

  /// The negative call: `CallOnMax` at `sigma2 = -0.30` prices at
  /// **-11.382123**, while the `CallOnMin` leg of the same model returns
  /// a plausible `14.204697` against the correct `6.572032`.
  #[test]
  #[should_panic(
    expected = "StulzRainbowPricer::new: sigma2 must be a non-negative volatility (got -0.3)"
  )]
  fn stulz_rejects_a_negative_second_volatility() {
    let _ = StulzRainbowPricer::new(RainbowPayoff::CallOnMax, 0.20, -0.30, 0.5);
  }

  /// `8.485818` on the `CallOnMin` leg, `3.445723` on `CallOnMax`.
  #[test]
  #[should_panic(
    expected = "StulzRainbowPricer::new: sigma1 must be a non-negative volatility (got -0.2)"
  )]
  fn stulz_rejects_a_negative_first_volatility() {
    let _ = StulzRainbowPricer::new(RainbowPayoff::CallOnMin, -0.20, 0.30, 0.5);
  }

  /// Not a wrong number — an assertion inside the third-party
  /// `owens_t::biv_norm` whose message is the bare offending float,
  /// `13000000.000000002`, naming neither the parameter nor the pricer.
  /// The same `rho` is a silent wrong number on the `PutOnMin` /
  /// `PutOnMax` legs, which route through `MargrabePricer`.
  #[test]
  #[should_panic(expected = "StulzRainbowPricer::new: rho must be in [-1, 1] (got 5)")]
  fn stulz_rejects_a_correlation_above_one() {
    let _ = StulzRainbowPricer::new(RainbowPayoff::CallOnMin, 0.20, 0.30, 5.0);
  }

  #[test]
  #[should_panic(expected = "StulzRainbowPricer::new: rho must be in [-1, 1] (got -5)")]
  fn stulz_rejects_a_correlation_below_minus_one() {
    let _ = StulzRainbowPricer::new(RainbowPayoff::CallOnMin, 0.20, 0.30, -5.0);
  }

  /// Perfect correlation either way and a zero-volatility leg are
  /// admissible and stay accepted.
  #[test]
  fn the_admissible_edges_stay_constructible() {
    assert_eq!(
      StulzRainbowPricer::new(RainbowPayoff::CallOnMin, 0.20, 0.30, 1.0).rho,
      1.0
    );
    assert_eq!(
      StulzRainbowPricer::new(RainbowPayoff::CallOnMax, 0.0, 0.30, -1.0).sigma1,
      0.0
    );
  }

  #[cfg(feature = "openblas")]
  #[test]
  #[should_panic(
    expected = "McRainbowPricer::new: sigma[1] must be a non-negative volatility (got -0.3)"
  )]
  fn mc_rainbow_rejects_a_negative_volatility() {
    let _ = McRainbowPricer::new(
      RainbowPayoff::CallOnMin,
      array![0.25, -0.30],
      array![[1.0, 0.4], [0.4, 1.0]],
      1_000,
    );
  }

  #[cfg(feature = "openblas")]
  #[test]
  #[should_panic(expected = "McRainbowPricer::new: n_paths must be at least 1 (got 0)")]
  fn mc_rainbow_rejects_a_zero_path_count() {
    let _ = McRainbowPricer::new(
      RainbowPayoff::CallOnMin,
      array![0.25, 0.30],
      array![[1.0, 0.4], [0.4, 1.0]],
      0,
    );
  }

  /// The deliberate omission, pinned so it cannot drift: an out-of-range
  /// `rho` entry stays constructible, because
  /// `mc_rainbow_try_price_reports_a_non_spd_correlation` needs it to
  /// reach `try_price` and come back as an `Err` rather than a panic.
  #[cfg(feature = "openblas")]
  #[test]
  fn mc_rainbow_leaves_the_correlation_matrix_to_try_price() {
    let model = McRainbowPricer::new(
      RainbowPayoff::CallOnMin,
      array![0.25, 0.30],
      array![[1.0, 2.0], [2.0, 1.0]],
      1_000,
    );
    assert_eq!(model.rho[[0, 1]], 2.0);
  }
}

/// A `NaN` leg used to be **dropped**, so an $n$-asset best-of priced as
/// an $(n-1)$-asset best-of.
///
/// The identity with the two-asset answer is what makes it a silent
/// defect rather than a visible one: `CallOnMax` on `[120, NaN, 90]` at
/// `K = 100` returned `20.0`, bit-for-bit the value of the same contract
/// written on `[120, 90]`. Nothing in the number marks the third asset
/// as missing.
///
/// All four payoffs are pinned. Two of them (`CallOnMin`, `PutOnMax`)
/// returned `0.0` instead, through the *second* copy of the same trap —
/// the surviving `(min_p - k).max(0.0)` floor — so a fix to the fold
/// alone would have left them laundering.
#[test]
fn a_nan_leg_poisons_the_rainbow_payoff_instead_of_dropping_out() {
  let legs = [120.0, f64::NAN, 90.0];
  for payoff in [
    RainbowPayoff::CallOnMax,
    RainbowPayoff::CallOnMin,
    RainbowPayoff::PutOnMax,
    RainbowPayoff::PutOnMin,
  ] {
    let got = payoff.evaluate(&legs, 100.0);
    assert!(got.is_nan(), "{payoff:?} on a NaN leg must not pay: {got}");
    // Every payoff is also poisoned by an undefined strike.
    let by_strike = payoff.evaluate(&[120.0, 90.0], f64::NAN);
    assert!(
      by_strike.is_nan(),
      "{payoff:?} at a NaN strike must not pay: {by_strike}"
    );
  }

  // The two-asset value the three-asset contract used to impersonate.
  assert_eq!(
    RainbowPayoff::CallOnMax.evaluate(&[120.0, 90.0], 100.0),
    20.0
  );
  // And the floor is still a floor for a real, out-of-the-money basket.
  assert_eq!(
    RainbowPayoff::CallOnMax.evaluate(&[80.0, 90.0], 100.0),
    0.0,
    "a worthless best-of still pays zero"
  );
  assert_eq!(
    RainbowPayoff::PutOnMin.evaluate(&[120.0, 90.0], 100.0),
    10.0,
    "the surviving payoffs must be unchanged"
  );
}

/// Stulz put-on-min via parity should be positive.
#[test]
fn stulz_put_on_min_positive() {
  let p = StulzRainbowPricer::new(RainbowPayoff::PutOnMin, 0.25, 0.20, 0.3);
  let price = p.price(100.0, 105.0, 100.0, 0.05, 0.0, 0.0, 0.5);
  assert!(price > 0.0, "put_on_min={price}");
}
