use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

/// The goldens' last bit is `norm_cdf`'s, which FMA contraction and libm move between
/// aarch64-darwin and CI's ubuntu x86_64.
const TOL: f64 = 1e-12;

/// Regression goldens for Margrabe's price and deltas at and in the money.
#[test]
fn margrabe_matches_pre_refactor_goldens() {
  let atm = MargrabePricer::new(0.20, 0.20, 0.0);
  let price = atm.price(100.0, 100.0, 0.0, 0.0, 1.0);
  assert!((price - 11.246291601828489).abs() < TOL, "atm {price}");
  let d1 = atm.delta1(100.0, 100.0, 0.0, 0.0, 1.0);
  assert!((d1 - 0.5562314580091424).abs() < TOL, "delta1 {d1}");
  let d2 = atm.delta2(100.0, 100.0, 0.0, 0.0, 1.0);
  assert!((d2 + 0.4437685419908575).abs() < TOL, "delta2 {d2}");

  let itm = MargrabePricer::new(0.20, 0.20, 0.5);
  let price = itm.price(200.0, 100.0, 0.01, 0.02, 0.5);
  assert!((price - 99.99751394787123).abs() < TOL, "itm {price}");

  let skewed = MargrabePricer::new(0.31, 0.17, -0.25);
  let price = skewed.price(95.0, 105.0, 0.03, 0.011, 2.25);
  assert!((price - 15.76554946677107).abs() < TOL, "skewed {price}");
  let d1 = skewed.delta1(95.0, 105.0, 0.03, 0.011, 2.25);
  assert!((d1 - 0.4848890339715629).abs() < TOL, "delta1 {d1}");
  let d2 = skewed.delta2(95.0, 105.0, 0.03, 0.011, 2.25);
  assert!((d2 + 0.28856103581454673).abs() < TOL, "delta2 {d2}");
}

/// One model instance prices a whole query grid.
#[test]
fn margrabe_one_model_prices_a_spot_grid() {
  let model = MargrabePricer::new(0.25, 0.20, 0.4);
  let prices = [90.0, 100.0, 110.0].map(|s1| model.price(s1, 100.0, 0.0, 0.0, 1.0));
  assert!(
    prices[0] < prices[1] && prices[1] < prices[2],
    "the exchange option must rise in S1: {prices:?}"
  );
}

/// One instance covers a term structure, which a `tau` cached at construction would answer with
/// one number.
#[test]
fn margrabe_one_model_prices_a_maturity_grid() {
  let model = MargrabePricer::new(0.25, 0.20, 0.4);
  let prices = [0.25, 1.0, 4.0].map(|tau| model.price(100.0, 100.0, 0.0, 0.0, tau));
  assert!(
    prices[0] < prices[1] && prices[1] < prices[2],
    "an at-the-money exchange option must rise in tau: {prices:?}"
  );
}

/// Margrabe with σ1=σ2 and ρ=1 must equal $\max(S_1 e^{-q_1 T} - S_2
/// e^{-q_2 T}, 0)$ — the spread is deterministic at maturity.
#[test]
fn margrabe_perfect_correlation_equal_vol() {
  let price = MargrabePricer::new(0.2, 0.2, 1.0).price(100.0, 100.0, 0.0, 0.0, 1.0);
  assert!(price.abs() < 1e-8, "perfect-corr Margrabe={price}");
}

/// At the money, `ρ = 0` and both vols `0.2`: `σ_M = √0.08`, `V = 100(2N(σ_M/2) − 1) ≈ 11.246`.
#[test]
fn margrabe_atm_zero_corr() {
  let price = MargrabePricer::new(0.20, 0.20, 0.0).price(100.0, 100.0, 0.0, 0.0, 1.0);
  let expected = 11.246;
  assert!((price - expected).abs() < 0.05, "Margrabe ATM={price}");
}

/// Margrabe with $S_1 \gg S_2$ approaches the discounted intrinsic.
#[test]
fn margrabe_deep_itm() {
  let price = MargrabePricer::new(0.20, 0.20, 0.5).price(200.0, 100.0, 0.01, 0.02, 0.5);
  let intrinsic = 200.0 * (-0.01_f64 * 0.5).exp() - 100.0 * (-0.02_f64 * 0.5).exp();
  assert!(
    price > intrinsic,
    "Margrabe deep ITM={price} vs intrinsic={intrinsic}"
  );
}

/// One Monte Carlo instance prices a strike grid, both legs; its pinned seed gives every strike the
/// same paths, so the order holds path by path.
#[test]
fn mc_spread_one_model_prices_a_strike_grid() {
  let model = McSpreadPricer::new(0.30, 0.25, 0.3, 100_000).with_seed(Deterministic::new(42));
  let calls =
    [0.0, 10.0, 25.0].map(|k| model.price_call(110.0, 100.0, k, 0.03, 0.0, 0.0, 1.0).mean);
  let puts = [0.0, 10.0, 25.0].map(|k| model.price_put(110.0, 100.0, k, 0.03, 0.0, 0.0, 1.0).mean);
  assert!(
    calls[0] > calls[1] && calls[1] > calls[2],
    "spread calls must decay in the strike: {calls:?}"
  );
  assert!(
    puts[0] < puts[1] && puts[1] < puts[2],
    "spread puts must rise in the strike: {puts:?}"
  );
}

/// A NaN `tau` on the degenerate branch (`sigma1 == sigma2`, `rho == 1`) prices NaN, as on every
/// other branch, not a confident `0.0`.
#[test]
fn margrabe_does_not_launder_a_nan_query_on_the_degenerate_branch() {
  let degenerate = MargrabePricer::new(0.2, 0.2, 1.0);
  assert_eq!(
    degenerate.combined_variance(),
    0.0,
    "this model must actually reach the degenerate branch"
  );
  for (name, got) in [
    ("tau", degenerate.price(100.0, 100.0, 0.0, 0.0, f64::NAN)),
    ("s1", degenerate.price(f64::NAN, 100.0, 0.0, 0.0, 1.0)),
    ("q1", degenerate.price(100.0, 100.0, f64::NAN, 0.0, 1.0)),
  ] {
    assert!(got.is_nan(), "a NaN {name} must not price: got {got}");
  }
  // The non-degenerate model already propagated, and must keep doing so.
  assert!(
    MargrabePricer::new(0.25, 0.20, 0.4)
      .price(100.0, 100.0, 0.0, 0.0, f64::NAN)
      .is_nan()
  );
  // The floor itself is untouched: the branch is still the discounted
  // intrinsic, floored at zero.
  assert_eq!(degenerate.price(100.0, 120.0, 0.0, 0.0, 1.0), 0.0);
  assert!((degenerate.price(120.0, 100.0, 0.0, 0.0, 1.0) - 20.0).abs() < 1e-12);
}

/// A NaN payoff passes the per-path `max(0)` floor, where a bare floor would zero every path.
#[test]
fn mc_spread_does_not_launder_a_nan_into_a_zero_price() {
  let model = McSpreadPricer::new(0.25, 0.20, 0.4, 2_000).with_seed(Deterministic::new(42));
  for (name, got) in [
    (
      "s1",
      model.price_call(f64::NAN, 100.0, 10.0, 0.02, 0.0, 0.0, 1.0),
    ),
    (
      "s2",
      model.price_call(110.0, f64::NAN, 10.0, 0.02, 0.0, 0.0, 1.0),
    ),
    (
      "k",
      model.price_call(110.0, 100.0, f64::NAN, 0.02, 0.0, 0.0, 1.0),
    ),
    (
      "q1",
      model.price_call(110.0, 100.0, 10.0, 0.02, f64::NAN, 0.0, 1.0),
    ),
    (
      "put s1",
      model.price_put(f64::NAN, 100.0, 10.0, 0.02, 0.0, 0.0, 1.0),
    ),
  ] {
    assert!(got.mean.is_nan(), "a NaN {name} must not price: got {got}");
    assert!(
      got.std_err.is_nan(),
      "a poisoned run has no error bar either: got {got}"
    );
  }

  // The floor is still a floor: a deep out-of-the-money spread call is
  // worth zero, not a small negative number.
  let deep = model.price_call(110.0, 100.0, 500.0, 0.02, 0.0, 0.0, 1.0);
  assert_eq!(deep.mean, 0.0, "the max(0) floor must survive: {deep}");
  assert_eq!(
    deep.std_err, 0.0,
    "constant payoffs have zero error: {deep}"
  );
}

/// Both constructors refuse a volatility or correlation off its domain: past `rho = 1` Margrabe
/// priced its discounted intrinsic, and `McSpreadPricer`'s `(1 − ρ²).max(0)` dropped `z₂`.
mod construction_validation {
  use super::*;

  /// `rho = 5` priced the discounted intrinsic, `10.0` against `16.190433`.
  #[test]
  #[should_panic(expected = "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = 5.0")]
  fn margrabe_rejects_a_correlation_above_one() {
    let _ = MargrabePricer::new(0.25, 0.20, 5.0);
  }

  /// `rho = -5` merely inflates the variance: `36.943253`.
  #[test]
  #[should_panic(expected = "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = -5.0")]
  fn margrabe_rejects_a_correlation_below_minus_one() {
    let _ = MargrabePricer::new(0.25, 0.20, -5.0);
  }

  /// `21.211364` at either sign flip — the two volatilities enter the
  /// combined variance the same way, so both are checked.
  #[test]
  #[should_panic(
    expected = "sigma1 must satisfy `sigma1.is_finite() && sigma1 >= 0.0`, got sigma1 = -0.25"
  )]
  fn margrabe_rejects_a_negative_first_volatility() {
    let _ = MargrabePricer::new(-0.25, 0.20, 0.4);
  }

  #[test]
  #[should_panic(
    expected = "sigma2 must satisfy `sigma2.is_finite() && sigma2 >= 0.0`, got sigma2 = -0.2"
  )]
  fn margrabe_rejects_a_negative_second_volatility() {
    let _ = MargrabePricer::new(0.25, -0.20, 0.4);
  }

  #[test]
  #[should_panic(expected = "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = NaN")]
  fn margrabe_rejects_a_nan_correlation() {
    let _ = MargrabePricer::new(0.25, 0.20, f64::NAN);
  }

  /// `13.620211` at `rho = 5`, `35.354151` at `rho = -5`, against
  /// `~10.69`.
  #[test]
  #[should_panic(expected = "rho must satisfy `(-1.0..=1.0).contains(&rho)`, got rho = 5.0")]
  fn mc_spread_rejects_a_correlation_above_one() {
    let _ = McSpreadPricer::new(0.25, 0.20, 5.0, 1_000);
  }

  #[test]
  #[should_panic(
    expected = "sigma1 must satisfy `sigma1.is_finite() && sigma1 >= 0.0`, got sigma1 = -0.25"
  )]
  fn mc_spread_rejects_a_negative_first_volatility() {
    let _ = McSpreadPricer::new(-0.25, 0.20, 0.4, 1_000);
  }

  #[test]
  #[should_panic(
    expected = "sigma2 must satisfy `sigma2.is_finite() && sigma2 >= 0.0`, got sigma2 = -0.2"
  )]
  fn mc_spread_rejects_a_negative_second_volatility() {
    let _ = McSpreadPricer::new(0.25, -0.20, 0.4, 1_000);
  }

  /// No path averages to NaN already; the count is refused where it is supplied.
  #[test]
  #[should_panic(expected = "n_paths must satisfy `n_paths >= 1`, got n_paths = 0")]
  fn mc_spread_rejects_a_zero_path_count() {
    let _ = McSpreadPricer::new(0.25, 0.20, 0.4, 0);
  }

  /// Perfect correlation either way and a zero-volatility leg build; equal vols at `rho == 1` are
  /// Margrabe's degenerate limit (`margrabe_perfect_correlation_equal_vol`).
  #[test]
  fn the_admissible_edges_stay_constructible() {
    assert_eq!(MargrabePricer::new(0.2, 0.2, 1.0).rho, 1.0);
    assert_eq!(MargrabePricer::new(0.2, 0.2, -1.0).rho, -1.0);
    assert_eq!(MargrabePricer::new(0.0, 0.35, 0.4).sigma1, 0.0);
    assert_eq!(McSpreadPricer::new(0.0, 0.35, -1.0, 1).n_paths, 1);
    // Inside the crate a field can still be set past `new`'s checks.
    let mut p = McSpreadPricer::new(0.25, 0.20, 0.4, 16);
    p.rho = 5.0;
    assert_eq!(p.rho, 5.0);
  }
}

/// Margrabe ↔ MC (K=0) consistency: with enough paths the MC spread call
/// should match Margrabe within 1.5%.
#[test]
fn margrabe_matches_mc_zero_strike() {
  let m_price = MargrabePricer::new(0.25, 0.20, 0.4).price(110.0, 100.0, 0.0, 0.0, 1.0);
  let mc = McSpreadPricer::new(0.25, 0.20, 0.4, 100_000).with_seed(Deterministic::new(42));
  let est = mc.price_call(110.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1.0);
  let rel = (m_price - est.mean).abs() / m_price;
  assert!(rel < 0.02, "margrabe={m_price}, mc={est}, rel={rel}");
  assert_eq!(est.n_samples, 100_000);
  assert!(
    est.std_err > 0.0 && est.std_err < 0.01 * m_price,
    "std_err out of range for n=100k: {est}"
  );
  let (lo, hi) = est.ci_95();
  assert!(lo < hi, "degenerate interval: [{lo}, {hi}]");
}
