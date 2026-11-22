use stochastic_rs_core::simd_rng::Deterministic;

use super::*;

const S: f64 = 100.0;
const K: f64 = 99.99;
const R: f64 = 0.1;
const TAU: f64 = 1.0;

/// The integration-test skill's §1.2 triple: of 200 seeds one breached the grid test's `c ≤ S`
/// (seed 156, at 579), so all three fail together near 1e-7.
const SEEDS: [u64; 3] = [2718, 999, 42];

fn pricer(seed: u64) -> GbmMalliavinPricer<Deterministic> {
  GbmMalliavinPricer::new(0.1, 2_000, 128, 0.5).with_seed(Deterministic::new(seed))
}

/// Finite and non-negative on every seed; the loose `2S` bounds share the grid test's ratio tail,
/// so they hold on the best of the three.
#[test]
fn malliavin_pricer_returns_finite_non_negative_prices() {
  let mut best_call = f64::INFINITY;
  let mut best_put = f64::INFINITY;
  for seed in SEEDS {
    let CallPut { call, put } = pricer(seed).price_call_put(S, K, R, 0.0, TAU);

    // Basic sanity checks: finite and non-negative prices
    assert!(call.is_finite(), "seed {seed}: call price should be finite");
    assert!(put.is_finite(), "seed {seed}: put price should be finite");
    assert!(call >= 0.0, "seed {seed}: call should be non-negative");
    assert!(put >= 0.0, "seed {seed}: put should be non-negative");

    // `is_finite` is asserted *before* the running min because `f64::min`
    // discards a `NaN` operand, which would hide a poisoned price.
    best_call = best_call.min(call);
    best_put = best_put.min(put);
  }

  // Very loose upper bounds; the estimator's tail is what needs three seeds.
  assert!(
    best_call < S * 2.0,
    "call price {best_call} is unreasonably large"
  );
  assert!(
    best_put < K * 2.0,
    "put price {best_put} is unreasonably large"
  );
}

/// The localised estimator's counterpart of the test above, split the same way.
#[test]
fn malliavin_pricer_localized_returns_finite_non_negative_prices() {
  let mut best_call = f64::INFINITY;
  let mut best_put = f64::INFINITY;
  for seed in SEEDS {
    let CallPut { call, put } = pricer(seed).price_call_put_localized(S, K, R, 0.0, TAU);

    // Basic sanity checks: finite and non-negative prices
    assert!(
      call.is_finite(),
      "seed {seed}: localized call should be finite"
    );
    assert!(
      put.is_finite(),
      "seed {seed}: localized put should be finite"
    );
    assert!(
      call >= 0.0,
      "seed {seed}: localized call should be non-negative"
    );
    assert!(
      put >= 0.0,
      "seed {seed}: localized put should be non-negative"
    );

    best_call = best_call.min(call);
    best_put = best_put.min(put);
  }

  // Very loose upper bounds to avoid flakiness due to Monte Carlo noise
  assert!(
    best_call < S * 2.0,
    "localized call price {best_call} is unreasonably large"
  );
  assert!(
    best_put < K * 2.0,
    "localized put price {best_put} is unreasonably large"
  );
}

/// The put satisfies put-call parity against the call from its own simulation, which the trait's
/// parity default, a second independent Monte Carlo, would not.
#[test]
fn malliavin_put_is_parity_against_its_own_call() {
  for seed in SEEDS {
    let CallPut { call, put } = pricer(seed).price_call_put(S, K, R, 0.02, TAU);
    let parity = call - S * (-0.02_f64 * TAU).exp() + K * (-R * TAU).exp();
    assert!(
      (put - parity.max(0.0)).abs() < 1e-12,
      "seed {seed}: put {put} must be the floored parity value {parity} of its own call"
    );
  }
}

/// `put == max(parity, 0)` at every point, which holds on either side of the floor and fails at
/// any point below it if the floor goes; which points floor depends on the SIMD stream.
#[test]
fn malliavin_put_is_never_negative() {
  for seed in SEEDS {
    let p = pricer(seed);
    for &k in &[1.0, 50.0, 99.99, 150.0] {
      let CallPut { call, put } = p.price_call_put(S, k, R, 0.0, TAU);
      assert!(
        put >= 0.0,
        "seed {seed}: put at K={k} must be floored, got {put}"
      );
      let parity = call - S + k * (-R * TAU).exp();
      assert!(
        (put - parity.max(0.0)).abs() < 1e-12,
        "seed {seed}, K={k}: put {put} must be the floored parity {parity}"
      );
    }
  }
}

/// `t_eval` is an absolute time, so a maturity at or inside it has no `C(t, S_t)` to average.
#[test]
fn a_maturity_at_or_inside_t_eval_is_nan() {
  let p = pricer(SEEDS[0]);
  for tau in [0.25, 0.5, 1e-300] {
    let CallPut { call, put } = p.price_call_put(S, K, R, 0.0, tau);
    assert!(call.is_nan() && put.is_nan(), "{tau}: ({call}, {put})");
    let localized = p.price_call_put_localized(S, K, R, 0.0, tau);
    assert!(localized.call.is_nan() && localized.put.is_nan(), "{tau}");
    for (s_t, c_t) in [
      p.conditional_call_malliavin(S, K, R, 0.0, tau),
      p.conditional_call_malliavin_localized(S, K, R, 0.0, tau),
    ] {
      assert!(s_t.iter().chain(&c_t).all(|x| x.is_nan()), "{tau}");
    }
  }
}

/// Read back, never written: a write would skip `new`'s checks.
#[test]
fn the_checked_fields_are_read_through_getters() {
  assert_eq!(
    (pricer(SEEDS[0]).sigma(), pricer(SEEDS[0]).t_eval()),
    (0.1, 0.5)
  );
}

/// The `l == 0` kernel must be the Dirac limit, the step the caller compares it with; otherwise the
/// weight `pdf + (H − cdf)·t2` biases the estimator invisibly.
#[test]
fn zero_bandwidth_kernel_is_the_dirac_limit() {
  for &x in &[-2.0, -1e-9, 1e-9, 2.0] {
    assert_eq!(laplace_pdf(x, 0.0), 0.0, "pdf at x={x}");
    let heaviside = if x >= 0.0 { 1.0 } else { 0.0 };
    assert_eq!(laplace_cdf(x, 0.0), heaviside, "cdf at x={x}");
    let tiny = laplace_cdf(x, 1e-12);
    assert!(
      (tiny - heaviside).abs() < 1e-12,
      "l -> 0 must approach the step off the atom: cdf({x}, 1e-12) = {tiny}"
    );
  }
}

/// At `x == 0` the `l == 0` cdf is `1.0` where every `l > 0` gives `0.5`, which cancels the
/// diagonal `j == i` term against the caller's Heaviside.
#[test]
fn the_atom_is_a_convention_not_a_limit() {
  assert_eq!(laplace_cdf(0.0, 0.0), 1.0);
  for &l in &[1e-12, 1e-3, 1.0] {
    assert_eq!(
      laplace_cdf(0.0, l),
      0.5,
      "the l > 0 family is symmetric at the atom, l={l}"
    );
  }
  let heaviside_at_tie = 1.0;
  assert_eq!(heaviside_at_tie - laplace_cdf(0.0, 0.0), 0.0);
}

/// A negative bandwidth has no limit interpretation, unlike `l == 0`.
#[test]
#[should_panic(expected = "l must satisfy `l >= 0.0`, got l = -1.0")]
fn negative_bandwidth_is_rejected() {
  let _ = laplace_pdf(0.5, -1.0);
}

#[test]
#[should_panic(expected = "l must satisfy `l >= 0.0`, got l = -1.0")]
fn negative_bandwidth_is_rejected_by_the_cdf_too() {
  let _ = laplace_cdf(0.5, -1.0);
}

/// A spot of `1e20` puts every weight `w ∝ 1/S_t` under the denominator's `1e-14` floor, so no path
/// has an estimate and the price is NaN, not the empty average's `0.0`.
#[test]
fn no_finite_pathwise_estimate_prices_nan() {
  let p = GbmMalliavinPricer::new(0.1, 64, 16, 0.5).with_seed(Deterministic::new(SEEDS[0]));
  let CallPut { call, put } = p.price_call_put(1e20, 1e20, R, 0.0, TAU);
  assert!(call.is_nan() && put.is_nan(), "({call}, {put})");
}

/// At `sigma = 0` the path `S e^{(r−q)t}` is certain, so both estimators price its discounted
/// payoff and both conditionals hold its `S_t` and conditional call, as `BarrierPricer` does.
#[test]
fn a_zero_volatility_prices_the_certain_path() {
  let p = GbmMalliavinPricer::new(0.0, 64, 16, 0.5).with_seed(Deterministic::new(SEEDS[0]));
  let k = 100.0;
  for (s, q) in [(110.0, 0.0), (90.0, 0.0), (110.0, 0.03)] {
    let (carried, strike) = (s * (-q * TAU).exp(), k * (-R * TAU).exp());
    let plain = p.price_call_put(s, k, R, q, TAU);
    let off_call = (plain.call - (carried - strike).max(0.0)).abs();
    let off_put = (plain.put - (strike - carried).max(0.0)).abs();
    assert!(off_call < 1e-12 && off_put < 1e-12, "({s}, {q}): {plain:?}");
    assert_eq!(
      p.price_call_put_localized(s, k, R, q, TAU),
      plain,
      "({s}, {q})"
    );
    for (s_t, c_t) in [
      p.conditional_call_malliavin(s, k, R, q, TAU),
      p.conditional_call_malliavin_localized(s, k, R, q, TAU),
    ] {
      let on_path = s_t
        .iter()
        .all(|x| (x - s * ((R - q) * 0.5).exp()).abs() < 1e-12);
      let towered = c_t
        .iter()
        .all(|c| (c * (-R * 0.5).exp() - plain.call).abs() < 1e-12);
      assert!(on_path && towered, "({s}, {q}): {s_t} {c_t}");
    }
  }
  let inside = p.price_call_put(110.0, k, R, 0.0, 0.5);
  assert!(inside.call.is_nan() && inside.put.is_nan(), "{inside:?}");
}

/// A spot of `1e300` overflows `payoff²`, so the localised bandwidth `lf = √(∞/∞)` is NaN: the
/// price is NaN, as the plain estimator's is, and the kernel's assert is never reached.
#[test]
fn a_bandwidth_that_is_not_finite_prices_nan() {
  let p = GbmMalliavinPricer::new(0.2, 64, 16, 0.5).with_seed(Deterministic::new(7));
  let localized = p.price_call_put_localized(1e300, 100.0, 0.1, -5.0, 1.0);
  assert!(
    localized.call.is_nan() && localized.put.is_nan(),
    "{localized:?}"
  );
  let plain = p.price_call_put(1e300, 100.0, 0.1, -5.0, 1.0);
  assert!(plain.call.is_nan() && plain.put.is_nan(), "{plain:?}");
}

/// A strike no path reaches zeroes `lf` and empties the localisation sum: the price is finite and
/// non-negative, and the same pricer still prices a live strike above it.
#[test]
fn an_all_worthless_sample_empties_the_kernel_without_poisoning_the_price() {
  for seed in SEEDS {
    let p = GbmMalliavinPricer::new(0.1, 200, 32, 0.5).with_seed(Deterministic::new(seed));
    let dead = p.price_call_put_localized(S, 1.0e12, R, 0.0, TAU).call;
    assert!(
      dead.is_finite() && dead >= 0.0,
      "seed {seed}: unreachable strike must price to a finite floor, got {dead}"
    );
    // The same instance, so the live strike is priced off the same seed —
    // the dead/live comparison is one sample, not two.
    let live = p.price_call_put_localized(S, K, R, 0.0, TAU).call;
    assert!(
      live > dead,
      "seed {seed}: live strike {live} must beat dead strike {dead}"
    );
  }
}

/// One model prices a grid: every price is finite and non-negative, and `c ≤ S` holds on the best
/// of three seeds, as the ratio estimator's thin denominator sends a point arbitrarily high.
#[test]
fn malliavin_one_model_prices_a_grid() {
  // `is_finite` before the running max, which `f64::max` would let a NaN slip through.
  let sweep = |seed: u64| {
    let model = GbmMalliavinPricer::new(0.2, 400, 64, 0.25).with_seed(Deterministic::new(seed));
    let mut worst = f64::NEG_INFINITY;
    for &tau in &[0.5_f64, 1.0] {
      for &k in &[90.0_f64, 100.0, 110.0] {
        let c = model.price_call(S, k, 0.03, 0.01, tau);
        assert!(
          c.is_finite() && c >= 0.0,
          "seed {seed}: call {c} at k={k} tau={tau} is not a price"
        );
        worst = worst.max(c);
      }
    }
    worst
  };

  let best = SEEDS.into_iter().map(sweep).fold(f64::INFINITY, f64::min);
  assert!(
    best <= S,
    "all three seeds breached the no-arbitrage ceiling; \
     best worst-point {best} against S = {S}"
  );
}

/// A pinned seed makes the price a function of the query; the third assert fails if `sample_paths`
/// takes `seed.derive()`, which advances the pricer's own stream between calls.
#[test]
fn a_pinned_seed_makes_the_estimator_a_function_of_its_query() {
  let p = pricer(SEEDS[0]);
  let first = p.price_call(S, K, R, 0.0, TAU);
  assert_eq!(
    first,
    pricer(SEEDS[0]).price_call(S, K, R, 0.0, TAU),
    "a fresh instance on the same seed must reproduce the price"
  );
  assert_ne!(
    first,
    pricer(SEEDS[1]).price_call(S, K, R, 0.0, TAU),
    "a different seed must give a different sample, or the seed is ignored"
  );
  assert_eq!(
    first,
    p.price_call(S, K, R, 0.0, TAU),
    "the same instance must not advance its own seed between calls"
  );
}

/// The bare name is the `Unseeded` pricer, which stays `Copy`; `Deterministic` holds an
/// `AtomicU64`, so the `Copy` derive is bounded rather than dropped.
#[test]
fn the_default_seed_strategy_is_unseeded() {
  const fn assert_same(_: GbmMalliavinPricer<Unseeded>) {}
  const fn assert_copy<T: Copy>() {}
  assert_copy::<GbmMalliavinPricer>();
  let bare: GbmMalliavinPricer = GbmMalliavinPricer::new(0.2, 8, 4, 0.5);
  assert_same(bare);
  assert_eq!(bare.n_paths, 8);
}

/// `n_paths = 0` and `n_steps = 1` would price a fabricated `0.0`, so `new` rejects them; `t_eval`
/// is also read against each query's `tau`, which answers NaN.
mod construction_validation {
  use super::*;

  #[test]
  #[should_panic(expected = "n_paths must satisfy `n_paths >= 1`, got n_paths = 0")]
  fn new_rejects_zero_paths() {
    let _ = GbmMalliavinPricer::new(0.2, 0, 100, 0.5);
  }

  #[test]
  #[should_panic(expected = "n_steps must satisfy `n_steps >= 2`, got n_steps = 1")]
  fn new_rejects_a_single_time_step() {
    let _ = GbmMalliavinPricer::new(0.2, 200, 1, 0.5);
  }

  #[test]
  #[should_panic(expected = "n_steps must satisfy `n_steps >= 2`, got n_steps = 0")]
  fn new_rejects_zero_time_steps() {
    let _ = GbmMalliavinPricer::new(0.2, 200, 0, 0.5);
  }

  /// The weight carries `+ σ·t_eval` linearly, so a negative volatility biases it unsquared.
  #[test]
  #[should_panic(
    expected = "sigma must satisfy `sigma.is_finite() && sigma >= 0.0`, got sigma = -0.2"
  )]
  fn new_rejects_negative_volatility() {
    let _ = GbmMalliavinPricer::new(-0.2, 200, 100, 0.5);
  }

  #[test]
  #[should_panic(
    expected = "sigma must satisfy `sigma.is_finite() && sigma >= 0.0`, got sigma = inf"
  )]
  fn new_rejects_an_infinite_volatility() {
    let _ = GbmMalliavinPricer::new(f64::INFINITY, 200, 100, 0.5);
  }

  #[test]
  #[should_panic(
    expected = "t_eval must satisfy `t_eval.is_finite() && t_eval > 0.0`, got t_eval = -0.5"
  )]
  fn new_rejects_non_positive_t_eval() {
    let _ = GbmMalliavinPricer::new(0.2, 200, 100, -0.5);
  }

  #[test]
  #[should_panic(
    expected = "t_eval must satisfy `t_eval.is_finite() && t_eval > 0.0`, got t_eval = inf"
  )]
  fn new_rejects_an_infinite_t_eval() {
    let _ = GbmMalliavinPricer::new(0.2, 200, 100, f64::INFINITY);
  }

  /// `t_eval < tau` depends on the query, and a query never panics.
  #[test]
  fn the_query_dependent_half_is_nan_at_the_accessor() {
    let model = GbmMalliavinPricer::new(0.2, 200, 64, 2.0);
    assert!(model.price_call(S, K, R, 0.0, TAU).is_nan());
  }

  #[test]
  fn the_smallest_usable_grid_stays_constructible() {
    let m = GbmMalliavinPricer::new(0.0, 1, 2, 1e-12);
    assert_eq!((m.n_paths, m.n_steps), (1, 2));
  }

  /// `with_seed` takes a type, so the numeric guards still fire on a seeded pricer.
  #[test]
  #[should_panic(expected = "n_paths must satisfy `n_paths >= 1`, got n_paths = 0")]
  fn the_numeric_guards_still_fire_with_a_pinned_seed() {
    let _ = GbmMalliavinPricer::new(0.2, 0, 100, 0.5).with_seed(Deterministic::new(SEEDS[0]));
  }
}

/// A query no path is simulated for answers its payoff or NaN on all four accessors.
mod degenerate_queries {
  use super::*;

  fn model() -> GbmMalliavinPricer<Deterministic> {
    GbmMalliavinPricer::new(0.2, 64, 16, 0.5).with_seed(Deterministic::new(SEEDS[0]))
  }

  #[test]
  fn a_spot_or_strike_that_is_not_positive_is_nan() {
    let p = model();
    for (s, k) in [
      (0.0, K),
      (S, 0.0),
      (S, -5.0),
      (S, f64::NAN),
      (S, f64::INFINITY),
    ] {
      let CallPut { call, put } = p.price_call_put(s, k, R, 0.0, TAU);
      assert!(call.is_nan() && put.is_nan(), "({s}, {k}): ({call}, {put})");
      let localized = p.price_call_put_localized(s, k, R, 0.0, TAU);
      assert!(
        localized.call.is_nan() && localized.put.is_nan(),
        "({s}, {k})"
      );
      let (s_t, c_t) = p.conditional_call_malliavin(s, k, R, 0.0, TAU);
      assert!(s_t.iter().chain(&c_t).all(|x| x.is_nan()), "({s}, {k})");
    }
    assert_eq!(p.price_put(0.0, K, R, 0.0, 0.0), K);
  }

  #[test]
  fn a_zero_horizon_prices_at_the_intrinsic_value() {
    let p = model();
    assert_eq!(
      p.price_call_put(110.0, 100.0, R, 0.0, 0.0),
      CallPut {
        call: 10.0,
        put: 0.0
      }
    );
    assert_eq!(
      p.price_call_put_localized(90.0, 100.0, R, 0.0, 0.0),
      CallPut {
        call: 0.0,
        put: 10.0
      }
    );
    let (s_t, c_t) = p.conditional_call_malliavin_localized(110.0, 100.0, R, 0.0, 0.0);
    assert!(s_t.iter().all(|x| *x == 110.0) && c_t.iter().all(|x| *x == 10.0));
  }

  #[test]
  fn an_off_domain_query_is_nan() {
    let p = model();
    let queries = [
      (f64::NAN, R, 0.0, TAU),
      (-1.0, R, 0.0, TAU),
      (S, f64::NAN, 0.0, TAU),
      (S, R, f64::INFINITY, TAU),
      (S, R, 0.0, -1.0),
      (S, R, 0.0, f64::NAN),
    ];
    for (s, r, q, tau) in queries {
      let CallPut { call, put } = p.price_call_put(s, K, r, q, tau);
      let (s_t, c_t) = p.conditional_call_malliavin(s, K, r, q, tau);
      assert!(
        call.is_nan() && put.is_nan(),
        "({s}, {r}, {q}, {tau}): ({call}, {put})"
      );
      assert!(
        s_t.iter().chain(&c_t).all(|x| x.is_nan()),
        "({s}, {r}, {q}, {tau})"
      );
    }
  }
}
