use super::*;

const S: f64 = 100.0;
const K: f64 = 99.99;
const R: f64 = 0.1;
const TAU: f64 = 1.0;

fn pricer() -> GbmMalliavinPricer {
  GbmMalliavinPricer::new(0.1, 2_000, 128, 0.5)
}

#[test]
fn malliavin_pricer_returns_finite_non_negative_prices() {
  let p = pricer();
  let (call, put) = p.call_put(S, K, R, 0.0, TAU);

  // Basic sanity checks: finite and non-negative prices
  assert!(call.is_finite(), "Call price should be finite");
  assert!(put.is_finite(), "Put price should be finite");
  assert!(call >= 0.0, "Call price should be non-negative");
  assert!(put >= 0.0, "Put price should be non-negative");

  // Very loose upper bounds to avoid flakiness due to Monte Carlo noise
  assert!(call < S * 2.0, "Call price is unreasonably large");
  assert!(put < K * 2.0, "Put price is unreasonably large");
}

#[test]
fn malliavin_pricer_localized_returns_finite_non_negative_prices() {
  let p = pricer();
  let (call, put) = p.call_put_localized(S, K, R, 0.0, TAU);

  // Basic sanity checks: finite and non-negative prices
  assert!(call.is_finite(), "Localized call price should be finite");
  assert!(put.is_finite(), "Localized put price should be finite");
  assert!(call >= 0.0, "Localized call price should be non-negative");
  assert!(put >= 0.0, "Localized put price should be non-negative");

  // Very loose upper bounds to avoid flakiness due to Monte Carlo noise
  assert!(call < S * 2.0, "Localized call price is unreasonably large");
  assert!(put < K * 2.0, "Localized put price is unreasonably large");
}

/// `price_call` and `price_put` are the two legs of `call_put`, so the
/// put must satisfy put-call parity **against the call from its own
/// simulation**, not against some other run's. This is what the trait's
/// `price_put` default would break: it would run a second, independent
/// Monte Carlo for its `price_call` term.
#[test]
fn malliavin_put_is_parity_against_its_own_call() {
  let (call, put) = pricer().call_put(S, K, R, 0.02, TAU);
  let parity = call - S * (-0.02_f64 * TAU).exp() + K * (-R * TAU).exp();
  assert!(
    (put - parity.max(0.0)).abs() < 1e-12,
    "put {put} must be the floored parity value {parity} of its own call"
  );
}

/// The `max(0)` floor the pre-query `calculate_call_put` applied — which
/// the trait's `price_put` default does **not** have — still guards the
/// output. It cannot be pinned by a single deterministic value: the floor
/// fires exactly when the Monte Carlo call estimate lands below its
/// parity-implied lower bound, which is an estimator-noise event on an
/// `Unseeded` RNG. What *is* deterministic is the guarantee, so that is
/// what this asserts, across the strike range where a negative
/// parity value is reachable.
#[test]
fn malliavin_put_is_never_negative() {
  let p = pricer();
  for &k in &[1.0, 50.0, 99.99, 150.0] {
    let put = p.price_put(S, k, R, 0.0, TAU);
    assert!(put >= 0.0, "put at K={k} must be floored, got {put}");
  }
}

/// `t_eval` is an absolute time, so a maturity shorter than it is not a
/// query this instance can price — and it says so rather than returning
/// a number.
#[test]
#[should_panic(expected = "t_eval must be in (0, T)")]
fn malliavin_rejects_a_maturity_shorter_than_t_eval() {
  let _ = pricer().price_call(S, K, R, 0.0, 0.25);
}

/// The `l == 0` branch is the Dirac limit, so it has to be the *limit*:
/// the kernel pair must agree with the step function the caller compares
/// it against, or the localisation weight `pdf + (H - cdf)·t2` would come
/// out of a half-evaluated mixture instead of collapsing cleanly to zero.
/// A single mismatched sign here is invisible in a price — it just biases
/// the estimator — which is why it is pinned directly.
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

/// The atom is where the `l == 0` branch stops being a limit and becomes a
/// convention: the symmetric Laplace cdf is `0.5` at `x == 0` for every
/// `l > 0`, and this branch returns `1.0`. Pinned because the choice is
/// load-bearing — it is what cancels the diagonal `j == i` term against the
/// caller's Heaviside instead of leaving half of it behind.
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

/// A negative bandwidth has no limit interpretation, and the old
/// `l <= 0.0` guard folded it in with the well-defined `l == 0`.
#[test]
#[should_panic(expected = "laplace bandwidth l must be non-negative (got -1)")]
fn negative_bandwidth_is_rejected() {
  let _ = laplace_pdf(0.5, -1.0);
}

/// The reachable route to `l == 0`: a strike so far out of the money that
/// every simulated payoff is zero, which zeroes `lf` and empties the
/// localisation sum. The price that comes back is `0.0` because the option
/// is worthless on this sample, not because the kernel vanished — so the
/// guard must be that it is *finite and non-negative*, and that the same
/// pricer still prices a live strike.
#[test]
fn an_all_worthless_sample_empties_the_kernel_without_poisoning_the_price() {
  let p = GbmMalliavinPricer::new(0.1, 200, 32, 0.5);
  let dead = p.call_put_localized(S, 1.0e12, R, 0.0, TAU).0;
  assert!(
    dead.is_finite() && dead >= 0.0,
    "unreachable strike must price to a finite floor, got {dead}"
  );
  let live = p.call_put_localized(S, K, R, 0.0, TAU).0;
  assert!(
    live > dead,
    "live strike {live} must beat dead strike {dead}"
  );
}

/// The capability the reshape exists for: one model, many query points.
/// Monte Carlo noise makes a strict monotonicity assertion flaky, so this
/// pins the weaker property that every point is priced and finite, plus
/// the no-arbitrage upper bound.
#[test]
fn malliavin_one_model_prices_a_grid() {
  let model = GbmMalliavinPricer::new(0.2, 400, 64, 0.25);
  for &tau in &[0.5, 1.0] {
    for &k in &[90.0, 100.0, 110.0] {
      let c = model.price_call(S, k, 0.03, 0.01, tau);
      assert!(
        c.is_finite() && (0.0..=S).contains(&c),
        "call {c} out of bounds"
      );
    }
  }
}

/// `GbmMalliavinPricer::new` validates the estimator's own shape.
///
/// `n_paths` and `n_steps` are simulation *counts*, and both degenerate
/// into a fabricated **0.0** price rather than a complaint: with
/// `n_paths = 0` there is nothing to average, `call_put_from_conditional`
/// takes its `count == 0` branch and returns the discounted average of an
/// empty sample; with `n_steps = 1` the step size `tau / (n_steps - 1)` is
/// infinite and the same zero comes out the other end. `n_steps = 0` is the
/// loud one — `n_steps - 1` underflows — so of the three degenerate counts
/// only one announced itself.
///
/// `t_eval` keeps its accessor guard: the constructor can only check the
/// half of `0 < t_eval < tau` that does not need the query, since `tau`
/// arrives per call. The two messages are deliberately different — neither
/// is a substring of the other — so a test anchored on one cannot be
/// satisfied by the other firing.
mod construction_validation {
  use super::*;

  #[test]
  #[should_panic(expected = "GbmMalliavinPricer::new: n_paths must be at least 1 (got 0)")]
  fn new_rejects_zero_paths() {
    let _ = GbmMalliavinPricer::new(0.2, 0, 100, 0.5);
  }

  #[test]
  #[should_panic(expected = "GbmMalliavinPricer::new: n_steps must be at least 2 (got 1)")]
  fn new_rejects_a_single_time_step() {
    let _ = GbmMalliavinPricer::new(0.2, 200, 1, 0.5);
  }

  #[test]
  #[should_panic(expected = "GbmMalliavinPricer::new: n_steps must be at least 2 (got 0)")]
  fn new_rejects_zero_time_steps() {
    let _ = GbmMalliavinPricer::new(0.2, 200, 0, 0.5);
  }

  /// The Malliavin weight carries `+ σ·t_eval` *linearly*, so unlike the
  /// squared-only uses elsewhere a negative volatility is not absorbed —
  /// it biases the weight and the conditional expectation built on it.
  #[test]
  #[should_panic(
    expected = "GbmMalliavinPricer::new: v must be a non-negative volatility (got -0.2)"
  )]
  fn new_rejects_negative_volatility() {
    let _ = GbmMalliavinPricer::new(-0.2, 200, 100, 0.5);
  }

  #[test]
  #[should_panic(expected = "GbmMalliavinPricer::new: t_eval must be strictly positive (got -0.5)")]
  fn new_rejects_non_positive_t_eval() {
    let _ = GbmMalliavinPricer::new(0.2, 200, 100, -0.5);
  }

  /// The `t_eval < tau` half is unreachable from the constructor and stays
  /// where it was, with its own wording.
  #[test]
  #[should_panic(expected = "t_eval must be in (0, T)")]
  fn the_query_dependent_half_stays_at_the_accessor() {
    let _ = GbmMalliavinPricer::new(0.2, 200, 64, 2.0).price_call(S, K, R, 0.0, TAU);
  }

  #[test]
  fn the_smallest_usable_grid_stays_constructible() {
    let m = GbmMalliavinPricer::new(0.0, 1, 2, 1e-12);
    assert_eq!((m.n_paths, m.n_steps), (1, 2));
  }
}
