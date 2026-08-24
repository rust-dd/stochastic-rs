//! # Snell Envelope (American Options)
//!
//! Discrete-time Snell envelope recursion on a CRR binomial tree:
//! $$
//! Y_N = g(S_N),\qquad
//! Y_i = \max\left(g(S_i), e^{-r\Delta t}\mathbb{E}^{\mathbb{Q}}[Y_{i+1}\mid\mathcal{F}_i]\right).
//! $$
//!
//! With two-state transition each step:
//! $$
//! \mathbb{E}^{\mathbb{Q}}[Y_{i+1}\mid\mathcal{F}_i]
//! = pY_{i+1}^{u} + (1-p)Y_{i+1}^{d},
//! $$
//! where, in the CRR tree,
//! $$
//! u=e^{\sigma\sqrt{\Delta t}},\quad d=u^{-1},\quad
//! p=\frac{e^{(r-q)\Delta t}-d}{u-d}.
//! $$
//!
//! Source:
//! - Snell envelope / optimal stopping foundation
//! - Cox-Ross-Rubinstein binomial tree discretization

use crate::OptionType;
use crate::traits::ModelPricer;

#[derive(Debug, Clone)]
pub struct SnellEnvelopeResult {
  pub price: f64,
  pub european_price: f64,
  pub early_exercise_premium: f64,
  /// Exercise boundary as `(time_in_years, critical_stock_price)` pairs.
  pub exercise_boundary: Vec<(f64, f64)>,
}

/// American option priced by Snell-envelope recursion on a CRR tree.
///
/// The struct holds **model and method state only** — the volatility and
/// the number of binomial time steps. Spot, strike, rate, dividend yield,
/// maturity and the option direction are the pricing *query* and travel as
/// arguments to [`ModelPricer::price_call`], so one instance prices a whole
/// strike/maturity grid. The tree itself is rebuilt per call from the
/// query, so nothing derived from a spot or a maturity is cached across
/// queries.
///
/// ```
/// use stochastic_rs_quant::pricing::snell_envelope::SnellEnvelopePricer;
/// use stochastic_rs_quant::traits::ModelPricer;
///
/// let model = SnellEnvelopePricer::new(0.2, 200);
/// let american = model.price_put(100.0, 100.0, 0.03, 0.01, 1.0);
/// assert!(american > 0.0);
/// ```
///
/// # Panics
/// [`price_call`](ModelPricer::price_call), [`price_put`], and
/// [`price_detailed`](Self::price_detailed) panic on a non-positive or
/// non-finite spot / strike / rate / dividend yield / maturity, and on a
/// risk-neutral probability outside `[0, 1]` (raise `steps`). These are the
/// same assertions, with the same messages, that the pre-query
/// `SnellEnvelopePricer::new` made at construction time; they moved to the
/// call because that is where those values now arrive.
#[derive(Debug, Clone, Copy)]
pub struct SnellEnvelopePricer {
  /// Volatility $\sigma$.
  pub v: f64,
  /// Number of binomial time steps.
  pub steps: usize,
}

impl SnellEnvelopePricer {
  pub fn new(v: f64, steps: usize) -> Self {
    assert!(v.is_finite() && v > 0.0, "v must be finite and positive");
    assert!(steps > 0, "steps must be > 0");

    Self { v, steps }
  }

  /// Assert the query is a well-posed pricing point, with the messages the
  /// pre-query constructor used.
  fn validate_query(s: f64, k: f64, r: f64, q: f64, tau: f64) {
    assert!(s.is_finite() && s > 0.0, "s must be finite and positive");
    assert!(k.is_finite() && k > 0.0, "k must be finite and positive");
    assert!(r.is_finite(), "r must be finite");
    assert!(q.is_finite(), "q must be finite");
    assert!(tau.is_finite() && tau > 0.0, "tau must be positive");
  }

  /// CRR lattice constants `(dt, u, d, disc, p)` at one query point.
  fn lattice(&self, r: f64, q: f64, tau: f64) -> (f64, f64, f64, f64, f64) {
    let dt = tau / self.steps as f64;
    let sqrt_dt = dt.sqrt();
    let u = (self.v * sqrt_dt).exp();
    let d = 1.0 / u;
    let disc = (-r * dt).exp();
    let growth = ((r - q) * dt).exp();
    let p = (growth - d) / (u - d);
    assert!(
      (0.0..=1.0).contains(&p),
      "risk-neutral probability out of range: p={p}. Increase steps or adjust parameters."
    );
    (dt, u, d, disc, p)
  }

  fn price_american(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> f64 {
    Self::validate_query(s, k, r, q, tau);
    let (_dt, u, d, disc, p) = self.lattice(r, q, tau);

    let mut values = vec![0.0_f64; self.steps + 1];
    let mut s_node = s * d.powi(self.steps as i32);
    let ud_ratio = u / d;
    for val in values.iter_mut().take(self.steps + 1) {
      *val = payoff(option_type, s_node, k);
      s_node *= ud_ratio;
    }

    for i in (0..self.steps).rev() {
      let mut s_i0 = s * d.powi(i as i32);
      for j in 0..=i {
        let continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
        let exercise = payoff(option_type, s_i0, k);
        values[j] = continuation.max(exercise);
        s_i0 *= ud_ratio;
      }
    }

    values[0]
  }

  /// American price plus the European price on the same tree, their
  /// difference, and the early-exercise boundary.
  pub fn price_detailed(
    &self,
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    tau: f64,
    option_type: OptionType,
  ) -> SnellEnvelopeResult {
    Self::validate_query(s, k, r, q, tau);
    let (dt, u, d, disc, p) = self.lattice(r, q, tau);
    let ud_ratio = u / d;

    let mut am_values = vec![0.0_f64; self.steps + 1];
    let mut eu_values = vec![0.0_f64; self.steps + 1];
    let mut s_node = s * d.powi(self.steps as i32);
    for idx in 0..=self.steps {
      let pv = payoff(option_type, s_node, k);
      am_values[idx] = pv;
      eu_values[idx] = pv;
      s_node *= ud_ratio;
    }

    let mut exercise_boundary = Vec::new();

    for i in (0..self.steps).rev() {
      let mut s_i0 = s * d.powi(i as i32);
      let mut boundary_s = f64::NAN;

      for j in 0..=i {
        let am_cont = disc * (p * am_values[j + 1] + (1.0 - p) * am_values[j]);
        let eu_cont = disc * (p * eu_values[j + 1] + (1.0 - p) * eu_values[j]);
        let exercise = payoff(option_type, s_i0, k);

        am_values[j] = am_cont.max(exercise);
        eu_values[j] = eu_cont;

        if exercise > am_cont + 1e-12 {
          match option_type {
            OptionType::Put => {
              if boundary_s.is_nan() || s_i0 > boundary_s {
                boundary_s = s_i0;
              }
            }
            OptionType::Call => {
              if boundary_s.is_nan() || s_i0 < boundary_s {
                boundary_s = s_i0;
              }
            }
          }
        }

        s_i0 *= ud_ratio;
      }

      if boundary_s.is_finite() {
        exercise_boundary.push(((i as f64) * dt, boundary_s));
      }
    }

    // boundary times come from `(i as f64) * dt` with `i, dt` finite, so NaN
    // is unreachable here; fall back to Equal in case future refactors plumb
    // user data through this path.
    exercise_boundary.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    SnellEnvelopeResult {
      price: am_values[0],
      european_price: eu_values[0],
      early_exercise_premium: am_values[0] - eu_values[0],
      exercise_boundary,
    }
  }
}

impl ModelPricer for SnellEnvelopePricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_american(s, k, r, q, tau, OptionType::Call)
  }

  /// Overrides the trait's vanilla-parity default. European put-call parity
  /// does not hold for an American option: the put carries an
  /// early-exercise premium the call does not, so the default would
  /// understate it. This re-runs the same Snell recursion with the put
  /// payoff, which is what the pre-query `calculate_call_put().1` did — see
  /// `snell_price_put_overrides_vanilla_parity`.
  fn price_put(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_american(s, k, r, q, tau, OptionType::Put)
  }
}

fn payoff(option_type: OptionType, s: f64, k: f64) -> f64 {
  match option_type {
    OptionType::Call => (s - k).max(0.0),
    OptionType::Put => (k - s).max(0.0),
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::pricing::bsm::BSMCoc;
  use crate::pricing::bsm::BSMPricer;

  #[test]
  fn american_put_is_at_least_european_put() {
    let amer = SnellEnvelopePricer::new(0.2, 800).price_put(100.0, 100.0, 0.03, 0.01, 1.0);
    let euro = BSMPricer::new(0.2, BSMCoc::Merton1973).price_put(100.0, 100.0, 0.03, 0.01, 1.0);

    assert!(amer + 1e-10 >= euro);
  }

  #[test]
  fn american_call_matches_european_without_dividend() {
    let amer = SnellEnvelopePricer::new(0.2, 1200).price_call(100.0, 100.0, 0.05, 0.0, 1.0);
    let euro = BSMPricer::new(0.2, BSMCoc::Merton1973).price_call(100.0, 100.0, 0.05, 0.0, 1.0);

    assert!((amer - euro).abs() < 5e-2);
  }

  #[test]
  fn price_detailed_returns_exercise_boundary_and_premium() {
    let pricer = SnellEnvelopePricer::new(0.2, 800);
    let result = pricer.price_detailed(100.0, 100.0, 0.03, 0.01, 1.0, OptionType::Put);

    assert!(result.price > 0.0);
    assert!(result.european_price > 0.0);
    assert!(result.early_exercise_premium >= -1e-10);
    assert!(result.price >= result.european_price - 1e-10);
    assert!(!result.exercise_boundary.is_empty());

    for &(t, s_star) in &result.exercise_boundary {
      assert!((0.0..1.0).contains(&t));
      assert!(s_star > 0.0 && s_star <= 100.0);
    }

    let times: Vec<f64> = result.exercise_boundary.iter().map(|p| p.0).collect();
    for w in times.windows(2) {
      assert!(w[0] <= w[1]);
    }
  }

  #[test]
  fn american_call_can_exceed_european_with_dividend() {
    let amer = SnellEnvelopePricer::new(0.25, 1000).price_call(100.0, 90.0, 0.03, 0.08, 1.0);
    let euro = BSMPricer::new(0.25, BSMCoc::Merton1973).price_call(100.0, 90.0, 0.03, 0.08, 1.0);

    assert!(amer + 1e-10 >= euro);
  }

  const S: f64 = 100.0;
  const K: f64 = 105.0;
  const R: f64 = 0.05;
  const Q: f64 = 0.02;
  const TAU: f64 = 0.75;
  const V: f64 = 0.25;
  const STEPS: usize = 200;

  /// Cross-arch tolerance: 200 tree steps of `exp`/`max` accumulate a last
  /// bit that differs between aarch64-darwin and CI's ubuntu x86_64.
  const TOL: f64 = 1e-12;

  /// Captured from `PricerExt::calculate_call_put()` and
  /// `price_detailed(Put)` **before** the `ModelPricer` reshape. The
  /// reshape is an API change only.
  #[test]
  fn snell_model_pricer_matches_pre_refactor_goldens() {
    let model = SnellEnvelopePricer::new(V, STEPS);
    let call = model.price_call(S, K, R, Q, TAU);
    let put = model.price_put(S, K, R, Q, TAU);
    assert!((call - 7.365701683033536).abs() < TOL, "call {call}");
    assert!((put - 10.353752943250777).abs() < TOL, "put {put}");

    let d = model.price_detailed(S, K, R, Q, TAU, OptionType::Put);
    assert!((d.price - 10.353752943250777).abs() < TOL, "{}", d.price);
    assert!(
      (d.european_price - 9.98992154928077).abs() < TOL,
      "{}",
      d.european_price
    );
    assert!(
      (d.early_exercise_premium - 0.36383139397000797).abs() < TOL,
      "{}",
      d.early_exercise_premium
    );
    assert_eq!(d.exercise_boundary.len(), 184);
  }

  /// American puts carry an early-exercise premium the call does not, so
  /// the trait's European-parity default is the wrong model here.
  #[test]
  fn snell_price_put_overrides_vanilla_parity() {
    let model = SnellEnvelopePricer::new(V, STEPS);
    let call = model.price_call(S, K, R, Q, TAU);
    let put = model.price_put(S, K, R, Q, TAU);
    let vanilla = call - S * (-Q * TAU).exp() + K * (-R * TAU).exp();
    assert!(
      put > vanilla + 1e-3,
      "American put must exceed the European-parity value: {put} vs {vanilla}"
    );
  }

  /// The capability the reshape exists for: one model, a whole grid.
  #[test]
  fn snell_one_model_prices_a_grid() {
    let model = SnellEnvelopePricer::new(V, 100);
    for &tau in &[0.25, 0.5, 1.0] {
      let mut prev = f64::INFINITY;
      for &k in &[90.0, 100.0, 110.0] {
        let c = model.price_call(S, k, R, Q, tau);
        assert!(c.is_finite() && c < prev, "call must fall in strike");
        prev = c;
      }
    }
  }

  /// The query-time assertions the pre-query constructor used to make still
  /// fire, with the same messages.
  #[test]
  #[should_panic(expected = "s must be finite and positive")]
  fn snell_rejects_a_nonpositive_spot() {
    let _ = SnellEnvelopePricer::new(V, 10).price_call(-1.0, K, R, Q, TAU);
  }

  #[test]
  #[should_panic(expected = "tau must be positive")]
  fn snell_rejects_a_nonpositive_tau() {
    let _ = SnellEnvelopePricer::new(V, 10).price_call(S, K, R, Q, 0.0);
  }
}
