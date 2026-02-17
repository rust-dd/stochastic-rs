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

use crate::quant::OptionType;
use crate::traits::PricerExt;
use crate::traits::TimeExt;

pub struct SnellEnvelopePricer {
  /// Spot level $S_0$.
  pub s: f64,
  /// Volatility $\sigma$.
  pub v: f64,
  /// Strike $K$.
  pub k: f64,
  /// Risk-free rate $r$.
  pub r: f64,
  /// Continuous dividend yield $q$.
  pub q: Option<f64>,
  /// Number of binomial time steps.
  pub steps: usize,
  /// Time-to-maturity in years.
  pub tau: Option<f64>,
  /// Evaluation date (optional if `tau` is set).
  pub eval: Option<chrono::NaiveDate>,
  /// Expiration date (optional if `tau` is set).
  pub expiration: Option<chrono::NaiveDate>,
  /// Option direction.
  pub option_type: OptionType,
}

impl SnellEnvelopePricer {
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    s: f64,
    v: f64,
    k: f64,
    r: f64,
    q: Option<f64>,
    steps: usize,
    tau: Option<f64>,
    eval: Option<chrono::NaiveDate>,
    expiration: Option<chrono::NaiveDate>,
    option_type: OptionType,
  ) -> Self {
    assert!(s.is_finite() && s > 0.0, "s must be finite and positive");
    assert!(v.is_finite() && v > 0.0, "v must be finite and positive");
    assert!(k.is_finite() && k > 0.0, "k must be finite and positive");
    assert!(r.is_finite(), "r must be finite");
    if let Some(q) = q {
      assert!(q.is_finite(), "q must be finite");
    }
    assert!(steps > 0, "steps must be > 0");

    Self {
      s,
      v,
      k,
      r,
      q,
      steps,
      tau,
      eval,
      expiration,
      option_type,
    }
  }

  fn price_american(&self, option_type: OptionType) -> f64 {
    let tau = self.tau_or_from_dates();
    assert!(tau.is_finite() && tau > 0.0, "tau must be positive");

    let dt = tau / self.steps as f64;
    let sqrt_dt = dt.sqrt();
    let u = (self.v * sqrt_dt).exp();
    let d = 1.0 / u;
    let disc = (-self.r * dt).exp();
    let growth = ((self.r - self.q.unwrap_or(0.0)) * dt).exp();
    let p = (growth - d) / (u - d);
    assert!(
      (0.0..=1.0).contains(&p),
      "risk-neutral probability out of range: p={p}. Increase steps or adjust parameters."
    );

    let mut values = vec![0.0_f64; self.steps + 1];
    let mut s_node = self.s * d.powi(self.steps as i32);
    let ud_ratio = u / d;
    for val in values.iter_mut().take(self.steps + 1) {
      *val = payoff(option_type, s_node, self.k);
      s_node *= ud_ratio;
    }

    for i in (0..self.steps).rev() {
      let mut s_i0 = self.s * d.powi(i as i32);
      for j in 0..=i {
        let continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
        let exercise = payoff(option_type, s_i0, self.k);
        values[j] = continuation.max(exercise);
        s_i0 *= ud_ratio;
      }
    }

    values[0]
  }
}

impl PricerExt for SnellEnvelopePricer {
  fn calculate_call_put(&self) -> (f64, f64) {
    (
      self.price_american(OptionType::Call),
      self.price_american(OptionType::Put),
    )
  }

  fn calculate_price(&self) -> f64 {
    self.price_american(self.option_type)
  }
}

impl TimeExt for SnellEnvelopePricer {
  fn tau(&self) -> Option<f64> {
    self.tau
  }

  fn eval(&self) -> Option<chrono::NaiveDate> {
    self.eval
  }

  fn expiration(&self) -> Option<chrono::NaiveDate> {
    self.expiration
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
  use crate::quant::pricing::bsm::BSMCoc;
  use crate::quant::pricing::bsm::BSMPricer;

  #[test]
  fn american_put_is_at_least_european_put() {
    let amer = SnellEnvelopePricer::new(
      100.0,
      0.2,
      100.0,
      0.03,
      Some(0.01),
      800,
      Some(1.0),
      None,
      None,
      OptionType::Put,
    )
    .calculate_price();

    let euro = BSMPricer::new(
      100.0,
      0.2,
      100.0,
      0.03,
      None,
      None,
      Some(0.01),
      Some(1.0),
      None,
      None,
      OptionType::Put,
      BSMCoc::Merton1973,
    )
    .calculate_price();

    assert!(amer + 1e-10 >= euro);
  }

  #[test]
  fn american_call_matches_european_without_dividend() {
    let amer = SnellEnvelopePricer::new(
      100.0,
      0.2,
      100.0,
      0.05,
      Some(0.0),
      1200,
      Some(1.0),
      None,
      None,
      OptionType::Call,
    )
    .calculate_price();

    let euro = BSMPricer::new(
      100.0,
      0.2,
      100.0,
      0.05,
      None,
      None,
      Some(0.0),
      Some(1.0),
      None,
      None,
      OptionType::Call,
      BSMCoc::Merton1973,
    )
    .calculate_price();

    assert!((amer - euro).abs() < 5e-2);
  }

  #[test]
  fn american_call_can_exceed_european_with_dividend() {
    let amer = SnellEnvelopePricer::new(
      100.0,
      0.25,
      90.0,
      0.03,
      Some(0.08),
      1000,
      Some(1.0),
      None,
      None,
      OptionType::Call,
    )
    .calculate_price();

    let euro = BSMPricer::new(
      100.0,
      0.25,
      90.0,
      0.03,
      None,
      None,
      Some(0.08),
      Some(1.0),
      None,
      None,
      OptionType::Call,
      BSMCoc::Merton1973,
    )
    .calculate_price();

    assert!(amer + 1e-10 >= euro);
  }
}
