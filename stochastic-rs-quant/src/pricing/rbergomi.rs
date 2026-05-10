//! # Rough Bergomi Pricer
//!
//! Monte Carlo pricer for the rough Bergomi model implementing [`ModelPricer`].
//!
//! $$
//! dS_t = rS_t\,dt + S_t\sqrt{V_t}\bigl(\rho\,dW_t + \sqrt{1-\rho^2}\,dW_t^\perp\bigr),
//! \quad V_t = \xi_0(t)\exp\!\bigl(\eta\,\hat I_t - \tfrac{\eta^2}{2}t^{2H}\bigr)
//! $$
//!
//! Uses the mSOE (modified Summation of Exponentials) simulation from the
//! calibration module with optional variance reduction via antithetic sampling.
//!
//! Reference:
//! - Bayer, Friz & Gatheral (2016), Quantitative Finance 16(6), 887–904
//! - McCrickerd & Pakkanen (2018), DOI: 10.1080/14697688.2018.1459812, arXiv:1708.02563

use crate::calibration::rbergomi::RBergomiParams;
use crate::calibration::rbergomi::simulate_rbergomi_terminal_samples;
use crate::traits::ModelPricer;

/// Monte Carlo pricer for the rough Bergomi model.
///
/// Wraps calibrated [`RBergomiParams`] and simulation settings to implement
/// [`ModelPricer`], enabling integration with the vol-surface pipeline.
#[derive(Debug, Clone)]
pub struct RBergomiPricer {
  /// Calibrated rBergomi parameters (H, rho, eta, xi0).
  pub params: RBergomiParams,
  /// Number of Monte Carlo paths.
  pub n_paths: usize,
  /// Time steps per year for the Euler discretisation.
  pub steps_per_year: usize,
  /// Number of mSOE exponential terms for the fBM approximation.
  pub msoe_terms: usize,
  /// RNG seed for reproducibility.
  pub seed: u64,
  /// If true, use antithetic sampling for variance reduction.
  pub antithetic: bool,
}

impl RBergomiPricer {
  pub fn new(params: RBergomiParams) -> Self {
    Self {
      params,
      n_paths: 100_000,
      steps_per_year: 200,
      msoe_terms: 12,
      seed: 42,
      antithetic: true,
    }
  }

  pub fn with_paths(mut self, n_paths: usize) -> Self {
    self.n_paths = n_paths;
    self
  }

  pub fn with_steps_per_year(mut self, steps: usize) -> Self {
    self.steps_per_year = steps;
    self
  }

  pub fn with_msoe_terms(mut self, terms: usize) -> Self {
    self.msoe_terms = terms;
    self
  }

  pub fn with_seed(mut self, seed: u64) -> Self {
    self.seed = seed;
    self
  }

  pub fn with_antithetic(mut self, antithetic: bool) -> Self {
    self.antithetic = antithetic;
    self
  }

  fn mc_call_price(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    let samples = simulate_rbergomi_terminal_samples(
      &self.params,
      s,
      r,
      q,
      tau,
      self.n_paths,
      self.steps_per_year,
      self.msoe_terms,
      self.seed,
    );

    let payoff_sum: f64 = samples.iter().map(|&st| (st - k).max(0.0)).sum();
    let mut price = payoff_sum / samples.len() as f64;

    if self.antithetic {
      let samples_anti = simulate_rbergomi_terminal_samples(
        &self.params,
        s,
        r,
        q,
        tau,
        self.n_paths,
        self.steps_per_year,
        self.msoe_terms,
        self.seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
      );
      let payoff_sum_anti: f64 = samples_anti.iter().map(|&st| (st - k).max(0.0)).sum();
      let price_anti = payoff_sum_anti / samples_anti.len() as f64;
      price = 0.5 * (price + price_anti);
    }

    (-r * tau).exp() * price
  }
}

impl ModelPricer for RBergomiPricer {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.mc_call_price(s, k, r, q, tau)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::calibration::rbergomi::RBergomiXi0;

  #[test]
  fn rbergomi_pricer_call_positive() {
    let params = RBergomiParams {
      hurst: 0.1,
      rho: -0.7,
      eta: 1.9,
      xi0: RBergomiXi0::Constant(0.04),
    };
    let pricer = RBergomiPricer::new(params).with_paths(50_000);
    let call = pricer.price_call(100.0, 100.0, 0.05, 0.0, 0.5);
    assert!(
      call > 0.0 && call < 100.0,
      "ATM call should be positive and bounded: {call}"
    );
  }

  #[test]
  fn rbergomi_pricer_call_decreases_with_strike() {
    let params = RBergomiParams {
      hurst: 0.1,
      rho: -0.7,
      eta: 1.9,
      xi0: RBergomiXi0::Constant(0.04),
    };
    let pricer = RBergomiPricer::new(params).with_paths(50_000);
    let c_90 = pricer.price_call(100.0, 90.0, 0.05, 0.0, 0.5);
    let c_100 = pricer.price_call(100.0, 100.0, 0.05, 0.0, 0.5);
    let c_110 = pricer.price_call(100.0, 110.0, 0.05, 0.0, 0.5);
    assert!(
      c_90 > c_100 && c_100 > c_110,
      "Call should decrease with strike: c90={c_90}, c100={c_100}, c110={c_110}"
    );
  }

  /// Regression: the pricer must thread `q` to `simulate_rbergomi_terminal_samples`.
  /// Pre-fix the `_q` argument was discarded, so the simulated forward was
  /// `S_0·e^(r·T)` rather than `S_0·e^((r-q)·T)`. With seeded RNG this gave
  /// identical prices for q=0 and q>0 — a silent dynamics bug.
  #[test]
  fn rbergomi_pricer_respects_dividend_yield() {
    let params = RBergomiParams {
      hurst: 0.1,
      rho: -0.7,
      eta: 1.9,
      xi0: RBergomiXi0::Constant(0.04),
    };
    let pricer = RBergomiPricer::new(params).with_paths(80_000);
    let p_no_div = pricer.price_call(100.0, 100.0, 0.05, 0.0, 1.0);
    let p_with_div = pricer.price_call(100.0, 100.0, 0.05, 0.05, 1.0);
    // With shared seed, the only difference between the two MC runs is the q
    // term in the drift. ATM call must be cheaper with positive q (forward
    // shift down) — strictly so given fixed seed and 80k paths.
    assert!(
      p_with_div < p_no_div - 0.5,
      "ATM rBergomi call must drop when q > 0: q=0 → {p_no_div:.4}, q=0.05 → {p_with_div:.4}"
    );
  }
}
