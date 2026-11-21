//! Monte Carlo pricing of the rough Bergomi model on the calibration module's mSOE simulation,
//! antithetic pairs optional.
//!
//! $$
//! dS_t = rS_t\,dt + S_t\sqrt{V_t}\bigl(\rho\,dW_t + \sqrt{1-\rho^2}\,dW_t^\perp\bigr),
//! \quad V_t = \xi_0(t)\exp\!\bigl(\eta\,\hat I_t - \tfrac{\eta^2}{2}t^{2H}\bigr)
//! $$
//!
//! Reference: Bayer, Friz & Gatheral (2016), Quant. Finance 16(6), 887–904
//! Reference: McCrickerd & Pakkanen (2018), DOI 10.1080/14697688.2018.1459812, arXiv:1708.02563

use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::mc::McEstimate;

use crate::OptionType;
use crate::calibration::rbergomi::RBergomiParams;
use crate::calibration::rbergomi::simulate_rbergomi_antithetic_pairs;
use crate::calibration::rbergomi::simulate_rbergomi_terminal_samples;
use crate::pricing::digital::expired;
use crate::pricing::digital::vanilla_payoff;
use crate::pricing::mc_stats::pathless;
use crate::pricing::mc_stats::std_err_from_sums;
use crate::traits::ModelPricer;
use crate::traits::RealExt;
use crate::traits::VanillaEuropeanCall;

/// Monte Carlo [`ModelPricer`] of calibrated [`RBergomiParams`], so the model reaches the
/// vol-surface pipeline.
#[derive(Debug, Clone)]
pub struct RBergomiPricer<S: SeedExt = Unseeded> {
  /// Calibrated rBergomi parameters (H, rho, eta, xi0).
  pub params: RBergomiParams,
  /// Number of Monte Carlo paths.
  pub n_paths: usize,
  /// Time steps per year for the Euler discretisation.
  pub steps_per_year: usize,
  /// Number of mSOE exponential terms for the fBM approximation.
  pub msoe_terms: usize,
  /// Each call draws from a clone, so a `Deterministic` seed replays.
  pub seed: S,
  /// Runs each path a second time on its normals negated, `2 n_paths` paths in all; the default.
  pub antithetic: bool,
}

impl RBergomiPricer {
  pub fn new(params: RBergomiParams) -> Self {
    Self {
      params,
      n_paths: 100_000,
      steps_per_year: 200,
      msoe_terms: 12,
      seed: Unseeded,
      antithetic: true,
    }
  }
}

impl<S: SeedExt> RBergomiPricer<S> {
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

  pub fn with_seed<S2: SeedExt>(self, seed: S2) -> RBergomiPricer<S2> {
    RBergomiPricer {
      params: self.params,
      n_paths: self.n_paths,
      steps_per_year: self.steps_per_year,
      msoe_terms: self.msoe_terms,
      seed,
      antithetic: self.antithetic,
    }
  }

  pub fn with_antithetic(mut self, antithetic: bool) -> Self {
    self.antithetic = antithetic;
    self
  }

  /// Call estimate, its error bar over the pair means under antithetic sampling; the payoff on no
  /// path at `tau = 0`, NaN past expiry, for a NaN query or a spot or strike not positive.
  pub fn price_call_estimate(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> McEstimate<f64> {
    if s.is_nan() || k.is_nan() {
      return pathless(f64::NAN);
    }
    if let Some(payoff) = expired(tau, &[r, q], || vanilla_payoff(OptionType::Call, s, k)) {
      return pathless(payoff);
    }
    if !(s > 0.0 && k > 0.0 && (r - q).is_finite() && tau.is_finite()) {
      return pathless(f64::NAN);
    }
    let seed = self.seed.clone();
    let discount = (-r * tau).exp();
    let call = |st: f64| (st - k).max_or_nan(0.0);
    let (mut sum, mut sum_sq) = (0.0_f64, 0.0_f64);
    let mut add = |y: f64| {
      sum += y;
      sum_sq += y * y;
    };
    let (paths, steps, terms) = (self.n_paths, self.steps_per_year, self.msoe_terms);
    if self.antithetic {
      let query = (s, r, q, tau);
      let pairs =
        simulate_rbergomi_antithetic_pairs(&self.params, query, paths, steps, terms, &seed);
      pairs
        .iter()
        .for_each(|&[up, down]| add(0.5 * (call(up) + call(down))));
    } else {
      let samples =
        simulate_rbergomi_terminal_samples(&self.params, s, r, q, tau, paths, steps, terms, &seed);
      samples.iter().for_each(|&st| add(call(st)));
    }
    McEstimate {
      mean: discount * (sum / paths as f64),
      std_err: discount * std_err_from_sums(sum, sum_sq, paths),
      n_samples: paths,
    }
  }
}

impl<S: SeedExt> ModelPricer for RBergomiPricer<S> {
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_call_estimate(s, k, r, q, tau).mean
  }
}

/// European vanilla call: the Monte Carlo averages $(S_T-K)^+$ over
/// terminal samples drifted at $r-q$, so the default forward applies.
impl<S: SeedExt> VanillaEuropeanCall for RBergomiPricer<S> {}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;

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
    let pricer = RBergomiPricer::new(params)
      .with_paths(50_000)
      .with_seed(Deterministic::new(42));
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
    let pricer = RBergomiPricer::new(params)
      .with_paths(50_000)
      .with_seed(Deterministic::new(42));
    let c_90 = pricer.price_call(100.0, 90.0, 0.05, 0.0, 0.5);
    let c_100 = pricer.price_call(100.0, 100.0, 0.05, 0.0, 0.5);
    let c_110 = pricer.price_call(100.0, 110.0, 0.05, 0.0, 0.5);
    assert!(
      c_90 > c_100 && c_100 > c_110,
      "Call should decrease with strike: c90={c_90}, c100={c_100}, c110={c_110}"
    );
  }

  /// The pricer passes `q` to the simulation, so the forward is `S_0 e^{(r−q)T}`.
  #[test]
  fn rbergomi_pricer_respects_dividend_yield() {
    let params = RBergomiParams {
      hurst: 0.1,
      rho: -0.7,
      eta: 1.9,
      xi0: RBergomiXi0::Constant(0.04),
    };
    let pricer = RBergomiPricer::new(params)
      .with_paths(80_000)
      .with_seed(Deterministic::new(42));
    let p_no_div = pricer.price_call(100.0, 100.0, 0.05, 0.0, 1.0);
    let p_with_div = pricer.price_call(100.0, 100.0, 0.05, 0.05, 1.0);
    // On one seed only `q` moves the drift, so the forward drops and the call with it.
    assert!(
      p_with_div < p_no_div - 0.5,
      "ATM rBergomi call must drop when q > 0: q=0 → {p_no_div:.4}, q=0.05 → {p_with_div:.4}"
    );
  }
}
