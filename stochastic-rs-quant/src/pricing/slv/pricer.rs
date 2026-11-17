//! Monte Carlo pricing under a pre-calibrated leverage surface.

use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::normal::SimdNormal;

use super::HestonSlvParams;
use super::LeverageSurface;
use crate::mc::McEstimate;
use crate::pricing::mc_stats::std_err_from_sums;
use crate::traits::ModelPricer;
use crate::traits::VanillaEuropeanCall;

/// Largest absolute deviation between a queried rate and the rate the
/// leverage surface was calibrated at that still counts as a match. Rates
/// live on the order of $10^{-2}$, so this is "equal up to float round-trip
/// noise" rather than a tolerance band.
const RATE_MATCH_TOL: f64 = 1e-12;

/// Heston-SLV Monte Carlo under a calibrated leverage surface: NaN outside the surface's `(s, tau)`
/// extent, where it would hold flat, and, from [`new`](Self::new), at rates other than its own.
#[derive(Debug, Clone)]
pub struct HestonSlvPricer {
  /// Model parameters.
  pub params: HestonSlvParams,
  /// Calibrated leverage surface.
  pub leverage: LeverageSurface,
  /// The `(r, q)` pair [`leverage`](Self::leverage) was calibrated against,
  /// or `None` when the surface carries no rate provenance. Not a pricing
  /// input — the rates that price come from the [`ModelPricer`] query; this
  /// is the guard those queries are checked against.
  pub calibration_rates: Option<(f64, f64)>,
  /// Number of MC paths.
  pub n_paths: usize,
  /// Time-discretization steps per year.
  pub steps_per_year: usize,
  /// RNG seed.
  pub seed: u64,
}

impl HestonSlvPricer {
  /// A pricer for a surface calibrated at `(r, q)`, which prices NaN at any other rates: `L(S, t)`
  /// depends on the drift it was fitted under.
  pub fn new(params: HestonSlvParams, leverage: LeverageSurface, r: f64, q: f64) -> Self {
    Self {
      params,
      leverage,
      calibration_rates: Some((r, q)),
      n_paths: 100_000,
      steps_per_year: 200,
      seed: 42,
    }
  }

  /// Pricer for a `leverage` surface with no rate provenance: the query's
  /// `r` and `q` drive drift and discounting, and no rate is rejected.
  ///
  /// Use this only for a surface that was *not* produced by calibrating
  /// against market data at a particular rate. A calibrated surface priced
  /// this way returns a confident number that no longer reprices the surface
  /// it was fitted to.
  pub fn unanchored(params: HestonSlvParams, leverage: LeverageSurface) -> Self {
    Self {
      params,
      leverage,
      calibration_rates: None,
      n_paths: 100_000,
      steps_per_year: 200,
      seed: 42,
    }
  }

  pub fn with_paths(mut self, n: usize) -> Self {
    self.n_paths = n;
    self
  }

  pub fn with_steps_per_year(mut self, n: usize) -> Self {
    self.steps_per_year = n;
    self
  }

  pub fn with_seed(mut self, seed: u64) -> Self {
    self.seed = seed;
    self
  }

  /// Whether `(r, q)` is the pair the surface was calibrated at; always for an unanchored pricer.
  fn rates_match(&self, r: f64, q: f64) -> bool {
    self
      .calibration_rates
      .is_none_or(|(r0, q0)| (r - r0).abs() <= RATE_MATCH_TOL && (q - q0).abs() <= RATE_MATCH_TOL)
  }

  /// The call price with its Monte Carlo error bar; NaN in every field, and no sample, where
  /// [`price_call`](ModelPricer::price_call) is NaN.
  pub fn price_call_estimate(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> McEstimate<f64> {
    if !(self.rates_match(r, q) && self.leverage.covers(s, tau)) {
      return McEstimate {
        mean: f64::NAN,
        std_err: f64::NAN,
        n_samples: 0,
      };
    }
    let n_steps = ((tau * self.steps_per_year as f64).round() as usize).max(1);
    let dt = tau / n_steps as f64;
    let sqrt_dt = dt.sqrt();
    let sigma_mixed = self.params.sigma_mixed();
    let rho_bar = (1.0 - self.params.heston.rho * self.params.heston.rho).sqrt();

    let mut normals = SimdNormal::<f64>::new(0.0, 1.0).seeded(&Deterministic::new(self.seed));
    let mut payoff_sum = 0.0;
    let mut payoff_sq_sum = 0.0;

    for _ in 0..self.n_paths {
      let mut x = s.ln();
      let mut v = self.params.heston.v0;

      for step in 0..n_steps {
        let t = step as f64 * dt;
        let dw_v = normals.sample() * sqrt_dt;
        let dw_ind = normals.sample() * sqrt_dt;
        let dw_x = self.params.heston.rho * dw_v + rho_bar * dw_ind;

        let v_pos = v.max(0.0);
        let sqrt_v = v_pos.sqrt();
        let s_curr = x.exp();

        let l = self.leverage.interpolate(s_curr, t);

        v = (v
          + self.params.heston.kappa * (self.params.heston.theta - v_pos) * dt
          + sigma_mixed * sqrt_v * dw_v)
          .max(0.0);

        let drift = (r - q) - 0.5 * l * l * v_pos;
        x += drift * dt + l * sqrt_v * dw_x;
      }

      let s_t = x.exp();
      let payoff = (s_t - k).max(0.0);
      payoff_sum += payoff;
      payoff_sq_sum += payoff * payoff;
    }

    let discount = (-r * tau).exp();
    McEstimate {
      mean: discount * payoff_sum / self.n_paths as f64,
      std_err: discount * std_err_from_sums(payoff_sum, payoff_sq_sum, self.n_paths),
      n_samples: self.n_paths,
    }
  }
}

impl ModelPricer for HestonSlvPricer {
  /// NaN for `(s, tau)` outside [`LeverageSurface::covers`] and, for a pricer from
  /// [`new`](HestonSlvPricer::new), at rates other than its calibration's.
  fn price_call(&self, s: f64, k: f64, r: f64, q: f64, tau: f64) -> f64 {
    self.price_call_estimate(s, k, r, q, tau).mean
  }
}

/// European vanilla call: the particle simulation averages $(S_T-K)^+$
/// under a log-spot drifted at $r-q$, so the default forward applies.
impl VanillaEuropeanCall for HestonSlvPricer {}

#[cfg(test)]
mod tests;
