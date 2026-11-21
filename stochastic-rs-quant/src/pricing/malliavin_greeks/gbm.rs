use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;
use stochastic_rs_stochastic::mc::McEstimate;

use super::estimate;
use super::undefined;
use crate::traits::ProcessExt;

/// Malliavin-weighted Greeks computation for a European call under Gbm dynamics.
#[derive(Debug, Clone)]
pub struct GbmMalliavinGreeks<S: SeedExt = Unseeded> {
  /// Spot price.
  pub s: f64,
  /// Annualised Black–Scholes volatility.
  pub sigma: f64,
  /// Risk-free rate.
  pub r: f64,
  /// Dividend yield
  pub q: f64,
  /// Time to maturity in years.
  pub tau: f64,
  /// Strike price
  pub k: f64,
  /// Number of Monte Carlo paths
  pub n_paths: usize,
  /// Number of time steps
  pub n_steps: usize,
  /// Each estimator simulates from a clone, so a `Deterministic` seed replays its paths.
  pub seed: S,
}

impl<S: SeedExt> GbmMalliavinGreeks<S> {
  pub fn with_seed<S2: SeedExt>(self, seed: S2) -> GbmMalliavinGreeks<S2> {
    GbmMalliavinGreeks {
      s: self.s,
      sigma: self.sigma,
      r: self.r,
      q: self.q,
      tau: self.tau,
      k: self.k,
      n_paths: self.n_paths,
      n_steps: self.n_steps,
      seed,
    }
  }

  /// Terminal prices and Brownian values `(S_T, W_T)` of `n_paths` Gbm paths; both are NaN, and so
  /// every Greek, for a spot or horizon that is not finite and positive or a non-finite rate.
  pub fn simulate(&self) -> (Array1<f64>, Array1<f64>) {
    if self.degenerate_query() {
      let nan = Array1::from_elem(self.n_paths, f64::NAN);
      return (nan.clone(), nan);
    }
    let mu = self.r - self.q;
    let dt = self.tau / (self.n_steps - 1) as f64;

    let gbm = Gbm::new(
      mu,
      self.sigma,
      self.n_steps,
      self.s,
      Some(self.tau),
      self.seed.clone(),
    );

    let mut s_terminal = Array1::<f64>::zeros(self.n_paths);
    let mut w_terminal = Array1::<f64>::zeros(self.n_paths);

    for i in 0..self.n_paths {
      let path = gbm.sample();
      let n = path.len();

      // Reconstruct W_T from the Gbm path:
      //   dW_k = (S_{k+1} - S_k - mu * S_k * dt) / (sigma * S_k)
      //   W_T  = sum of all dW_k
      let mut w = 0.0;
      for k in 0..(n - 1) {
        let s_prev = path[k];
        let s_curr = path[k + 1];
        let dw = if s_prev.abs() > 1e-14 {
          (s_curr - s_prev - mu * s_prev * dt) / (self.sigma * s_prev)
        } else {
          0.0
        };
        w += dw;
      }

      s_terminal[i] = path[n - 1];
      w_terminal[i] = w;
    }

    (s_terminal, w_terminal)
  }

  fn degenerate_query(&self) -> bool {
    let query = self.s > 0.0 && self.s.is_finite() && self.tau > 0.0 && self.tau.is_finite();
    !(query && self.r.is_finite() && self.q.is_finite())
  }

  /// Malliavin delta of the call, `E[e^{-rT} (S_T − K)^+ W_T / (S_0 σ T)]`.
  pub fn delta(&self) -> McEstimate<f64> {
    self.weighted(|w, t| w / (self.s * self.sigma * t))
  }

  /// Malliavin gamma, `E[e^{-rT} (S_T − K)^+ (W_T² − σT W_T − T) / (S_0² σ² T²)]`.
  pub fn gamma(&self) -> McEstimate<f64> {
    self.weighted(|w, t| {
      (w * w - self.sigma * t * w - t) / (self.s * self.s * self.sigma * self.sigma * t * t)
    })
  }

  /// Malliavin vega, `E[e^{-rT} (S_T − K)^+ ((W_T² − T) / (σT) − W_T)]`.
  pub fn vega(&self) -> McEstimate<f64> {
    self.weighted(|w, t| (w * w - t) / (self.sigma * t) - w)
  }

  /// Malliavin rho, `E[e^{-rT} (S_T − K)^+ (W_T / σ − T)]`; `rho_greek`, as `rho` names a
  /// correlation elsewhere.
  pub fn rho_greek(&self) -> McEstimate<f64> {
    self.weighted(|w, t| w / self.sigma - t)
  }

  fn weighted(&self, weight: impl Fn(f64, f64) -> f64) -> McEstimate<f64> {
    if self.degenerate_query() {
      return undefined();
    }
    let (s_t, w_t) = self.simulate();
    let discount = (-self.r * self.tau).exp();
    let (mut sum, mut sum_sq) = (0.0, 0.0);
    for i in 0..self.n_paths {
      let payoff = (s_t[i] - self.k).max(0.0);
      let contribution = discount * payoff * weight(w_t[i], self.tau);
      sum += contribution;
      sum_sq += contribution * contribution;
    }
    estimate(sum, sum_sq, self.n_paths)
  }

  /// Delta, gamma, vega and rho from one simulation for the set, which `Unseeded` accessors do
  /// not share; theta and the second-order Greeks are NaN.
  pub fn all_greeks(&self) -> crate::traits::Greeks {
    let (s_t, w_t) = self.simulate();
    let discount = (-self.r * self.tau).exp();
    let m = self.n_paths as f64;
    let t = self.tau;

    let mut sum_delta = 0.0;
    let mut sum_gamma = 0.0;
    let mut sum_vega = 0.0;
    let mut sum_rho = 0.0;

    for i in 0..self.n_paths {
      let payoff = (s_t[i] - self.k).max(0.0);
      let w = w_t[i];
      let disc_payoff = discount * payoff;

      // Delta weight
      let w_delta = w / (self.s * self.sigma * t);
      sum_delta += disc_payoff * w_delta;

      // Gamma weight
      let w_gamma =
        (w * w - self.sigma * t * w - t) / (self.s * self.s * self.sigma * self.sigma * t * t);
      sum_gamma += disc_payoff * w_gamma;

      // Vega weight
      let w_vega = (w * w - t) / (self.sigma * t) - w;
      sum_vega += disc_payoff * w_vega;

      // Rho weight
      let w_rho = w / self.sigma - t;
      sum_rho += disc_payoff * w_rho;
    }

    // theta and the four second-order Greeks stay NaN: this estimator has no
    // Malliavin weight for them, and NaN is how the crate says "not exposed"
    // — see `crate::traits::GreeksExt`.
    crate::traits::Greeks {
      delta: sum_delta / m,
      gamma: sum_gamma / m,
      vega: sum_vega / m,
      theta: f64::NAN,
      rho: sum_rho / m,
      vanna: f64::NAN,
      charm: f64::NAN,
      volga: f64::NAN,
      veta: f64::NAN,
    }
  }
}

impl<S: SeedExt> crate::traits::GreeksExt for GbmMalliavinGreeks<S> {
  fn delta(&self) -> Option<McEstimate<f64>> {
    Some(GbmMalliavinGreeks::delta(self))
  }

  fn gamma(&self) -> Option<McEstimate<f64>> {
    Some(GbmMalliavinGreeks::gamma(self))
  }

  fn vega(&self) -> Option<McEstimate<f64>> {
    Some(GbmMalliavinGreeks::vega(self))
  }

  fn rho(&self) -> Option<McEstimate<f64>> {
    Some(GbmMalliavinGreeks::rho_greek(self))
  }

  /// One simulation for the set, which `Unseeded` accessors do not share.
  fn greeks(&self) -> crate::traits::Greeks {
    self.all_greeks()
  }
}
