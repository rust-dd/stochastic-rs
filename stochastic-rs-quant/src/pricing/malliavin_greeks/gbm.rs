use ndarray::Array1;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::diffusion::gbm::Gbm;

use crate::traits::ProcessExt;

/// Malliavin-weighted Greeks computation for a European call under Gbm dynamics.
#[derive(Debug, Clone)]
pub struct GbmMalliavinGreeks {
  /// Spot price
  pub s0: f64,
  /// Volatility
  pub sigma: f64,
  /// Risk-free rate
  pub r: f64,
  /// Dividend yield
  pub q: f64,
  /// Time to maturity T
  pub tau: f64,
  /// Strike price
  pub k: f64,
  /// Number of Monte Carlo paths
  pub n_paths: usize,
  /// Number of time steps
  pub n_steps: usize,
}

impl GbmMalliavinGreeks {
  /// Simulate Gbm paths and return terminal stock prices and terminal Brownian values.
  ///
  /// Returns `(S_T, W_T)` each of length `n_paths`.
  pub fn simulate(&self) -> (Array1<f64>, Array1<f64>) {
    let mu = self.r - self.q;
    let dt = self.tau / (self.n_steps - 1) as f64;

    let gbm = Gbm::new(
      mu,
      self.sigma,
      self.n_steps,
      Some(self.s0),
      Some(self.tau),
      Unseeded,
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

  /// Malliavin Delta for a European call.
  ///
  /// ```text
  /// Delta = (1/M) * sum_i [ e^{-rT} * payoff(S_T^i) * W_T^i / (S_0 * sigma * T) ]
  /// ```
  pub fn delta(&self) -> f64 {
    let (s_t, w_t) = self.simulate();
    let discount = (-self.r * self.tau).exp();
    let m = self.n_paths as f64;

    let mut sum = 0.0;
    for i in 0..self.n_paths {
      let payoff = (s_t[i] - self.k).max(0.0);
      let weight = w_t[i] / (self.s0 * self.sigma * self.tau);
      sum += discount * payoff * weight;
    }

    sum / m
  }

  /// Malliavin Gamma for a European call.
  ///
  /// ```text
  /// Gamma = (1/M) * sum_i [ e^{-rT} * payoff(S_T^i)
  ///         * (W_T^i^2 - sigma*T*W_T^i - T) / (S_0^2 * sigma^2 * T^2) ]
  /// ```
  pub fn gamma(&self) -> f64 {
    let (s_t, w_t) = self.simulate();
    let discount = (-self.r * self.tau).exp();
    let m = self.n_paths as f64;
    let t = self.tau;

    let mut sum = 0.0;
    for i in 0..self.n_paths {
      let payoff = (s_t[i] - self.k).max(0.0);
      let w = w_t[i];
      let weight =
        (w * w - self.sigma * t * w - t) / (self.s0 * self.s0 * self.sigma * self.sigma * t * t);
      sum += discount * payoff * weight;
    }

    sum / m
  }

  /// Malliavin Vega for a European call.
  ///
  /// ```text
  /// Vega = (1/M) * sum_i [ e^{-rT} * payoff(S_T^i)
  ///        * ((W_T^i^2 - T) / (sigma*T) - W_T^i) ]
  /// ```
  pub fn vega(&self) -> f64 {
    let (s_t, w_t) = self.simulate();
    let discount = (-self.r * self.tau).exp();
    let m = self.n_paths as f64;
    let t = self.tau;

    let mut sum = 0.0;
    for i in 0..self.n_paths {
      let payoff = (s_t[i] - self.k).max(0.0);
      let w = w_t[i];
      let weight = (w * w - t) / (self.sigma * t) - w;
      sum += discount * payoff * weight;
    }

    sum / m
  }

  /// Malliavin Rho for a European call (named `rho_greek` to avoid conflict with the
  /// correlation parameter `rho` used elsewhere).
  ///
  /// ```text
  /// Rho = (1/M) * sum_i [ e^{-rT} * payoff(S_T^i) * (W_T^i / sigma - T) ]
  /// ```
  pub fn rho_greek(&self) -> f64 {
    let (s_t, w_t) = self.simulate();
    let discount = (-self.r * self.tau).exp();
    let m = self.n_paths as f64;
    let t = self.tau;

    let mut sum = 0.0;
    for i in 0..self.n_paths {
      let payoff = (s_t[i] - self.k).max(0.0);
      let w = w_t[i];
      let weight = w / self.sigma - t;
      sum += discount * payoff * weight;
    }

    sum / m
  }

  /// Compute all four Malliavin Greeks in a single MC pass.
  ///
  /// Sharing simulated paths between the four estimators ensures the
  /// returned [`crate::traits::Greeks`] are mutually consistent — calling
  /// `delta()`, `gamma()`, `vega()`, `rho()` individually each runs a
  /// fresh simulation and would mix four different sample paths.
  /// First-order Greeks not produced by this Malliavin formula
  /// (`theta`, second-order `vanna`/`charm`/`volga`/`veta`) are returned
  /// as NaN.
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
      let w_delta = w / (self.s0 * self.sigma * t);
      sum_delta += disc_payoff * w_delta;

      // Gamma weight
      let w_gamma =
        (w * w - self.sigma * t * w - t) / (self.s0 * self.s0 * self.sigma * self.sigma * t * t);
      sum_gamma += disc_payoff * w_gamma;

      // Vega weight
      let w_vega = (w * w - t) / (self.sigma * t) - w;
      sum_vega += disc_payoff * w_vega;

      // Rho weight
      let w_rho = w / self.sigma - t;
      sum_rho += disc_payoff * w_rho;
    }

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

impl crate::traits::GreeksExt for GbmMalliavinGreeks {
  fn delta(&self) -> f64 {
    GbmMalliavinGreeks::delta(self)
  }
  fn gamma(&self) -> f64 {
    GbmMalliavinGreeks::gamma(self)
  }
  fn vega(&self) -> f64 {
    GbmMalliavinGreeks::vega(self)
  }
  fn rho(&self) -> f64 {
    GbmMalliavinGreeks::rho_greek(self)
  }
  /// Override the trait default — calling `delta()`/`gamma()`/`vega()`/`rho()`
  /// individually each runs an independent MC simulation, so the resulting
  /// Greeks would mix four different sample paths. [`Self::all_greeks`] uses
  /// a single shared simulation and returns mutually consistent estimators.
  fn greeks(&self) -> crate::traits::Greeks {
    self.all_greeks()
  }
}
