//! # Malliavin-weighted Greeks
//!
//! $$
//! \Delta=\mathbb E\!\left[e^{-rT}\Phi(S_T)\,\frac{W_T}{S_0\sigma T}\right]
//! $$
//!
use ndarray::Array1;
use ndarray::Array2;

use crate::stochastic::diffusion::gbm::GBM;
use crate::stochastic::volatility::HestonPow;
use crate::stochastic::volatility::heston::Heston;
use crate::traits::ProcessExt;

/// Container for the four standard Greeks.
pub struct Greeks {
  pub delta: f64,
  pub gamma: f64,
  pub vega: f64,
  pub rho: f64,
}

/// Malliavin-weighted Greeks computation for a European call under GBM dynamics.
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
  /// Simulate GBM paths and return terminal stock prices and terminal Brownian values.
  ///
  /// Returns `(S_T, W_T)` each of length `n_paths`.
  fn simulate(&self) -> (Array1<f64>, Array1<f64>) {
    let mu = self.r - self.q;
    let dt = self.tau / (self.n_steps - 1) as f64;

    let gbm = GBM::new(mu, self.sigma, self.n_steps, Some(self.s0), Some(self.tau));

    let mut s_terminal = Array1::<f64>::zeros(self.n_paths);
    let mut w_terminal = Array1::<f64>::zeros(self.n_paths);

    for i in 0..self.n_paths {
      let path = gbm.sample();
      let n = path.len();

      // Reconstruct W_T from the GBM path:
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

  /// Compute all four Greeks in a single pass (single simulation).
  pub fn all_greeks(&self) -> Greeks {
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

    Greeks {
      delta: sum_delta / m,
      gamma: sum_gamma / m,
      vega: sum_vega / m,
      rho: sum_rho / m,
    }
  }
}

/// Malliavin-weighted Greeks computation for a European call under Heston dynamics.
pub struct HestonMalliavinGreeks {
  /// Spot price
  pub s0: f64,
  /// Initial variance
  pub v0: f64,
  /// Mean reversion speed
  pub kappa: f64,
  /// Long-run variance
  pub theta: f64,
  /// Vol of vol
  pub xi: f64,
  /// Correlation between price and variance Brownians
  pub rho: f64,
  /// Risk-free rate
  pub r: f64,
  /// Time to maturity
  pub tau: f64,
  /// Strike price
  pub k: f64,
  /// Number of Monte Carlo paths
  pub n_paths: usize,
  /// Number of time steps
  pub n_steps: usize,
}

impl HestonMalliavinGreeks {
  /// Simulate Heston paths and return terminal prices, payoffs, full variance paths,
  /// and full W1 (price-driving Brownian) paths.
  ///
  /// Returns `(S_T, payoff, V_paths, W1_paths)`.
  fn simulate(&self) -> (Array1<f64>, Array1<f64>, Array2<f64>, Array2<f64>) {
    let dt = self.tau / (self.n_steps - 1) as f64;

    let heston = Heston::new(
      Some(self.s0),
      Some(self.v0),
      self.kappa,
      self.theta,
      self.xi,
      self.rho,
      self.r,
      self.n_steps,
      Some(self.tau),
      HestonPow::Sqrt,
      Some(false),
    );

    let mut s_terminal = Array1::<f64>::zeros(self.n_paths);
    let mut payoffs = Array1::<f64>::zeros(self.n_paths);
    let mut v_paths = Array2::<f64>::zeros((self.n_paths, self.n_steps));
    let mut w1_paths = Array2::<f64>::zeros((self.n_paths, self.n_steps));

    for i in 0..self.n_paths {
      let [s_path, v_path] = heston.sample();

      // Store the full variance path
      for j in 0..self.n_steps {
        v_paths[[i, j]] = v_path[j];
      }

      // Reconstruct the W1 Brownian increments from the price path:
      //   dW1_k = (S_{k+1} - S_k - r * S_k * dt) / (sqrt(V_k) * S_k)
      let mut w1 = 0.0;
      w1_paths[[i, 0]] = 0.0;
      for k in 0..(self.n_steps - 1) {
        let s_prev = s_path[k];
        let v_k = v_path[k].max(1e-12);
        let dw1 = if s_prev.abs() > 1e-14 {
          (s_path[k + 1] - s_prev - self.r * s_prev * dt) / (v_k.sqrt() * s_prev)
        } else {
          0.0
        };
        w1 += dw1;
        w1_paths[[i, k + 1]] = w1;
      }

      s_terminal[i] = s_path[self.n_steps - 1];
      payoffs[i] = (s_terminal[i] - self.k).max(0.0);
    }

    (s_terminal, payoffs, v_paths, w1_paths)
  }

  /// Pathwise Delta for a European call under Heston dynamics.
  ///
  /// Since the variance process does not depend on S₀, we have ∂S_T/∂S₀ = S_T/S₀,
  /// giving the exact pathwise estimator:
  /// ```text
  /// Delta = E[ e^{-rT} · 1{S_T > K} · S_T / S_0 ]
  /// ```
  ///
  /// This is exact for smooth payoffs (vanilla calls/puts) and does not require
  /// Malliavin weights. For non-smooth payoffs (digitals, barriers), use [`delta`].
  pub fn delta_pathwise(&self) -> f64 {
    let discount = (-self.r * self.tau).exp();
    let m = self.n_paths as f64;

    let heston = Heston::new(
      Some(self.s0),
      Some(self.v0),
      self.kappa,
      self.theta,
      self.xi,
      self.rho,
      self.r,
      self.n_steps,
      Some(self.tau),
      HestonPow::Sqrt,
      Some(false),
    );

    let mut sum = 0.0;

    for _ in 0..self.n_paths {
      let [s_path, _] = heston.sample();
      let s_t = s_path[self.n_steps - 1];
      if s_t > self.k {
        sum += discount * s_t / self.s0;
      }
    }

    sum / m
  }

  /// Malliavin Delta for a European call under Heston dynamics.
  ///
  /// Uses the zeroth-order Malliavin covariance approximation from
  /// El-Khatib (2009, arXiv:0904.3247) and the Fourier-Malliavin framework:
  ///
  /// ```text
  /// D_t S_T ≈ S_T √V_t   (zeroth-order, ignoring σ'(V) correction)
  /// γ = S_T² ∫₀ᵀ V_t dt  (Malliavin covariance)
  ///
  /// π_Δ = (1/S₀) · ∫₀ᵀ √V_t dW_t^S / ∫₀ᵀ V_t dt
  /// ```
  ///
  /// Works for non-smooth payoffs (digitals, barriers) where pathwise fails.
  /// For vanilla calls, prefer [`delta_pathwise`] which is exact.
  pub fn delta(&self) -> f64 {
    let dt = self.tau / (self.n_steps - 1) as f64;
    let discount = (-self.r * self.tau).exp();
    let m = self.n_paths as f64;

    let heston = Heston::new(
      Some(self.s0),
      Some(self.v0),
      self.kappa,
      self.theta,
      self.xi,
      self.rho,
      self.r,
      self.n_steps,
      Some(self.tau),
      HestonPow::Sqrt,
      Some(false),
    );

    let mut sum = 0.0;

    for _ in 0..self.n_paths {
      let [s_path, v_path] = heston.sample();
      let payoff = (s_path[self.n_steps - 1] - self.k).max(0.0);

      // Reconstruct dW^S from the arithmetic Euler step:
      //   S_{k+1} = S_k + r·S_k·dt + √V_k·S_k·dW^S_k
      //   dW^S_k  = (S_{k+1} - S_k - r·S_k·dt) / (√V_k · S_k)
      let mut numerator = 0.0; // ∫ √V_t dW_t^S
      let mut int_v = 0.0; // ∫ V_t dt

      for k in 0..(self.n_steps - 1) {
        let v_k = v_path[k].max(1e-12);
        let sqrt_v_k = v_k.sqrt();

        int_v += v_k * dt;

        let s_prev = s_path[k];
        let dw_s = if s_prev.abs() > 1e-14 {
          (s_path[k + 1] - s_prev - self.r * s_prev * dt) / (sqrt_v_k * s_prev)
        } else {
          0.0
        };

        // √V_k · dW^S_k  (note: dw_s already contains √dt via reconstruction)
        numerator += sqrt_v_k * dw_s;
      }

      let int_v_safe = int_v.max(1e-12);
      let pi_delta = numerator / (self.s0 * int_v_safe);

      sum += discount * payoff * pi_delta;
    }

    sum / m
  }

  /// Malliavin Gamma for a European call under Heston dynamics.
  ///
  /// Uses the second-order Malliavin weight derived from the covariance:
  /// ```text
  /// π_Γ = (π_Δ² − ∫₀ᵀ dt / (S₀² · (∫₀ᵀ V_t dt)²))
  /// ```
  pub fn gamma(&self) -> f64 {
    let dt = self.tau / (self.n_steps - 1) as f64;
    let discount = (-self.r * self.tau).exp();
    let m = self.n_paths as f64;

    let heston = Heston::new(
      Some(self.s0),
      Some(self.v0),
      self.kappa,
      self.theta,
      self.xi,
      self.rho,
      self.r,
      self.n_steps,
      Some(self.tau),
      HestonPow::Sqrt,
      Some(false),
    );

    let mut sum = 0.0;

    for _ in 0..self.n_paths {
      let [s_path, v_path] = heston.sample();
      let payoff = (s_path[self.n_steps - 1] - self.k).max(0.0);

      let mut numerator = 0.0;
      let mut int_v = 0.0;

      for k in 0..(self.n_steps - 1) {
        let v_k = v_path[k].max(1e-12);
        let sqrt_v_k = v_k.sqrt();
        int_v += v_k * dt;

        let s_prev = s_path[k];
        let dw_s = if s_prev.abs() > 1e-14 {
          (s_path[k + 1] - s_prev - self.r * s_prev * dt) / (sqrt_v_k * s_prev)
        } else {
          0.0
        };
        numerator += sqrt_v_k * dw_s;
      }

      let int_v_safe = int_v.max(1e-12);
      let pi_delta = numerator / (self.s0 * int_v_safe);
      let pi_gamma = pi_delta * pi_delta - self.tau / (self.s0 * self.s0 * int_v_safe);

      sum += discount * payoff * pi_gamma;
    }

    sum / m
  }

  /// Vega with respect to initial variance v₀.
  ///
  /// Uses finite-difference bump on v₀ with common random numbers (CRN)
  /// via seeded Heston simulation.
  pub fn vega_v0(&self) -> f64 {
    let dv = 0.002_f64.min(self.v0 * 0.1);
    let discount = (-self.r * self.tau).exp();
    let m = self.n_paths as f64;

    let heston_up = Heston::seeded(
      Some(self.s0),
      Some(self.v0 + dv),
      self.kappa,
      self.theta,
      self.xi,
      self.rho,
      self.r,
      self.n_steps,
      Some(self.tau),
      HestonPow::Sqrt,
      Some(false),
      0xCAFE,
    );
    let heston_dn = Heston::seeded(
      Some(self.s0),
      Some((self.v0 - dv).max(1e-6)),
      self.kappa,
      self.theta,
      self.xi,
      self.rho,
      self.r,
      self.n_steps,
      Some(self.tau),
      HestonPow::Sqrt,
      Some(false),
      0xCAFE,
    );

    let mut sum_up = 0.0;
    let mut sum_dn = 0.0;

    for _ in 0..self.n_paths {
      let [s_up, _] = heston_up.sample();
      let [s_dn, _] = heston_dn.sample();
      sum_up += (s_up[self.n_steps - 1] - self.k).max(0.0);
      sum_dn += (s_dn[self.n_steps - 1] - self.k).max(0.0);
    }

    discount * (sum_up - sum_dn) / (m * 2.0 * dv)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn gbm_delta_positive_for_call() {
    let greeks = GbmMalliavinGreeks {
      s0: 100.0,
      sigma: 0.2,
      r: 0.05,
      q: 0.0,
      tau: 1.0,
      k: 100.0,
      n_paths: 50_000,
      n_steps: 252,
    };
    let delta = greeks.delta();
    assert!(
      delta > 0.0 && delta < 1.0,
      "Call delta should be in (0,1), got {delta}"
    );
  }

  #[test]
  fn gbm_gamma_positive() {
    let greeks = GbmMalliavinGreeks {
      s0: 100.0,
      sigma: 0.2,
      r: 0.05,
      q: 0.0,
      tau: 1.0,
      k: 100.0,
      n_paths: 50_000,
      n_steps: 252,
    };
    let gamma = greeks.gamma();
    assert!(gamma > 0.0, "Call gamma should be > 0, got {gamma}");
  }

  #[test]
  fn gbm_vega_positive() {
    let greeks = GbmMalliavinGreeks {
      s0: 100.0,
      sigma: 0.2,
      r: 0.05,
      q: 0.0,
      tau: 1.0,
      k: 100.0,
      n_paths: 50_000,
      n_steps: 252,
    };
    let vega = greeks.vega();
    assert!(vega > 0.0, "Call vega should be > 0, got {vega}");
  }

  #[test]
  fn gbm_all_greeks_consistent() {
    let greeks = GbmMalliavinGreeks {
      s0: 100.0,
      sigma: 0.2,
      r: 0.05,
      q: 0.0,
      tau: 1.0,
      k: 100.0,
      n_paths: 100_000,
      n_steps: 252,
    };
    let g = greeks.all_greeks();
    // BS analytical: Delta ~ 0.64, Gamma ~ 0.019, Vega ~ 37.5, Rho ~ 53.2
    assert!(g.delta > 0.3 && g.delta < 0.9, "delta={}", g.delta);
    assert!(g.gamma > 0.005 && g.gamma < 0.05, "gamma={}", g.gamma);
    assert!(g.vega > 10.0 && g.vega < 60.0, "vega={}", g.vega);
  }

  #[test]
  fn heston_delta_positive_for_call() {
    let greeks = HestonMalliavinGreeks {
      s0: 100.0,
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      xi: 0.3,
      rho: -0.7,
      r: 0.05,
      tau: 1.0,
      k: 100.0,
      n_paths: 50_000,
      n_steps: 252,
    };
    let delta = greeks.delta();
    assert!(
      delta > 0.0 && delta < 1.0,
      "Heston Malliavin delta should be in (0,1), got {delta}"
    );
  }

  #[test]
  fn heston_delta_pathwise_positive() {
    let greeks = HestonMalliavinGreeks {
      s0: 100.0,
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      xi: 0.3,
      rho: -0.7,
      r: 0.05,
      tau: 1.0,
      k: 100.0,
      n_paths: 50_000,
      n_steps: 252,
    };
    let delta = greeks.delta_pathwise();
    assert!(
      delta > 0.3 && delta < 0.9,
      "Heston pathwise delta should be ~0.6 for ATM, got {delta}"
    );
  }

  #[test]
  fn heston_gamma_positive() {
    let greeks = HestonMalliavinGreeks {
      s0: 100.0,
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      xi: 0.3,
      rho: -0.7,
      r: 0.05,
      tau: 1.0,
      k: 100.0,
      n_paths: 100_000,
      n_steps: 252,
    };
    let gamma = greeks.gamma();
    assert!(
      gamma > 0.0 && gamma < 0.1,
      "Heston gamma should be positive and reasonable, got {gamma}"
    );
  }

  #[test]
  fn heston_vega_v0_positive() {
    let greeks = HestonMalliavinGreeks {
      s0: 100.0,
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      xi: 0.3,
      rho: -0.7,
      r: 0.05,
      tau: 1.0,
      k: 100.0,
      n_paths: 50_000,
      n_steps: 252,
    };
    let vega = greeks.vega_v0();
    assert!(vega > 0.0, "Heston vega_v0 should be > 0, got {vega}");
  }

  #[test]
  fn heston_malliavin_vs_pathwise_consistent() {
    let greeks = HestonMalliavinGreeks {
      s0: 100.0,
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      xi: 0.1, // low vol-of-vol so zeroth-order approx is good
      rho: -0.3,
      r: 0.05,
      tau: 1.0,
      k: 100.0,
      n_paths: 200_000,
      n_steps: 252,
    };
    let mall = greeks.delta();
    let path = greeks.delta_pathwise();
    let rel_err = ((mall - path) / path).abs();
    assert!(
      rel_err < 0.15,
      "Malliavin and pathwise delta should be close for low xi, got mall={mall} path={path} rel_err={rel_err}"
    );
  }
}
