use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::heston::Heston;

use crate::traits::ProcessExt;

/// Malliavin-weighted Greeks computation for a European call under Heston dynamics.
#[derive(Debug, Clone)]
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
  pub(super) const EL_KHATIB_A_FLOOR: f64 = 1e-10;
  pub(super) const EL_KHATIB_DU_EPS: f64 = 1e-3;
  pub(super) const EL_KHATIB_VAR_FLOOR: f64 = 1e-12;

  /// Simulate Heston paths and return terminal prices, payoffs, full variance paths,
  /// and full W1 (price-driving Brownian) paths.
  ///
  /// Returns `(S_T, payoff, V_paths, W1_paths)`.
  pub fn simulate(&self) -> (Array1<f64>, Array1<f64>, Array2<f64>, Array2<f64>) {
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
      Unseeded,
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
      Unseeded,
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
  /// Uses the adapted covariance weight obtained from the leading Heston
  /// Malliavin derivative `D_t S_T ≈ S_T √V_t`:
  ///
  /// ```text
  /// γ ≈ S_T² ∫₀ᵀ V_t dt
  /// π_Δ = (1/S₀) · ∫₀ᵀ √V_t dW_t^S / ∫₀ᵀ V_t dt
  /// ```
  ///
  /// This is the same O(n) covariance approximation used by the
  /// Malliavin-Thalmaier engine. The exact El-Khatib `G(t,T)` kernel is
  /// non-adapted and requires Skorohod correction terms; using only its Ito
  /// integral part gives a biased and very slow estimator.
  ///
  /// Reference for the full non-adapted Heston kernel: Y. El-Khatib,
  /// "Computations of Greeks in stochastic volatility models via the Malliavin
  /// calculus", arXiv:0904.3247 (2009), Proposition 5 and the Delta section.
  /// <https://arxiv.org/abs/0904.3247>
  ///
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
      Unseeded,
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

      let pi_delta = numerator / (self.s0 * int_v.max(1e-12));
      sum += discount * payoff * pi_delta;
    }

    sum / m
  }

  /// Malliavin Gamma for a European call under Heston dynamics.
  ///
  /// Uses the second-order covariance weight:
  /// ```text
  /// π_Γ = π_Δ² − T / (S₀² · ∫₀ᵀ V_t dt)
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
      Unseeded,
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
  /// Uses finite-difference bump on v₀ with common random numbers (CRN).
  /// Each path pair (up/down) shares the same seed for noise cancellation.
  pub fn vega_v0(&self) -> f64 {
    let dv = 0.002_f64.min(self.v0 * 0.1);
    let discount = (-self.r * self.tau).exp();
    let m = self.n_paths as f64;

    let mut sum_up = 0.0;
    let mut sum_dn = 0.0;

    for i in 0..self.n_paths {
      let seed = 0xCAFE_u64.wrapping_add(i as u64);

      let heston_up = Heston::new(
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
        Deterministic::new(seed),
      );
      let heston_dn = Heston::new(
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
        Deterministic::new(seed),
      );

      let [s_up, _] = heston_up.sample();
      let [s_dn, _] = heston_dn.sample();
      sum_up += (s_up[self.n_steps - 1] - self.k).max(0.0);
      sum_dn += (s_dn[self.n_steps - 1] - self.k).max(0.0);
    }

    discount * (sum_up - sum_dn) / (m * 2.0 * dv)
  }
}

impl HestonMalliavinGreeks {
  /// Single-pass delta + gamma (sharing simulated paths). Vega is computed
  /// separately by [`Self::vega_v0`] because it uses an independent
  /// finite-difference bump with common-random-numbers and cannot share
  /// paths with the Malliavin estimator.
  fn delta_gamma_single_pass(&self) -> (f64, f64) {
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
      Unseeded,
    );

    let mut sum_delta = 0.0;
    let mut sum_gamma = 0.0;

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
      let disc_payoff = discount * payoff;
      sum_delta += disc_payoff * pi_delta;
      sum_gamma += disc_payoff * pi_gamma;
    }

    (sum_delta / m, sum_gamma / m)
  }
}

impl crate::traits::GreeksExt for HestonMalliavinGreeks {
  fn delta(&self) -> f64 {
    HestonMalliavinGreeks::delta(self)
  }
  fn gamma(&self) -> f64 {
    HestonMalliavinGreeks::gamma(self)
  }
  fn vega(&self) -> f64 {
    HestonMalliavinGreeks::vega_v0(self)
  }
  /// Override the trait default — calling `delta()` and `gamma()`
  /// individually each runs an independent Heston simulation, mixing two
  /// disjoint sets of paths. This impl shares one MC pass between delta
  /// and gamma. Vega still runs its own bumped-CRN simulation (it cannot
  /// share paths with the Malliavin estimator).
  fn greeks(&self) -> crate::traits::Greeks {
    let (delta, gamma) = self.delta_gamma_single_pass();
    let vega = self.vega_v0();
    crate::traits::Greeks {
      delta,
      gamma,
      vega,
      theta: f64::NAN,
      rho: f64::NAN,
      vanna: f64::NAN,
      charm: f64::NAN,
      volga: f64::NAN,
      veta: f64::NAN,
    }
  }
}
