//! # Malliavin-weighted Greeks
//!
//! $$
//! \Delta=\mathbb E\!\left[e^{-rT}\Phi(S_T)\,\frac{W_T}{S_0\sigma T}\right]
//! $$
//!
use ndarray::Array1;
use ndarray::Array2;

use crate::distributions::normal::SimdNormal;
use crate::stochastic::diffusion::gbm::GBM;
use crate::stochastic::volatility::heston::Heston;
use crate::stochastic::volatility::HestonPow;
use crate::traits::ProcessExt;

struct HestonElKhatibPath {
  s: Array1<f64>,
  v: Array1<f64>,
  dw_s: Array1<f64>,
  dw_v: Array1<f64>,
}

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
  const EL_KHATIB_A_FLOOR: f64 = 1e-10;
  const EL_KHATIB_DU_EPS: f64 = 1e-3;
  const EL_KHATIB_VAR_FLOOR: f64 = 1e-12;

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

  fn sample_el_khatib_path(
    &self,
    normal_s: &SimdNormal<f64>,
    normal_perp: &SimdNormal<f64>,
  ) -> HestonElKhatibPath {
    let n_increments = self.n_steps - 1;
    let corr_scale = (1.0 - self.rho * self.rho).max(0.0).sqrt();
    let mut dw_s = Array1::<f64>::zeros(n_increments);
    let mut dw_v = Array1::<f64>::zeros(n_increments);

    for k in 0..n_increments {
      let dws = normal_s.sample_fast();
      let dwp = normal_perp.sample_fast();
      dw_s[k] = dws;
      dw_v[k] = self.rho * dws + corr_scale * dwp;
    }

    let (s, v) = self.simulate_heston_from_increments(&dw_s, &dw_v);
    HestonElKhatibPath { s, v, dw_s, dw_v }
  }

  fn simulate_heston_from_increments(
    &self,
    dw_s: &Array1<f64>,
    dw_v: &Array1<f64>,
  ) -> (Array1<f64>, Array1<f64>) {
    let dt = self.tau / (self.n_steps - 1) as f64;
    let mut s = Array1::<f64>::zeros(self.n_steps);
    let v = self.variance_path_from_shifted_increments(dw_v, 0.0);

    s[0] = self.s0;
    for k in 0..(self.n_steps - 1) {
      let s_prev = s[k];
      let sqrt_v = v[k].max(0.0).sqrt();
      s[k + 1] = s_prev + self.r * s_prev * dt + s_prev * sqrt_v * dw_s[k];
    }

    (s, v)
  }

  fn variance_path_from_shifted_increments(&self, dw_v: &Array1<f64>, shift: f64) -> Array1<f64> {
    let dt = self.tau / (self.n_steps - 1) as f64;
    let mut v = Array1::<f64>::zeros(self.n_steps);
    v[0] = self.v0.max(0.0);

    for k in 0..(self.n_steps - 1) {
      let v_prev = v[k].max(0.0);
      let dv =
        self.kappa * (self.theta - v_prev) * dt + self.xi * v_prev.sqrt() * (dw_v[k] + shift);
      v[k + 1] = (v[k] + dv).max(0.0);
    }

    v
  }

  fn el_khatib_a(&self, v: &Array1<f64>, dw_s: &Array1<f64>, dw_v: &Array1<f64>) -> f64 {
    self.el_khatib_a_with_shifts(v, dw_s, dw_v, 0.0, 0.0)
  }

  fn el_khatib_a_with_shifts(
    &self,
    v: &Array1<f64>,
    dw_s: &Array1<f64>,
    dw_v: &Array1<f64>,
    dw_s_shift: f64,
    dw_v_shift: f64,
  ) -> f64 {
    let dt = self.tau / (self.n_steps - 1) as f64;
    let n_increments = self.n_steps - 1;

    if self.xi.abs() <= f64::EPSILON || self.rho.abs() <= f64::EPSILON {
      let mut a = 0.0;
      for i in 0..n_increments {
        a += v[i].max(0.0).sqrt() * dt;
      }
      return a;
    }

    let mut a = 0.0;
    for i in 0..n_increments {
      let sqrt_v_i = v[i].max(0.0).sqrt();
      let mut g = sqrt_v_i;
      let mut d_v = self.rho * self.xi * sqrt_v_i;

      for j in i..n_increments {
        let sqrt_v_j = v[j].max(Self::EL_KHATIB_VAR_FLOOR).sqrt();
        let d_sigma = 0.5 / sqrt_v_j;
        g += d_sigma * d_v * (dw_s[j] + dw_s_shift - sqrt_v_j * dt);

        let raw_next = v[j]
          + self.kappa * (self.theta - v[j].max(0.0)) * dt
          + self.xi * v[j].max(0.0).sqrt() * (dw_v[j] + dw_v_shift);
        let tangent = 1.0 - self.kappa * dt + self.xi * 0.5 / sqrt_v_j * (dw_v[j] + dw_v_shift);
        d_v = if raw_next > 0.0 { d_v * tangent } else { 0.0 };
      }

      a += g * dt;
    }

    a
  }

  fn shifted_el_khatib_a(
    &self,
    dw_s: &Array1<f64>,
    dw_v: &Array1<f64>,
    dw_s_shift: f64,
    dw_v_shift: f64,
  ) -> f64 {
    let v = self.variance_path_from_shifted_increments(dw_v, dw_v_shift);
    self.el_khatib_a_with_shifts(&v, dw_s, dw_v, dw_s_shift, dw_v_shift)
  }

  fn el_khatib_du_a(&self, dw_s: &Array1<f64>, dw_v: &Array1<f64>) -> f64 {
    if self.xi.abs() <= f64::EPSILON || self.rho.abs() <= f64::EPSILON {
      return 0.0;
    }

    let dt = self.tau / (self.n_steps - 1) as f64;
    let eps = Self::EL_KHATIB_DU_EPS;
    let dw_s_shift = eps * dt;
    let dw_v_shift = self.rho * dw_s_shift;
    let a_up = self.shifted_el_khatib_a(dw_s, dw_v, dw_s_shift, dw_v_shift);
    let a_dn = self.shifted_el_khatib_a(dw_s, dw_v, -dw_s_shift, -dw_v_shift);

    (a_up - a_dn) / (2.0 * eps)
  }

  fn regularize_el_khatib_a(a: f64) -> f64 {
    if !a.is_finite() {
      return Self::EL_KHATIB_A_FLOOR;
    }

    if a.abs() >= Self::EL_KHATIB_A_FLOOR {
      a
    } else if a.is_sign_negative() {
      -Self::EL_KHATIB_A_FLOOR
    } else {
      Self::EL_KHATIB_A_FLOOR
    }
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

  /// Full El-Khatib Heston Malliavin Delta with the Skorohod correction.
  ///
  /// This uses the non-adapted kernel
  /// `G(t,T) = sqrt(V_t) + integral_t^T d sqrt(V_s)/dV_s * D_t V_s
  /// * (dW_s - sqrt(V_s) ds)` and the deterministic direction `u_t = 1`.
  ///
  /// Reference: Y. El-Khatib, "Computations of Greeks in stochastic volatility
  /// models via the Malliavin calculus", arXiv:0904.3247 (2009), Proposition 5
  /// for `G(t,T)` and the Delta section for the Skorohod correction.
  /// <https://arxiv.org/abs/0904.3247>
  ///
  /// The resulting discrete weight is:
  ///
  /// ```text
  /// pi_delta = (1/S0) * (W_T / A + D_u A / A^2),  A = integral_0^T G(t,T) dt
  /// ```
  ///
  /// `D_u A` is computed by a central Cameron-Martin perturbation of the
  /// stored Brownian increments. This is intentionally separate from [`delta`]:
  /// it is O(n_steps^2), more variance-sensitive, and mainly useful when the
  /// paper's full estimator is required explicitly.
  pub fn delta_el_khatib(&self) -> f64 {
    let dt = self.tau / (self.n_steps - 1) as f64;
    let sqrt_dt = dt.sqrt();
    let normal_s = SimdNormal::new(0.0, sqrt_dt);
    let normal_perp = SimdNormal::new(0.0, sqrt_dt);

    self.delta_el_khatib_from_normals(&normal_s, &normal_perp)
  }

  /// Seeded variant of [`delta_el_khatib`] for reproducible tests and benchmarks.
  pub fn delta_el_khatib_with_seed(&self, seed: u64) -> f64 {
    let dt = self.tau / (self.n_steps - 1) as f64;
    let sqrt_dt = dt.sqrt();
    let normal_s = SimdNormal::with_seed(0.0, sqrt_dt, seed);
    let normal_perp = SimdNormal::with_seed(0.0, sqrt_dt, seed ^ 0x9E37_79B9_7F4A_7C15);

    self.delta_el_khatib_from_normals(&normal_s, &normal_perp)
  }

  fn delta_el_khatib_from_normals(
    &self,
    normal_s: &SimdNormal<f64>,
    normal_perp: &SimdNormal<f64>,
  ) -> f64 {
    let discount = (-self.r * self.tau).exp();
    let m = self.n_paths as f64;
    let mut sum = 0.0;

    for _ in 0..self.n_paths {
      let path = self.sample_el_khatib_path(normal_s, normal_perp);
      let payoff = (path.s[self.n_steps - 1] - self.k).max(0.0);
      let w_t: f64 = path.dw_s.iter().sum();
      let a = Self::regularize_el_khatib_a(self.el_khatib_a(&path.v, &path.dw_s, &path.dw_v));
      let du_a = self.el_khatib_du_a(&path.dw_s, &path.dw_v);
      let weight = (w_t / a + du_a / (a * a)) / self.s0;

      if weight.is_finite() {
        sum += discount * payoff * weight;
      }
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
        seed,
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
        seed,
      );

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
  fn heston_delta_el_khatib_gbm_limit() {
    let greeks = HestonMalliavinGreeks {
      s0: 100.0,
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      xi: 0.0,
      rho: -0.7,
      r: 0.05,
      tau: 1.0,
      k: 100.0,
      n_paths: 20_000,
      n_steps: 64,
    };
    let delta = greeks.delta_el_khatib_with_seed(0x5EED);
    assert!(
      delta > 0.3 && delta < 0.9,
      "El-Khatib delta should reduce to the GBM Malliavin range, got {delta}"
    );
  }

  #[test]
  fn heston_delta_el_khatib_full_kernel_finite() {
    let greeks = HestonMalliavinGreeks {
      s0: 100.0,
      v0: 0.04,
      kappa: 2.0,
      theta: 0.04,
      xi: 0.15,
      rho: -0.4,
      r: 0.05,
      tau: 1.0,
      k: 100.0,
      n_paths: 2_000,
      n_steps: 32,
    };
    let delta = greeks.delta_el_khatib_with_seed(0xDEC0DE);
    assert!(
      delta.is_finite() && delta.abs() < 5.0,
      "El-Khatib delta should stay finite under the full kernel, got {delta}"
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
