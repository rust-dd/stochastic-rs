use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_stochastic::mc::McEstimate;
use stochastic_rs_stochastic::params::VarianceBoundary;
use stochastic_rs_stochastic::volatility::HestonPow;
use stochastic_rs_stochastic::volatility::heston::Euler;
use stochastic_rs_stochastic::volatility::heston::Heston;

use super::estimate;
use super::undefined;
use crate::pricing::heston::HestonParams;
use crate::pricing::mc_stats::std_err_from_sums;
use crate::traits::ProcessExt;

/// Malliavin-weighted Greeks of a European call under Heston dynamics; every estimator is NaN for a
/// spot or horizon that is not finite and positive or a non-finite rate.
#[derive(Debug, Clone)]
pub struct HestonMalliavinGreeks<S: SeedExt = Unseeded> {
  /// Spot price.
  pub s: f64,
  pub(super) params: HestonParams,
  /// Risk-free rate.
  pub r: f64,
  /// Strike price
  pub k: f64,
  /// Time to maturity in years.
  pub tau: f64,
  /// Number of Monte Carlo paths
  pub(super) n_paths: usize,
  /// Number of time steps
  pub(super) n_steps: usize,
  /// Each estimator simulates from a clone, so a `Deterministic` seed replays its paths.
  pub seed: S,
}

impl HestonMalliavinGreeks {
  /// Panics unless `params` passes [`HestonParams::new`], `n_paths >= 1` and `n_steps >= 2`; the
  /// estimators answer NaN for an `s`, `r` or `tau` off their domain.
  pub fn new(
    params: HestonParams,
    s: f64,
    k: f64,
    r: f64,
    tau: f64,
    n_paths: usize,
    n_steps: usize,
  ) -> Self {
    assert!(
      n_paths >= 1,
      "n_paths must satisfy `n_paths >= 1`, got n_paths = {n_paths:?}"
    );
    assert!(
      n_steps >= 2,
      "n_steps must satisfy `n_steps >= 2`, got n_steps = {n_steps:?}"
    );
    Self {
      s,
      params: params.checked(),
      r,
      k,
      tau,
      n_paths,
      n_steps,
      seed: Unseeded,
    }
  }
}

impl<S: SeedExt> HestonMalliavinGreeks<S> {
  pub fn with_seed<S2: SeedExt>(self, seed: S2) -> HestonMalliavinGreeks<S2> {
    HestonMalliavinGreeks {
      s: self.s,
      params: self.params,
      r: self.r,
      k: self.k,
      tau: self.tau,
      n_paths: self.n_paths,
      n_steps: self.n_steps,
      seed,
    }
  }

  pub(super) const EL_KHATIB_A_FLOOR: f64 = 1e-10;
  pub(super) const EL_KHATIB_DU_EPS: f64 = 1e-3;
  pub(super) const EL_KHATIB_VAR_FLOOR: f64 = 1e-12;

  pub(super) fn degenerate_query(&self) -> bool {
    let query = self.s > 0.0 && self.s.is_finite() && self.tau > 0.0 && self.tau.is_finite();
    !(query && self.r.is_finite())
  }

  fn process<P: SeedExt>(&self, v0: f64, seed: P) -> Heston<f64, Euler, P> {
    let HestonParams {
      kappa,
      theta,
      sigma,
      rho,
      ..
    } = self.params;
    Heston::new(
      self.s,
      v0,
      kappa,
      theta,
      sigma,
      rho,
      self.r,
      self.n_steps,
      Some(self.tau),
      HestonPow::Sqrt,
      // Absorption: the El Khatib tangent in `heston_el_khatib.rs` is that of the absorbed step.
      VarianceBoundary::Absorption,
      seed,
    )
  }

  /// The simulated `(S_T, payoff, V_paths, W1_paths)`, `W1` the Brownian that drives the price.
  pub fn simulate(&self) -> (Array1<f64>, Array1<f64>, Array2<f64>, Array2<f64>) {
    if self.degenerate_query() {
      let nan = Array1::from_elem(self.n_paths, f64::NAN);
      let paths = Array2::from_elem((self.n_paths, self.n_steps), f64::NAN);
      return (nan.clone(), nan, paths.clone(), paths);
    }
    let dt = self.tau / (self.n_steps - 1) as f64;

    let heston = self.process(self.params.v0, self.seed.clone());

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

  /// Pathwise delta `E[e^{-rT} 1{S_T > K} S_T / S_0]`, exact for a vanilla since `∂S_T/∂S_0 =
  /// S_T/S_0`; a non-smooth payoff needs [`delta`](Self::delta).
  pub fn delta_pathwise(&self) -> McEstimate<f64> {
    if self.degenerate_query() {
      return undefined();
    }
    let discount = (-self.r * self.tau).exp();

    let heston = self.process(self.params.v0, self.seed.clone());

    let (mut sum, mut sum_sq) = (0.0, 0.0);

    for _ in 0..self.n_paths {
      let [s_path, _] = heston.sample();
      let s_t = s_path[self.n_steps - 1];
      if s_t > self.k {
        let contribution = discount * s_t / self.s;
        sum += contribution;
        sum_sq += contribution * contribution;
      }
    }

    estimate(sum, sum_sq, self.n_paths)
  }

  /// Malliavin delta from the adapted covariance weight `∫√V dW^S / (S_0 ∫V dt)` of `D_t S_T ≈ S_T
  /// √V_t`, the Malliavin–Thalmaier engine's; El-Khatib's exact kernel is `delta_el_khatib`.
  pub fn delta(&self) -> McEstimate<f64> {
    if self.degenerate_query() {
      return undefined();
    }
    let dt = self.tau / (self.n_steps - 1) as f64;
    let discount = (-self.r * self.tau).exp();

    let heston = self.process(self.params.v0, self.seed.clone());

    let (mut sum, mut sum_sq) = (0.0, 0.0);

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

      let pi_delta = numerator / (self.s * int_v.max(1e-12));
      let contribution = discount * payoff * pi_delta;
      sum += contribution;
      sum_sq += contribution * contribution;
    }

    estimate(sum, sum_sq, self.n_paths)
  }

  /// Malliavin gamma from the second-order covariance weight `π_Δ² − T / (S_0² ∫V dt)`.
  pub fn gamma(&self) -> McEstimate<f64> {
    if self.degenerate_query() {
      return undefined();
    }
    let dt = self.tau / (self.n_steps - 1) as f64;
    let discount = (-self.r * self.tau).exp();

    let heston = self.process(self.params.v0, self.seed.clone());

    let (mut sum, mut sum_sq) = (0.0, 0.0);

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
      let pi_delta = numerator / (self.s * int_v_safe);
      let pi_gamma = pi_delta * pi_delta - self.tau / (self.s * self.s * int_v_safe);
      let contribution = discount * payoff * pi_gamma;
      sum += contribution;
      sum_sq += contribution * contribution;
    }

    estimate(sum, sum_sq, self.n_paths)
  }

  /// `∂V/∂v0` by a central bump of `v0` on common random numbers: each up path shares its
  /// down path's stream, one draw of the estimator's seed.
  pub fn vega_v0(&self) -> McEstimate<f64> {
    if self.degenerate_query() {
      return undefined();
    }
    let dv = 0.002_f64.min(self.params.v0 * 0.1);
    let discount = (-self.r * self.tau).exp();
    let m = self.n_paths as f64;

    let mut sum_up = 0.0;
    let mut sum_dn = 0.0;
    let (mut sum_diff, mut sum_diff_sq) = (0.0, 0.0);

    let seeds = self.seed.clone();
    for _ in 0..self.n_paths {
      let seed = seeds.next_seed();

      let heston_up = self.process(self.params.v0 + dv, Deterministic::new(seed));
      let heston_dn = self.process((self.params.v0 - dv).max(1e-6), Deterministic::new(seed));

      let [s_up, _] = heston_up.sample();
      let [s_dn, _] = heston_dn.sample();
      let up = (s_up[self.n_steps - 1] - self.k).max(0.0);
      let dn = (s_dn[self.n_steps - 1] - self.k).max(0.0);
      sum_up += up;
      sum_dn += dn;
      sum_diff += up - dn;
      sum_diff_sq += (up - dn) * (up - dn);
    }

    McEstimate {
      mean: discount * (sum_up - sum_dn) / (m * 2.0 * dv),
      std_err: discount / (2.0 * dv) * std_err_from_sums(sum_diff, sum_diff_sq, self.n_paths),
      n_samples: self.n_paths,
    }
  }
}

impl<S: SeedExt> HestonMalliavinGreeks<S> {
  /// Delta and gamma from one set of paths; [`Self::vega_v0`] bumps by finite difference on common
  /// random numbers, which cannot share the Malliavin paths.
  fn delta_gamma_single_pass(&self) -> (f64, f64) {
    if self.degenerate_query() {
      return (f64::NAN, f64::NAN);
    }
    let dt = self.tau / (self.n_steps - 1) as f64;
    let discount = (-self.r * self.tau).exp();
    let m = self.n_paths as f64;

    let heston = self.process(self.params.v0, self.seed.clone());

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
      let pi_delta = numerator / (self.s * int_v_safe);
      let pi_gamma = pi_delta * pi_delta - self.tau / (self.s * self.s * int_v_safe);
      let disc_payoff = discount * payoff;
      sum_delta += disc_payoff * pi_delta;
      sum_gamma += disc_payoff * pi_gamma;
    }

    (sum_delta / m, sum_gamma / m)
  }
}

impl<S: SeedExt> crate::traits::GreeksExt for HestonMalliavinGreeks<S> {
  fn delta(&self) -> Option<McEstimate<f64>> {
    Some(HestonMalliavinGreeks::delta(self))
  }

  fn gamma(&self) -> Option<McEstimate<f64>> {
    Some(HestonMalliavinGreeks::gamma(self))
  }

  fn vega(&self) -> Option<McEstimate<f64>> {
    Some(HestonMalliavinGreeks::vega_v0(self))
  }

  /// Delta and gamma from one simulation for the set, which `Unseeded` accessors do not share;
  /// vega from its own common-random-numbers bump.
  fn greeks(&self) -> crate::traits::Greeks {
    let (delta, gamma) = self.delta_gamma_single_pass();
    let vega = self.vega_v0().mean;
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
