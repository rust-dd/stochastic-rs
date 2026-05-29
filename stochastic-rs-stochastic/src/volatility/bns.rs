//! # Barndorff-Nielsen-Shephard (BNS) stochastic volatility
//!
//! $$
//! \begin{aligned}
//! \frac{dS_t}{S_t} &= \mu\,dt + \sigma_t\,dW_t,\\
//! d\sigma^2_t &= -\lambda\,\sigma^2_t\,dt + dZ_{\lambda t},
//! \end{aligned}
//! $$
//!
//! where $Z = (Z_t)_{t \ge 0}$ is a **non-negative Lévy subordinator**
//! (Barndorff-Nielsen & Shephard 2001, *Journal of the Royal Statistical
//! Society* Series B, §2). The volatility process is the
//! [`non-Gaussian Ornstein-Uhlenbeck process`][BNS01] with mean-reversion
//! rate $\lambda$; the jumps of $Z$ provide the positive shocks that
//! keep $\sigma^2_t \ge 0$ at all times.
//!
//! ## Subordinator choice — compound Poisson + Gamma jumps
//!
//! We use the standard "BNS-Gamma" approximation: $Z$ is a compound
//! Poisson process with Poisson intensity $\nu$ (jumps per unit time on
//! the $\lambda$-rescaled clock) and i.i.d. Gamma jump sizes
//! $Y_j \sim \mathrm{Gamma}(\omega, 1)$. The discretised dynamics on a
//! grid of step $\Delta t$ are:
//!
//! 1. $N_i \sim \mathrm{Poisson}(\nu \cdot \lambda \cdot \Delta t)$ — number
//!    of jumps in $[t_i, t_{i+1}]$.
//! 2. $\Delta Z_i = \sum_{j=1}^{N_i} Y_j$, $Y_j \sim \mathrm{Gamma}(\omega, 1)$.
//! 3. $\sigma^2_{t_{i+1}} = e^{-\lambda \Delta t} \sigma^2_{t_i} + \Delta Z_i$.
//! 4. $S_{t_{i+1}} = S_{t_i} \exp\!\bigl((\mu - \tfrac{1}{2}\sigma^2_{t_i}) \Delta t + \sigma_{t_i} \sqrt{\Delta t}\,\varepsilon_i\bigr)$ with $\varepsilon_i \sim \mathcal{N}(0, 1)$.
//!
//! Pure-jump leverage ($\rho \ne 0$ between $W$ and the jumps of $Z$) is
//! not yet implemented; the Barndorff-Nielsen-Shephard 2001b
//! shifted-Brownian construction needs careful pre-allocation of two
//! correlated noise streams.
//!
//! ## References
//!
//! - [BNS01]: Barndorff-Nielsen, O.E., Shephard, N. (2001), "Non-Gaussian
//!   Ornstein-Uhlenbeck-based models and some of their uses in financial
//!   economics", *Journal of the Royal Statistical Society* Series B,
//!   63(2), 167-241.
//! - Arai, T. (2015), "Local risk-minimization for Barndorff-Nielsen and
//!   Shephard models with volatility risk premium", arXiv:1506.01477.
//! - Roberts, G.O., Papaspiliopoulos, O., Dellaportas, P. (2004),
//!   "Bayesian inference for non-Gaussian Ornstein-Uhlenbeck stochastic
//!   volatility processes", *JRSS* B 66(2), 369-393.

use ndarray::Array1;
use rand::Rng;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::FloatExt;
use stochastic_rs_distributions::gamma::SimdGamma;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::traits::ProcessExt;

/// BNS-Gamma stochastic volatility model. See module documentation for
/// the SDE form and the discretisation.
pub struct Bns<T: FloatExt, S: SeedExt = Unseeded> {
  /// Initial stock price.
  pub s0: Option<T>,
  /// Initial variance $\sigma^2_0 > 0$.
  pub sigma2_0: T,
  /// Mean-reversion rate $\lambda > 0$.
  pub lambda: T,
  /// Drift $\mu$ (risk-neutral $r - q$ when pricing).
  pub mu: T,
  /// Jump intensity $\nu > 0$ — expected number of jumps per unit time
  /// **on the $\lambda$-rescaled clock**.
  pub nu: T,
  /// Gamma shape parameter $\omega > 0$ for the i.i.d. jump sizes.
  pub jump_shape: T,
  /// Number of time steps.
  pub n: usize,
  /// Time to maturity.
  pub t: Option<T>,
  /// Seed strategy.
  pub seed: S,
}

impl<T: FloatExt, S: SeedExt> Bns<T, S> {
  pub fn new(
    s0: Option<T>,
    sigma2_0: T,
    lambda: T,
    mu: T,
    nu: T,
    jump_shape: T,
    n: usize,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(sigma2_0 > T::zero(), "σ²₀ must be positive");
    assert!(lambda > T::zero(), "λ must be positive");
    assert!(nu > T::zero(), "jump intensity ν must be positive");
    assert!(jump_shape > T::zero(), "Gamma shape ω must be positive");
    Self {
      s0,
      sigma2_0,
      lambda,
      mu,
      nu,
      jump_shape,
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt> ProcessExt<T> for Bns<T, S> {
  /// `(log-stock path, variance σ² path)`.
  type Output = [Array1<T>; 2];

  fn sample(&self) -> Self::Output {
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);
    let lam_dt = self.lambda * dt;
    let decay = (-lam_dt).exp(); // e^{-λ·Δt}
    let jump_rate_per_step = (self.nu * lam_dt).to_f64().unwrap();

    let mut s = Array1::<T>::zeros(self.n);
    let mut sigma2 = Array1::<T>::zeros(self.n);

    s[0] = self.s0.unwrap_or(T::one());
    sigma2[0] = self.sigma2_0;

    // Samplers are built inside `sample` because their internal
    // UnsafeCell-backed buffers are not `Sync` — keeping them local lets
    // `Bns` itself remain `Send + Sync` (required by `ProcessExt`).
    let jump_dist = SimdGamma::<T>::new(self.jump_shape, T::one(), &Unseeded);
    let normal_dist = SimdNormal::<T>::new(T::zero(), T::one(), &Unseeded);
    let mut rng = rand::rng();
    for i in 1..self.n {
      // 1. Number of jumps in [t_{i-1}, t_i] from the compound-Poisson
      //    subordinator.
      let n_jumps = rand_poisson(&mut rng, jump_rate_per_step);

      // 2. Sum N Gamma(ω, 1) jump sizes to form ΔZ.
      let mut dz = T::zero();
      for _ in 0..n_jumps {
        dz += jump_dist.sample_fast();
      }

      // 3. Variance update: σ²_t = e^{-λΔt} σ²_{t-Δt} + ΔZ.
      sigma2[i] = decay * sigma2[i - 1] + dz;

      // 4. Asset update under risk-neutral log-Euler.
      let v_prev = sigma2[i - 1];
      let eps = normal_dist.sample_fast();
      let log_inc =
        (self.mu - v_prev * T::from_f64_fast(0.5)) * dt + v_prev.sqrt() * dt.sqrt() * eps;
      s[i] = s[i - 1] * log_inc.exp();
    }

    [s, sigma2]
  }
}

/// Poisson sampler using Knuth's algorithm for small rates (`λ ≤ 30`) and
/// the rejection method based on a normal proposal for large rates.
/// `lambda < 0` returns 0.
fn rand_poisson<R: Rng + ?Sized>(rng: &mut R, lambda: f64) -> u32 {
  if lambda <= 0.0 {
    return 0;
  }
  if lambda < 30.0 {
    // Knuth — exponential of independent uniforms.
    let l = (-lambda).exp();
    let mut k = 0_u32;
    let mut p = 1.0_f64;
    loop {
      k += 1;
      let u: f64 = rng.random_range(0.0..1.0);
      p *= u;
      if p <= l {
        return k - 1;
      }
    }
  } else {
    // Atkinson 1979 rejection on a Gaussian proposal.
    let c = 0.767 - 3.36 / lambda;
    let beta = std::f64::consts::PI / (3.0 * lambda).sqrt();
    let alpha = beta * lambda;
    let k = c.ln() - lambda - beta.ln();
    loop {
      let u: f64 = rng.random_range(0.0..1.0);
      let x = (alpha - ((1.0 - u) / u).ln()) / beta;
      let n = (x + 0.5).floor() as i64;
      if n < 0 {
        continue;
      }
      let v: f64 = rng.random_range(0.0..1.0);
      let y = alpha - beta * x;
      let lhs = y + (v / (1.0 + y.exp()).powi(2)).ln();
      let rhs = k + (n as f64) * lambda.ln()
        - stochastic_rs_distributions::special::ln_gamma((n + 1) as f64);
      if lhs <= rhs {
        return n as u32;
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Variance path stays strictly positive.
  #[test]
  fn bns_variance_non_negative() {
    let p = Bns::<f64>::new(
      Some(100.0),
      0.04,
      2.0,
      0.0,
      5.0,
      2.0,
      512,
      Some(1.0),
      Unseeded,
    );
    let [_s, v] = p.sample();
    assert!(v.iter().all(|x| *x >= 0.0));
    assert!(v.iter().any(|x| *x > 0.0));
  }

  /// Stock stays strictly positive (log-Euler structure).
  #[test]
  fn bns_stock_strictly_positive() {
    let p = Bns::<f64>::new(
      Some(100.0),
      0.04,
      1.5,
      0.05,
      3.0,
      2.0,
      256,
      Some(1.0),
      Unseeded,
    );
    let [s, _v] = p.sample();
    assert!(s.iter().all(|x| *x > 0.0));
  }

  /// Stationary mean of σ² under the BNS-Gamma model is E[Z₁] / (1 - e^{-λ}).
  /// On a long horizon with many resamples, the time-averaged variance
  /// should approach $\nu \cdot \omega$ (jump rate × mean jump size).
  #[test]
  fn bns_stationary_variance_matches_jump_intensity() {
    let nu = 4.0_f64;
    let omega = 1.5_f64;
    let p = Bns::<f64>::new(
      Some(100.0),
      omega * nu,
      1.0,
      0.0,
      nu,
      omega,
      4096,
      Some(40.0),
      Unseeded,
    );
    let [_s, v] = p.sample();
    let burn_in = 1024;
    let mean: f64 = v.iter().skip(burn_in).copied().sum::<f64>() / (v.len() - burn_in) as f64;
    let expected = nu * omega;
    assert!(
      (mean - expected).abs() / expected < 0.20,
      "BNS-Gamma stationary E[σ²] = {mean}, expected ≈ {expected}"
    );
  }

  /// Reject zero / negative jump intensity.
  #[test]
  #[should_panic(expected = "jump intensity")]
  fn bns_zero_intensity_panics() {
    let _ = Bns::<f64>::new(
      Some(100.0),
      0.04,
      1.0,
      0.0,
      0.0,
      1.0,
      64,
      Some(1.0),
      Unseeded,
    );
  }
}
