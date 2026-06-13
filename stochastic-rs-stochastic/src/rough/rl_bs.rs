//! # Fractional Black–Scholes
//!
//! $$
//! S_t = S_0\,\exp\!\left(rt - \tfrac{1}{2}\sigma^2 t^{2H} + \sigma W^H_t\right)
//! $$
//!
//! Closed-form pathwise solution of the fractional Black–Scholes SDE
//! $dS_t = rS_t\,dt + \sigma S_t\,dW^H_t$ under Wick–Itô–Skorokhod integration
//! (Hu & Øksendal 2003). The fBM path is generated non-cumulatively via
//! [`RlFBm`], then the log-price formula is applied pointwise — no Euler
//! step, no drift discretisation.
//!
//! Reference: Bilokon & Wong (2026) §5.1; Hu Y., Øksendal B. *Fractional
//! white noise calculus and applications to finance*, IDAQP 6 (2003), 1–32;
//! Necula C. *Option pricing in a fractional Brownian motion environment*,
//! Working Paper (2008).
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::markov_lift::MarkovLift;
use super::markov_lift::RoughSimd;
use super::rl_fbm::RlFBm;
use crate::buffer::array1_from_fill;
use crate::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Fractional Black–Scholes asset path driven by RL-fBM.
#[derive(Clone)]
pub struct RlBlackScholes<T: FloatExt, S: SeedExt = Unseeded> {
  /// Hurst exponent.
  pub hurst: T,
  /// Initial spot $S_0$.
  pub s0: T,
  /// Risk-free rate $r$.
  pub r: T,
  /// Volatility $\sigma$.
  pub sigma: T,
  /// Number of simulation points.
  pub n: usize,
  /// Simulation horizon.
  pub t: Option<T>,
  /// Quadrature degree.
  pub degree: Option<usize>,
  /// Seed strategy.
  pub seed: S,
  fbm: RlFBm<T>,
}

impl<T: FloatExt, S: SeedExt> RlBlackScholes<T, S> {
  #[must_use]
  pub fn new(
    hurst: T,
    s0: T,
    r: T,
    sigma: T,
    n: usize,
    t: Option<T>,
    degree: Option<usize>,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    assert!(s0 > T::zero(), "s0 must be positive");
    assert!(sigma >= T::zero(), "sigma must be non-negative");
    Self {
      hurst,
      s0,
      r,
      sigma,
      n,
      t,
      degree,
      seed,
      fbm: RlFBm::new(hurst, n, t, degree, Unseeded),
    }
  }
}

impl<T: FloatExt + RoughSimd, S: SeedExt> RlBlackScholes<T, S> {
  /// Generate $m$ independent fractional Black–Scholes asset paths.
  /// Each path is $S_0 \exp(rt - \tfrac{1}{2}\sigma^2 t^{2H} + \sigma W^H_t)$
  /// applied pointwise to a batch of RL-fBM paths.
  pub fn sample_batch(&self, m: usize) -> Array2<T> {
    let fbm = self.fbm.sample_batch_impl(&self.seed.derive(), m);

    let horizon = self.t.unwrap_or(T::one());
    let dt = horizon / T::from_usize_(self.n - 1);
    let two_h = T::from_f64_fast(2.0) * self.hurst;
    let half_sigma_sq = T::from_f64_fast(0.5) * self.sigma * self.sigma;

    let mut spots = Array2::<T>::zeros((m, self.n));
    for p in 0..m {
      spots[[p, 0]] = self.s0;
      for i in 1..self.n {
        let t_i = dt * T::from_usize_(i);
        let log_s = self.r * t_i - half_sigma_sq * t_i.powf(two_h) + self.sigma * fbm[[p, i]];
        spots[[p, i]] = self.s0 * log_s.exp();
      }
    }
    spots
  }
}

impl<T: FloatExt + RoughSimd, S: SeedExt> ProcessExt<T> for RlBlackScholes<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = RlBlackScholesSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> RlBlackScholesSampler<T, S> {
    let horizon = self.t.unwrap_or(T::one());
    RlBlackScholesSampler {
      n: self.n,
      s0: self.s0,
      r: self.r,
      sigma: self.sigma,
      dt: horizon / T::from_usize_(self.n - 1),
      two_h: T::from_f64_fast(2.0) * self.hurst,
      half_sigma_sq: T::from_f64_fast(0.5) * self.sigma * self.sigma,
      gn: Gn::<T, S> {
        n: self.n - 1,
        t: self.t,
        seed: self.seed.derive(),
      },
      markov: self.fbm.markov().clone(),
    }
  }
}

/// Reusable [`RlBlackScholes`] sampling state: owns the cloned RL-fBM
/// [`MarkovLift`] stepper and the Gaussian-increment source, plus the
/// precomputed log-price scalars. `fill_path` regenerates the RL-fBM driver and
/// applies the closed-form fractional Black–Scholes log-price pointwise; the
/// owned `Gn` stream advances each call for independent paths.
#[doc(hidden)]
pub struct RlBlackScholesSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  s0: T,
  r: T,
  sigma: T,
  dt: T,
  two_h: T,
  half_sigma_sq: T,
  gn: Gn<T, S>,
  markov: MarkovLift<T>,
}

impl<T: FloatExt + RoughSimd, S: SeedExt> RlBlackScholesSampler<T, S> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    let dw = self.gn.sample();
    let fbm = self.markov.simulate(
      T::zero(),
      |_| T::zero(),
      |_| T::one(),
      dw.as_slice().expect("dw contiguous"),
    );

    out[0] = self.s0;
    for i in 1..out.len() {
      let t_i = self.dt * T::from_usize_(i);
      let log_s = self.r * t_i - self.half_sigma_sq * t_i.powf(self.two_h) + self.sigma * fbm[i];
      out[i] = self.s0 * log_s.exp();
    }
  }
}

impl<T: FloatExt + RoughSimd, S: SeedExt> PathSampler<T> for RlBlackScholesSampler<T, S> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("RlBlackScholes output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

#[cfg(test)]
mod tests {
  use stochastic_rs_core::simd_rng::Deterministic;
  use stochastic_rs_core::simd_rng::Unseeded;

  use super::RlBlackScholes;
  use crate::traits::ProcessExt;

  #[test]
  fn positive_path_and_correct_initial_value() {
    let p = RlBlackScholes::new(
      0.3_f64,
      100.0,
      0.05,
      0.2,
      200,
      Some(0.5),
      None,
      Deterministic::new(2),
    );
    let s = p.sample();
    assert_eq!(s[0], 100.0);
    assert!(s.iter().all(|v| *v > 0.0 && v.is_finite()));
  }

  /// With $\sigma = 0$ the process reduces to the deterministic compounding
  /// $S_t = S_0 e^{rt}$; check that the path matches term by term.
  #[test]
  fn zero_vol_matches_deterministic_compounding() {
    let s0 = 300.0;
    let r = 0.05;
    let n = 128;
    let t = 1.0;
    let p = RlBlackScholes::<f64>::new(0.2, s0, r, 0.0, n, Some(t), None, Unseeded);
    let s = p.sample();
    let dt = t / (n as f64 - 1.0);
    for i in 0..n {
      let expected = s0 * (r * dt * i as f64).exp();
      let rel = (s[i] - expected).abs() / expected;
      assert!(rel < 1e-12, "i={i} got={} expected={expected}", s[i]);
    }
  }

  /// The expected value of the terminal price under fBS is
  /// $\mathbb{E}[S_T] = S_0 e^{rT}$ (Hu–Øksendal 2003 Cor. 4.4). Monte Carlo.
  #[test]
  fn terminal_expectation_matches_risk_neutral_forward() {
    let s0 = 100.0_f64;
    let r = 0.05_f64;
    let t = 0.5_f64;
    let sigma = 0.2_f64;
    let samples = 4_000_usize;
    let mut sum = 0.0_f64;
    for k in 0..samples {
      let p = RlBlackScholes::new(
        0.25_f64,
        s0,
        r,
        sigma,
        128,
        Some(t),
        Some(35),
        Deterministic::new(10_000 + k as u64),
      );
      sum += *p.sample().last().unwrap();
    }
    let empirical = sum / samples as f64;
    let expected = s0 * (r * t).exp();
    let rel = (empirical - expected).abs() / expected;
    assert!(
      rel < 0.05,
      "empirical={empirical} expected={expected} rel={rel}"
    );
  }
}
