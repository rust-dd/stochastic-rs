//! # Fractional Ou driven by Riemann–Liouville fBM (RFSV log-volatility)
//!
//! $$
//! dX_t = \kappa(\mu - X_t)\,dt + \nu\,dW^H_t,\qquad H \in (0, 1/2)
//! $$
//!
//! The RFSV log-volatility dynamics of Gatheral, Jaisson & Rosenbaum (2018).
//! The noise is RL-fBM generated non-cumulatively via the Bilokon–Wong
//! modified fast algorithm; the Ou drift is integrated with the usual Euler
//! rule on the resulting increments $\delta W^H_n = W^H_{t_{n+1}} - W^H_{t_n}$.
//!
//! Reference: Bayer C., Friz P., Gatheral J. *Pricing under rough volatility*.
//! Quantitative Finance 16 (2016), 887–904.
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

/// Fractional Ornstein–Uhlenbeck driven by RL-fBM noise.
#[derive(Clone)]
pub struct RlFOU<T: FloatExt, S: SeedExt = Unseeded> {
  /// Hurst exponent of the driving fBM.
  pub hurst: T,
  /// Mean-reversion speed $\kappa$.
  pub kappa: T,
  /// Long-run mean $\mu$.
  pub mu: T,
  /// Diffusion scale $\nu$.
  pub sigma: T,
  /// Number of simulation points.
  pub n: usize,
  /// Initial value $X_0$.
  pub x0: Option<T>,
  /// Simulation horizon $T$.
  pub t: Option<T>,
  /// Quadrature degree passed through to the underlying [`RlFBm`].
  pub degree: Option<usize>,
  /// Seed strategy.
  pub seed: S,
  fbm: RlFBm<T>,
}

impl<T: FloatExt, S: SeedExt> RlFOU<T, S> {
  #[must_use]
  pub fn new(
    hurst: T,
    kappa: T,
    mu: T,
    sigma: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    degree: Option<usize>,
    seed: S,
  ) -> Self {
    assert!(n >= 2, "n must be at least 2");
    Self {
      hurst,
      kappa,
      mu,
      sigma,
      n,
      x0,
      t,
      degree,
      seed,
      fbm: RlFBm::new(hurst, n, t, degree, Unseeded),
    }
  }
}

impl<T: FloatExt + RoughSimd, S: SeedExt> RlFOU<T, S> {
  /// Generate $m$ independent RFSV log-volatility paths as an $(m, n)$ array.
  /// The RL-fBM noise is generated in a single batch via
  /// [`RlFBm::sample_batch`], then each path is Euler-integrated independently.
  pub fn sample_batch(&self, m: usize) -> Array2<T> {
    let fbm = self.fbm.sample_batch_impl(&self.seed.derive(), m);
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);
    let x0 = self.x0.unwrap_or(T::zero());
    let mut out = Array2::<T>::zeros((m, self.n));

    for p in 0..m {
      out[[p, 0]] = x0;
      for i in 1..self.n {
        let dfbm = fbm[[p, i]] - fbm[[p, i - 1]];
        out[[p, i]] =
          out[[p, i - 1]] + self.kappa * (self.mu - out[[p, i - 1]]) * dt + self.sigma * dfbm;
      }
    }
    out
  }
}

impl<T: FloatExt + RoughSimd, S: SeedExt> ProcessExt<T> for RlFOU<T, S> {
  type Output = Array1<T>;
  type Sampler<'s>
    = RlFOUSampler<T, S>
  where
    Self: 's;

  fn sampler(&self) -> RlFOUSampler<T, S> {
    RlFOUSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      kappa: self.kappa,
      mu: self.mu,
      sigma: self.sigma,
      dt: self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1),
      gn: Gn::<T, S> {
        n: self.n - 1,
        t: self.t,
        seed: self.seed.derive(),
      },
      markov: self.fbm.markov().clone(),
    }
  }
}

/// Reusable [`RlFOU`] sampling state: owns the cloned RL-fBM [`MarkovLift`]
/// stepper and the Gaussian-increment source, plus the precomputed Euler
/// scalars. `fill_path` regenerates the RL-fBM driver and Euler-integrates the
/// Ou drift in place; the owned `Gn` stream advances each call for independent
/// paths.
#[doc(hidden)]
pub struct RlFOUSampler<T: FloatExt, S: SeedExt> {
  n: usize,
  x0: T,
  kappa: T,
  mu: T,
  sigma: T,
  dt: T,
  gn: Gn<T, S>,
  markov: MarkovLift<T>,
}

impl<T: FloatExt + RoughSimd, S: SeedExt> RlFOUSampler<T, S> {
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

    out[0] = self.x0;
    for i in 1..out.len() {
      let dfbm = fbm[i] - fbm[i - 1];
      out[i] = out[i - 1] + self.kappa * (self.mu - out[i - 1]) * self.dt + self.sigma * dfbm;
    }
  }
}

impl<T: FloatExt + RoughSimd, S: SeedExt> PathSampler<T> for RlFOUSampler<T, S> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("RlFOU output must be contiguous");
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

  use super::RlFOU;
  use crate::traits::ProcessExt;

  #[test]
  fn fou_sigma_zero_matches_deterministic_euler() {
    let kappa = 1.3_f64;
    let mu = 0.8_f64;
    let n = 129;
    let x0 = 0.2_f64;
    let t = 1.0_f64;

    let p = RlFOU::<f64>::new(0.3, kappa, mu, 0.0, n, Some(x0), Some(t), None, Unseeded);
    let x = p.sample();

    let dt = t / (n as f64 - 1.0);
    let mut expected = x0;
    for i in 1..n {
      expected = expected + kappa * (mu - expected) * dt;
      assert!((x[i] - expected).abs() < 1e-12, "mismatch at {i}");
    }
  }

  #[test]
  fn finite_output_at_typical_rfsv_parameters() {
    let p = RlFOU::new(
      0.1_f64,
      2.0,
      0.15_f64.ln(),
      0.25,
      512,
      Some(0.15_f64.ln()),
      Some(1.0),
      None,
      Deterministic::new(7),
    );
    let x = p.sample();
    assert_eq!(x.len(), 512);
    assert!(x.iter().all(|v| v.is_finite()));
  }
}
