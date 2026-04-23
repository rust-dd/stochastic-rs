//! # Riemann–Liouville fractional Brownian motion
//!
//! $$
//! W^H_t = \frac{1}{\Gamma(H+1/2)}\int_0^t (t-s)^{H-1/2}\,dW_s,\qquad H\in(0, 1/2)
//! $$
//!
//! Direct (non-cumulative) path generation via the Bilokon–Wong modified fast
//! algorithm: the power-law kernel is replaced by a sum of $N' \approx \log n$
//! exponentials, and the whole path is built as the superposition of
//! independent OU-like Markov factors driven by the same Brownian motion.
//!
//! Not to be confused with Mandelbrot–Van Ness fBM: the two share the covariance
//! structure asymptotically, but RL-fBM has non-stationary increments near
//! $t = 0$. For classical stationary fGn / MVN-fBM use [`FGN`](crate::stochastic::noise::fgn::FGN).
//!
//! Reference: Bilokon & Wong (2026), doi:10.1017/jpr.2025.10071.
use ndarray::Array1;

use super::kernel::RlKernel;
use super::markov_lift::MarkovLift;
use super::markov_lift::RoughSimd;
use crate::simd_rng::Deterministic;
use crate::simd_rng::SeedExt;
use crate::simd_rng::Unseeded;
use crate::stochastic::noise::gn::Gn;
use crate::traits::FloatExt;
use crate::traits::ProcessExt;

/// Riemann–Liouville fractional Brownian motion, Hurst $H \in (0, 1/2)$.
#[derive(Clone)]
pub struct RlFBm<T: FloatExt, S: SeedExt = Unseeded> {
  /// Hurst exponent.
  pub hurst: T,
  /// Number of discrete simulation points.
  pub n: usize,
  /// Total simulation horizon (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Quadrature degree $N'$ (defaults to [`RlKernel::default_degree`] of `n`).
  pub degree: Option<usize>,
  /// Seed strategy.
  pub seed: S,
}

impl<T: FloatExt> RlFBm<T> {
  /// Build an RL-fBM generator for Hurst $H$ on an $n$-point grid.
  #[must_use]
  pub fn new(hurst: T, n: usize, t: Option<T>, degree: Option<usize>) -> Self {
    assert!(n >= 2, "n must be at least 2");
    Self {
      hurst,
      n,
      t,
      degree,
      seed: Unseeded,
    }
  }
}

impl<T: FloatExt> RlFBm<T, Deterministic> {
  /// Same as [`new`](Self::new) with a fixed seed.
  #[must_use]
  pub fn seeded(hurst: T, n: usize, t: Option<T>, degree: Option<usize>, seed: u64) -> Self {
    assert!(n >= 2, "n must be at least 2");
    Self {
      hurst,
      n,
      t,
      degree,
      seed: Deterministic(seed),
    }
  }
}

impl<T: FloatExt + RoughSimd, S: SeedExt> RlFBm<T, S> {
  /// Core sampler — monomorphised per caller-provided seed strategy so
  /// downstream processes (e.g. [`RlFOU`](super::rl_fou::RlFOU)) can derive a
  /// child seed and pass it in without cloning or rebuilding `self`.
  pub(crate) fn sample_impl<S2: SeedExt>(&self, seed: S2) -> Array1<T> {
    let dt = self.t.unwrap_or(T::one()) / T::from_usize_(self.n - 1);
    let degree = self
      .degree
      .unwrap_or_else(|| RlKernel::<T>::default_degree(self.n));
    let kernel = RlKernel::<T>::new(self.hurst, degree);
    let step = MarkovLift::new(kernel, dt);

    let gn = Gn::<T, S2> {
      n: self.n - 1,
      t: self.t,
      seed,
    };
    let dw = gn.sample();

    step.simulate(
      T::zero(),
      |_| T::zero(),
      |_| T::one(),
      dw.as_slice().expect("dw must be contiguous"),
    )
  }
}

impl<T: FloatExt + RoughSimd, S: SeedExt> ProcessExt<T> for RlFBm<T, S> {
  type Output = Array1<T>;

  fn sample(&self) -> Self::Output {
    self.sample_impl(self.seed)
  }
}

#[cfg(test)]
mod tests {
  use super::RlFBm;
  use crate::traits::ProcessExt;

  #[test]
  #[should_panic(expected = "n must be at least 2")]
  fn rejects_too_short_grid() {
    let _ = RlFBm::<f64>::new(0.3, 1, Some(1.0), None);
  }

  #[test]
  fn starts_at_zero_and_is_finite() {
    let p = RlFBm::seeded(0.25_f64, 256, Some(1.0), None, 42);
    let path = p.sample();
    assert_eq!(path.len(), 256);
    assert_eq!(path[0], 0.0);
    assert!(path.iter().all(|v| v.is_finite()));
  }

  /// Variance of RL-fBM at $t$ is $t^{2H}/(2H\,\Gamma(H+1/2)^2)$ (Lim 2001).
  /// With enough Monte Carlo samples and kernel degree the empirical variance
  /// at $t=T$ should match to within a few percent.
  #[test]
  fn variance_at_horizon_matches_theory_within_mc_error() {
    use statrs::function::gamma::gamma;
    let hurst = 0.2_f64;
    let n = 512;
    let t = 1.0_f64;
    let samples = 2_000_usize;

    let mut endpoints: Vec<f64> = Vec::with_capacity(samples);
    for k in 0..samples {
      let p = RlFBm::seeded(hurst, n, Some(t), Some(40), 1_000 + k as u64);
      endpoints.push(*p.sample().last().unwrap());
    }
    let mean: f64 = endpoints.iter().sum::<f64>() / samples as f64;
    let var: f64 = endpoints.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / samples as f64;

    let g = gamma(hurst + 0.5);
    let theoretical = t.powf(2.0 * hurst) / (2.0 * hurst * g * g);
    let rel = (var - theoretical).abs() / theoretical;
    assert!(
      rel < 0.15,
      "empirical var={var} theoretical={theoretical} rel={rel}"
    );
  }
}
