//! # Riemann–Liouville fractional Brownian motion
//!
//! $$
//! W^H_t = \frac{1}{\Gamma(H+1/2)}\int_0^t (t-s)^{H-1/2}\,dW_s,\qquad H\in(0, 1/2)
//! $$
//!
//! Direct (non-cumulative) path generation via the Bilokon–Wong modified fast
//! algorithm: the power-law kernel is replaced by a sum of $N' \approx \log n$
//! exponentials, and the whole path is built as the superposition of
//! independent Ou-like Markov factors driven by the same Brownian motion.
//!
//! The [`MarkovLift`] stepper (with the kernel quadrature) is built once per
//! struct in [`new`]/[`seeded`] and reused across every [`sample`] and
//! [`sample_batch`] call — repeated sampling does not re-run Golub–Welsch
//! or recompute per-$\delta t$ factors.
//!
//! Not to be confused with Mandelbrot–Van Ness fBM: the two share the
//! covariance structure asymptotically, but RL-fBM has non-stationary
//! increments near $t = 0$. For classical stationary fGn / MVN-fBM use
//! [`Fgn`](crate::noise::fgn::Fgn).
//!
//! Reference: Bilokon & Wong (2026), doi:10.1017/jpr.2025.10071.
use ndarray::Array1;
use ndarray::Array2;
use stochastic_rs_core::simd_rng::Deterministic;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;

use super::kernel::RlKernel;
use super::markov_lift::MarkovLift;
use super::markov_lift::RoughSimd;
use crate::noise::gn::Gn;
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
  markov: MarkovLift<T>,
}

fn build_markov<T: FloatExt>(
  hurst: T,
  n: usize,
  t: Option<T>,
  degree: Option<usize>,
) -> MarkovLift<T> {
  let dt = t.unwrap_or(T::one()) / T::from_usize_(n - 1);
  let deg = degree.unwrap_or_else(|| RlKernel::<T>::default_degree(n));
  let kernel = RlKernel::<T>::new(hurst, deg);
  MarkovLift::new(kernel, dt)
}

impl<T: FloatExt> RlFBm<T> {
  /// Build an RL-fBM generator for Hurst $H$ on an $n$-point grid. The
  /// underlying [`MarkovLift`] is constructed once here and reused by every
  /// sampling call.
  #[must_use]
  pub fn new(hurst: T, n: usize, t: Option<T>, degree: Option<usize>) -> Self {
    assert!(n >= 2, "n must be at least 2");
    let markov = build_markov(hurst, n, t, degree);
    Self {
      hurst,
      n,
      t,
      degree,
      seed: Unseeded,
      markov,
    }
  }
}

impl<T: FloatExt> RlFBm<T, Deterministic> {
  /// Same as [`new`](Self::new) with a fixed seed.
  #[must_use]
  pub fn seeded(hurst: T, n: usize, t: Option<T>, degree: Option<usize>, seed: u64) -> Self {
    assert!(n >= 2, "n must be at least 2");
    let markov = build_markov(hurst, n, t, degree);
    Self {
      hurst,
      n,
      t,
      degree,
      seed: Deterministic(seed),
      markov,
    }
  }
}

impl<T: FloatExt + RoughSimd, S: SeedExt> RlFBm<T, S> {
  /// Core single-path sampler accepting an external seed, used by downstream
  /// processes that derive a child seed for their noise.
  pub(crate) fn sample_impl<S2: SeedExt>(&self, seed: S2) -> Array1<T> {
    let dw = Gn::<T, S2> {
      n: self.n - 1,
      t: self.t,
      seed,
    }
    .sample();
    self.markov.simulate(
      T::zero(),
      |_| T::zero(),
      |_| T::one(),
      dw.as_slice().expect("dw contiguous"),
    )
  }

  /// Generate $m$ independent RL-fBM paths as an $(m, n)$ array. The
  /// underlying Markov-lift runs path-parallel SIMD (cache-tiled), matching
  /// the Python `RoughHestonFast` batching pattern — single-threaded.
  pub fn sample_batch(&self, m: usize) -> Array2<T> {
    let mut seed = self.seed;
    self.sample_batch_impl(seed.derive(), m)
  }

  /// Multi-core version of [`sample_batch`] — rayon parallelises the outer
  /// tile loop so batch-SIMD and core scheduling compound.
  pub fn sample_batch_par(&self, m: usize) -> Array2<T> {
    let mut seed = self.seed;
    self.sample_batch_par_impl(seed.derive(), m)
  }

  pub(crate) fn sample_batch_impl<S2: SeedExt>(&self, mut seed: S2, m: usize) -> Array2<T> {
    let dw = self.build_dw_matrix(&mut seed, m);
    self
      .markov
      .simulate_batch(T::zero(), |_| T::zero(), |_| T::one(), dw.view())
  }

  pub(crate) fn sample_batch_par_impl<S2: SeedExt>(&self, mut seed: S2, m: usize) -> Array2<T> {
    let dw = self.build_dw_matrix(&mut seed, m);
    self
      .markov
      .simulate_batch_par(T::zero(), |_| T::zero(), |_| T::one(), dw.view())
  }

  fn build_dw_matrix<S2: SeedExt>(&self, seed: &mut S2, m: usize) -> Array2<T> {
    let n_minus_1 = self.n - 1;
    let mut dw = Array2::<T>::zeros((m, n_minus_1));
    for p in 0..m {
      let gn = Gn::<T, S2> {
        n: n_minus_1,
        t: self.t,
        seed: seed.derive(),
      };
      let sample = gn.sample();
      dw.row_mut(p).assign(&sample);
    }
    dw
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

  #[test]
  fn batch_shape_and_first_column_is_zero() {
    let p = RlFBm::seeded(0.2_f64, 64, Some(1.0), Some(25), 11);
    let paths = p.sample_batch(17);
    assert_eq!(paths.dim(), (17, 64));
    for row in 0..17 {
      assert_eq!(paths[[row, 0]], 0.0);
    }
    assert!(paths.iter().all(|v| v.is_finite()));
  }

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
