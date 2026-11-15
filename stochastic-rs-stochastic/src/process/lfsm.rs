//! # Lfsm
//!
//! $$
//! X_t=\int_{\mathbb{R}}\left(\max(t-u,0)^{H-1/\alpha}-\max(-u,0)^{H-1/\alpha}\right)\,dL_u^{\alpha}
//! $$
//!
//! Reference: Stoev, S., Taqqu, M. S. (2004), "Simulation methods for linear fractional stable motion and FARIMA using the Fast Fourier Transform", Fractals 12, 95–121, DOI 10.1142/S0218348X04002379.

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::DistributionSampler;
use stochastic_rs_distributions::Seeded;
use stochastic_rs_distributions::SimdDistribution;
use stochastic_rs_distributions::alpha_stable::SimdAlphaStable;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Linear fractional stable motion (Lfsm), also commonly referred to as
/// Levy fractional stable motion in discretized form.
///
/// The implementation uses a one-sided moving-average discretization with
/// alpha-stable innovations:
///
/// `X_i = X_{i-1} + sum_{k=0}^{i-1} w_k * xi_{i-1-k}`,
/// where `w_k = dt^d * ((k+1)^d - k^d)` and `d = H - 1/alpha`.
#[derive(Clone)]
pub struct Lfsm<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Stability index of the Levy-stable driver (`0 < alpha <= 2`).
  /// Smaller values produce heavier tails and larger jumps.
  pub alpha: T,
  /// Skewness of the stable innovations (`-1 <= beta <= 1`).
  /// Controls left/right jump asymmetry.
  pub beta: T,
  /// Self-similarity / roughness parameter (`1/alpha < H < 1` here).
  /// Governs long-memory strength in the fractional kernel.
  pub hurst: T,
  /// Scale of the stable noise term.
  /// Larger values increase path variability.
  pub scale: T,
  /// Number of grid points along the path.
  pub n: usize,
  /// Initial process value.
  pub x0: Option<T>,
  /// Total simulated time horizon (defaults to `1` if `None`).
  pub t: Option<T>,
  /// Seed strategy (compile-time: [`Unseeded`] or the [`Deterministic` seed](stochastic_rs_core::simd_rng::Deterministic)).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> Lfsm<T, S> {
  pub fn new(
    alpha: T,
    beta: T,
    hurst: T,
    scale: T,
    n: usize,
    x0: Option<T>,
    t: Option<T>,
    seed: S,
  ) -> Self {
    assert!(
      alpha.is_finite(),
      "alpha must satisfy `alpha.is_finite()`, got alpha = {alpha:?}"
    );
    assert!(
      beta.is_finite(),
      "beta must satisfy `beta.is_finite()`, got beta = {beta:?}"
    );
    assert!(
      hurst.is_finite(),
      "hurst must satisfy `hurst.is_finite()`, got hurst = {hurst:?}"
    );
    assert!(
      scale.is_finite(),
      "scale must satisfy `scale.is_finite()`, got scale = {scale:?}"
    );
    assert!(n >= 2, "n must satisfy `n >= 2`, got n = {n:?}");
    assert!(
      x0.is_none_or(|v| v.is_finite()),
      "x0 must satisfy `x0.is_none_or(|v| v.is_finite())`, got x0 = {x0:?}"
    );
    assert!(
      t.is_none_or(|t| t > T::zero() && t.is_finite()),
      "t must satisfy `t.is_none_or(|t| t > T::zero() && t.is_finite())`, got t = {t:?}"
    );
    assert!(
      alpha > T::zero() && alpha <= T::from(2.0).unwrap(),
      "alpha must satisfy `alpha > T::zero() && alpha <= T::from(2.0).unwrap()`, got alpha = {alpha:?}"
    );
    assert!(
      (-T::one()..=T::one()).contains(&beta),
      "beta must satisfy `(-T::one()..=T::one()).contains(&beta)`, got beta = {beta:?}"
    );
    assert!(
      scale > T::zero(),
      "scale must satisfy `scale > T::zero()`, got scale = {scale:?}"
    );
    assert!(
      hurst > T::one() / alpha && hurst < T::one(),
      "hurst must satisfy `hurst > T::one() / alpha && hurst < T::one()`, got hurst = {hurst:?}, alpha = {alpha:?}"
    );
    Self {
      backend: Cpu,
      alpha,
      beta,
      hurst,
      scale,
      n,
      x0,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> Lfsm<T, S, B> {}

impl<T: FloatExt, S: SeedExt, B> Lfsm<T, S, B> {
  #[inline]
  fn dt(&self) -> T {
    self.t.unwrap_or(T::one()) / T::from_usize_(self.n.saturating_sub(1).max(1))
  }

  /// The weights `dt^d ((k + 1)^d − k^d)` of lags `0..len`, `d = H − 1/α`, formed in `f64`, where
  /// the difference of nearly equal powers keeps its digits.
  fn kernel_weights(&self, len: usize) -> Vec<T> {
    let d = self.hurst.to_f64().unwrap() - 1.0 / self.alpha.to_f64().unwrap();
    let kernel_scale = self.dt().to_f64().unwrap().powf(d);
    (0..len)
      .map(|k| {
        let kf = k as f64;
        T::from_f64_fast(kernel_scale * ((kf + 1.0).powf(d) - kf.powf(d)))
      })
      .collect()
  }

  fn innovation_scale(&self) -> T {
    let dt = self.dt().to_f64().unwrap();
    T::from_f64_fast(self.scale.to_f64().unwrap() * dt.powf(1.0 / self.alpha.to_f64().unwrap()))
  }
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for Lfsm<T, S, B>
{
  /// The Chambers–Mallows–Stuck constants of the host's stable generator for
  /// `α ≠ 1`, with the innovation scale `scale · dt^{1/α}` folded into `S`.
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    let pi = T::from_f64_fast(std::f64::consts::PI);
    let two = T::from_usize_(2);
    let beta_tan = self.beta * (pi * self.alpha / two).tan();
    let s = (T::one() + beta_tan * beta_tan).powf(T::one() / (two * self.alpha));
    crate::euler::EulerSpec::LinearFractionalStable {
      alpha: self.alpha,
      inv_alpha: T::one() / self.alpha,
      tail_exp: (T::one() - self.alpha) / self.alpha,
      b: beta_tan.atan() / self.alpha,
      scale_s: self.innovation_scale() * s,
      pi,
      half_pi: pi / two,
    }
  }

  fn initial_value(&self) -> T {
    self.x0.unwrap_or(T::zero())
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn time_step(&self) -> T {
    self.dt()
  }

  /// The moving-average weights `dt^d ((k + 1)^d − k^d)`, `d = H − 1/α`, one
  /// per lag, exactly as the host tabulates them.
  fn curves(&self) -> Option<Vec<Vec<T>>> {
    Some(vec![self.kernel_weights(self.n)])
  }

  fn device_seed(&self) -> u64 {
    crate::euler::draw_seed(&self.seed)
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] Lfsm<T, S> { alpha, beta, hurst, scale, n, x0, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::traits::Sealed
  for Lfsm<T, S, B>
{
}

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T> for Lfsm<T, S, B> {
  type Output = Array1<T>;
  type Sampler<'s>
    = LfsmSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> LfsmSampler<T> {
    let stable =
      SimdAlphaStable::<T>::new(self.alpha, self.beta, self.innovation_scale(), T::zero())
        .seeded(&self.seed);
    LfsmSampler {
      n: self.n,
      x0: self.x0.unwrap_or(T::zero()),
      stable,
      weights: Array1::from(self.kernel_weights(self.n.saturating_sub(1))),
    }
  }

  /// Through the Euler engine when the grid fits the kernels'
  /// per-path history and `α ≠ 1`; a longer grid keeps the process on the host,
  /// chunked exactly as [`ProcessExt`] chunks.
  fn sample(&self) -> Array1<T> {
    if self.device_ready() {
      self.backend.euler_sample(self)
    } else {
      let out = self.sampler().sample();
      self.advance_chunk_seed();
      out
    }
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, f)
    }
  }

  fn sample_map_view<R: Send>(
    &self,
    m: usize,
    f: impl Fn(ndarray::ArrayView1<T>) -> R + Sync,
  ) -> Vec<R> {
    if self.device_ready() {
      self.backend.euler_paths_map_view(self, m, f)
    } else {
      crate::traits::process::sample_map_chunked(self, m, |path| f(path.view()))
    }
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    if self.device_ready() {
      self.backend.euler_paths(self, m)
    } else {
      crate::traits::process::sample_par_chunked(self, m)
    }
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_sample(self)
    } else {
      Ok(<Self as ProcessExt<T>>::sample(self))
    }
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    if self.device_ready() {
      self.backend.try_euler_paths(self, m)
    } else {
      Ok(<Self as ProcessExt<T>>::sample_par(self, m))
    }
  }

  /// What a device runs here: the grid fits the kernels'
  /// per-path history and the stability index is not one, whose
  /// Chambers–Mallows–Stuck draw is a different formula the kernels do not
  /// carry. Anything else samples on the host.
  fn device_fallback(&self) -> Option<&'static str> {
    (!(self.n <= crate::euler::HISTORY_SLOTS && self.alpha != T::one()))
      .then_some("a grid past the kernels' per-path history, or a unit stability index")
  }
}

/// Reusable [`Lfsm`] sampling state: the owned alpha-stable innovation source
/// and the precomputed fractional moving-average weights.
#[doc(hidden)]
pub struct LfsmSampler<T: FloatExt> {
  n: usize,
  x0: T,
  stable: Seeded<SimdAlphaStable<T>>,
  weights: Array1<T>,
}

impl<T: FloatExt> LfsmSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    // Legacy behaviour: the degenerate `n <= 1` path returns zeros without
    // writing `x0` (the `if n <= 1` guard preceded the `x[0] = x0` line).
    if out.len() == 1 {
      out[0] = T::zero();
      return;
    }
    out[0] = self.x0;

    let mut innovations = Array1::<T>::zeros(out.len() - 1);
    self.stable.fill_slice(innovations.as_slice_mut().unwrap());

    let mut prev = self.x0;
    for i in 1..out.len() {
      let mut inc = T::zero();
      for k in 0..i {
        inc += self.weights[k] * innovations[i - 1 - k];
      }
      prev += inc;
      out[i] = prev;
    }
  }
}

impl<T: FloatExt> crate::traits::Sealed for LfsmSampler<T> {}

impl<T: FloatExt> PathSampler<T> for LfsmSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out.as_slice_mut().expect("Lfsm output must be contiguous");
    self.fill_path(slice);
  }

  fn sample(&mut self) -> Array1<T> {
    let n = self.n;
    array1_from_fill(n, |out| self.fill_path(out))
  }
}

py_process_1d!(PyLfsm, Lfsm,
  sig: (alpha, beta, hurst, scale, n, x0=None, t=None, seed=None, dtype=None),
  params: (alpha: f64, beta: f64, hurst: f64, scale: f64, n: usize, x0: Option<f64>, t: Option<f64>),
  device
);

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn lfsm_path_is_finite() {
    let p = Lfsm::new(1.7_f64, 0.0, 0.8, 1.0, 256, Some(0.0), Some(1.0), Unseeded);
    let x = p.sample();
    assert_eq!(x.len(), 256);
    assert!(x.iter().all(|v| v.is_finite()));
  }

  #[test]
  fn the_kernel_weights_keep_their_digits_in_single_precision() {
    let p = Lfsm::<f32, Unseeded>::new(1.5, 0.0, 0.7, 1.0, 1025, None, Some(1.0), Unseeded);
    let w = p.kernel_weights(1001);
    let exact = [
      (1usize, 0.018_551_870_37_f64),
      (50, 5.970_818_98e-4),
      (500, 6.502_967_821e-5),
      (1000, 3.329_090_422e-5),
    ];
    for (k, e) in exact {
      assert!(
        (w[k] as f64 / e - 1.0).abs() < 1e-6,
        "k = {k}: got {}, exact {e}",
        w[k]
      );
    }
  }
}
