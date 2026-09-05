//! Van Emmerich / Jacobi-type stochastic correlation (Eq. 15).
//!
//! $$
//! d\rho_t = \kappa(\mu - \rho_t)\,dt + \sigma\sqrt{1 - \rho_t^2}\,dW_t
//! $$

use ndarray::Array1;
use stochastic_rs_core::simd_rng::SeedExt;
use stochastic_rs_core::simd_rng::Unseeded;
use stochastic_rs_distributions::normal::SimdNormal;

use crate::buffer::array1_from_fill;
use crate::device::Cpu;
use crate::traits::FloatExt;
use crate::traits::PathSampler;
use crate::traits::ProcessExt;

/// Van Emmerich stochastic correlation process (Eq. 15 in Teng et al. 2016).
///
/// The diffusion coefficient vanishes *linearly* at ±1, keeping the
/// process inside (−1, 1) when κ ≥ σ²/(1 ± μ).
#[derive(Debug, Clone)]
pub struct VanEmmerich<T: FloatExt, S: SeedExt = Unseeded, B = Cpu> {
  /// Mean-reversion speed κ of the correlation process itself (ρ_t is
  /// the state being simulated directly, unlike
  /// [`TransformedOU`](crate::correlation::TransformedOU)'s X-space
  /// indirection).
  pub kappa: T,
  /// Long-run correlation level μ ∈ (−1, 1) that ρ_t reverts toward.
  pub mu: T,
  /// Diffusion scale σ multiplying the linearly-vanishing
  /// `√(1−ρ_t²) dW_t` term.
  pub sigma: T,
  /// Initial correlation ρ₀ ∈ (−1, 1).
  pub rho0: T,
  /// Number of points sampled along the correlation path.
  pub n: usize,
  /// Simulation horizon [0, t] for the path (defaults to 1 when omitted).
  pub t: Option<T>,
  /// Seed strategy (compile-time: `Unseeded` or `Deterministic`).
  pub seed: S,
  /// The sampling backend: [`Cpu`] by default, a device handle after
  /// [`on`](Self::on).
  pub backend: B,
}

impl<T: FloatExt, S: SeedExt> VanEmmerich<T, S> {
  pub fn new(kappa: T, mu: T, sigma: T, rho0: T, n: usize, t: Option<T>, seed: S) -> Self {
    Self {
      backend: Cpu,
      kappa,
      mu,
      sigma,
      rho0,
      n,
      t,
      seed,
    }
  }
}

impl<T: FloatExt, S: SeedExt, B> VanEmmerich<T, S, B> {}

/// The Euler engine's view of the Van Emmerich stochastic correlation
/// process.
impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> crate::euler::EulerCoefficients<T>
  for VanEmmerich<T, S, B>
{
  fn euler_spec(&self) -> crate::euler::EulerSpec<T> {
    crate::euler::EulerSpec::BoundedCorrelation {
      kappa: self.kappa,
      mu: self.mu,
      sigma: self.sigma,
    }
  }

  fn initial_value(&self) -> T {
    self.rho0
  }

  fn grid_points(&self) -> usize {
    self.n
  }

  fn horizon(&self) -> T {
    self.t.unwrap_or(T::one())
  }

  fn device_seed(&self) -> u64 {
    rand::Rng::random(&mut self.seed.rng())
  }

  fn host_sample(&self) -> Array1<T> {
    let out = <Self as ProcessExt<T>>::sampler(self).sample();
    <Self as ProcessExt<T>>::advance_chunk_seed(self);
    out
  }
}

backend_switch!([T: FloatExt, S: SeedExt] VanEmmerich<T, S> { kappa, mu, sigma, rho0, n, t, seed } via euler);

impl<T: FloatExt, S: SeedExt, B: crate::euler::EulerBackend<T>> ProcessExt<T>
  for VanEmmerich<T, S, B>
{
  type Output = Array1<T>;
  type Sampler<'s>
    = VanEmmerichSampler<T>
  where
    Self: 's;

  fn sampler(&self) -> VanEmmerichSampler<T> {
    let n_steps = self.n.saturating_sub(1);
    let dt = if n_steps > 0 {
      self.t.unwrap_or(T::one()) / T::from_usize_(n_steps)
    } else {
      T::zero()
    };
    VanEmmerichSampler {
      n: self.n,
      kappa: self.kappa,
      mu: self.mu,
      sigma: self.sigma,
      rho0: self.rho0,
      dt,
      normal: SimdNormal::<T>::new(T::zero(), dt.sqrt(), &self.seed),
    }
  }

  /// Through the Euler engine: on a device the recursion runs in the kernel,
  /// on the host devices it is this process's own sampler, chunked exactly as
  /// `ProcessExt` chunks.
  fn sample(&self) -> Array1<T> {
    self.backend.euler_sample(self)
  }

  fn sample_map<R: Send>(&self, m: usize, f: impl Fn(&Array1<T>) -> R + Sync) -> Vec<R> {
    self.backend.euler_paths_map(self, m, f)
  }

  fn sample_par(&self, m: usize) -> Vec<Array1<T>> {
    self.backend.euler_paths(self, m)
  }

  fn try_sample(&self) -> Result<Array1<T>, crate::device::DeviceError> {
    self.backend.try_sample(self)
  }

  fn try_sample_par(&self, m: usize) -> Result<Vec<Array1<T>>, crate::device::DeviceError> {
    self.backend.try_euler_paths(self, m)
  }
}

/// Reusable [`VanEmmerich`] sampling state: owns the Gaussian source and the
/// precomputed step size. `fill_path` Euler-steps the Jacobi-type diffusion
/// with the linear `√(1−ρ²)` coefficient and clamps to (−1, 1) in place; the
/// owned source advances each call for independent paths.
#[doc(hidden)]
pub struct VanEmmerichSampler<T: FloatExt> {
  n: usize,
  kappa: T,
  mu: T,
  sigma: T,
  rho0: T,
  dt: T,
  normal: SimdNormal<T>,
}

impl<T: FloatExt> VanEmmerichSampler<T> {
  fn fill_path(&mut self, out: &mut [T]) {
    if out.is_empty() {
      return;
    }
    let n_steps = out.len() - 1;
    let mut gn = Array1::<T>::zeros(n_steps);
    if let Some(slice) = gn.as_slice_mut() {
      self.normal.fill_slice(slice);
    }

    out[0] = self.rho0;
    let clamp_lo = T::from_f64_fast(-0.9999);
    let clamp_hi = T::from_f64_fast(0.9999);

    for i in 1..out.len() {
      let r = out[i - 1];
      let one_minus_r2 = (T::one() - r * r).max(T::zero());

      let drift = self.kappa * (self.mu - r) * self.dt;
      let diffusion = self.sigma * one_minus_r2.sqrt() * gn[i - 1];

      out[i] = (r + drift + diffusion).clamp(clamp_lo, clamp_hi);
    }
  }
}

impl<T: FloatExt> PathSampler<T> for VanEmmerichSampler<T> {
  type Output = Array1<T>;

  fn sample_into(&mut self, out: &mut Array1<T>) {
    let slice = out
      .as_slice_mut()
      .expect("VanEmmerich output must be contiguous");
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

  use super::*;

  #[test]
  fn stays_bounded() {
    let scp = VanEmmerich::new(
      5.0_f64,
      -0.3,
      0.8,
      -0.3,
      1000,
      Some(1.0),
      Deterministic::new(42),
    );
    let path = scp.sample();
    assert_eq!(path.len(), 1000);
    assert!(path.iter().all(|&r| r > -1.0 && r < 1.0));
  }
}
